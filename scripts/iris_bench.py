#!/usr/bin/env python3
"""Run the ziggurat audit workload set against an Iris `.iris` slide.

This script intentionally mirrors `wsi_bench` / `openslide_bench` JSON output so
`bench_driver` can compare Iris without adding Python dependencies to the Rust
crate. It requires the optional `Iris-Codec` Python package at runtime.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

SCHEMA_VERSION = 2
RUN_MODE_FULL_SUITE = "full_suite"
RUN_MODE_SINGLE_WORKLOAD = "single_workload"
SELECTED_WORKLOAD_ENV = "WSI_BENCH_ONLY"
REPEAT_INDEX_ENV = "WSI_BENCH_REPEAT_INDEX"


@dataclass(frozen=True)
class WorkloadSpec:
    name: str
    target_n: int
    gate_mode: str = "gating"
    comparability: str = "exact"
    comparability_note: Optional[str] = None


WORKLOAD_SPECS = [
    WorkloadSpec("cold_open", 10),
    WorkloadSpec("single_tile_l0", 200),
    WorkloadSpec("pan_trace_l0", 256),
    WorkloadSpec(
        "pan_trace_l2",
        256,
        comparability_note=(
            "Iris consumes the same accepted level-2 tile top-lefts as ziggurat"
        ),
    ),
    WorkloadSpec("pan_trace_l2_dense", 16),
    WorkloadSpec("region_2k", 30),
    WorkloadSpec("viewport_region_l2", 30),
    WorkloadSpec("thumbnail", 30),
]
WORKLOAD_BY_NAME = {spec.name: spec for spec in WORKLOAD_SPECS}


@dataclass(frozen=True)
class CenteredViewportRegion:
    level2_top_left: tuple[int, int]
    level0_top_left: tuple[int, int]
    side_px: int


@dataclass(frozen=True)
class WorkloadPlan:
    tile_px: int
    center_l0: tuple[int, int]
    level_count: int
    level0_dims: tuple[int, int]
    level2_idx: int
    level2_dims: tuple[int, int]
    pan_step_tiles: int
    pan_steps: int

    @classmethod
    def compute(cls, level_dims: list[tuple[int, int]]) -> "WorkloadPlan":
        if not level_dims:
            raise ValueError("no levels")
        level_count = len(level_dims)
        level0_dims = level_dims[0]
        level2_idx = min(2, level_count - 1)
        level2_dims = level_dims[level2_idx]
        return cls(
            tile_px=256,
            center_l0=(level0_dims[0] // 2, level0_dims[1] // 2),
            level_count=level_count,
            level0_dims=level0_dims,
            level2_idx=level2_idx,
            level2_dims=level2_dims,
            pan_step_tiles=1,
            pan_steps=256,
        )

    def centered_viewport_l2(self, desired_side_px: int) -> CenteredViewportRegion:
        side_px = max(1, min(desired_side_px, self.level2_dims[0], self.level2_dims[1]))
        x_l2 = max(0, (self.level2_dims[0] - side_px) // 2)
        y_l2 = max(0, (self.level2_dims[1] - side_px) // 2)
        dx = self.level0_dims[0] / self.level2_dims[0]
        dy = self.level0_dims[1] / self.level2_dims[1]
        return CenteredViewportRegion(
            level2_top_left=(x_l2, y_l2),
            level0_top_left=(int(x_l2 * dx), int(y_l2 * dy)),
            side_px=side_px,
        )

    def pan_trace_l0(self) -> list[tuple[int, int]]:
        tile_px = self.tile_px
        width, height = self.level0_dims
        return [
            (x, y)
            for (x, y) in self._diagonal_coords(self.center_l0)
            if x >= 0 and y >= 0 and x + tile_px <= width and y + tile_px <= height
        ]

    def pan_trace_l2(self) -> list[tuple[int, int]]:
        tile_px = self.tile_px
        width, height = self.level2_dims
        dx = self.level0_dims[0] / self.level2_dims[0]
        dy = self.level0_dims[1] / self.level2_dims[1]
        out = []
        for x_l0, y_l0 in self._diagonal_coords(self.center_l0):
            x = int(x_l0 / dx)
            y = int(y_l0 / dy)
            if x >= 0 and y >= 0 and x + tile_px <= width and y + tile_px <= height:
                out.append((x, y))
        return out

    def pan_trace_l2_dense(self) -> list[tuple[int, int]]:
        width, height = self.level2_dims
        dx = self.level0_dims[0] / self.level2_dims[0]
        dy = self.level0_dims[1] / self.level2_dims[1]
        cx_l2 = int(self.center_l0[0] / dx)
        cy_l2 = int(self.center_l0[1] / dy)
        base_col = max(0, cx_l2 // self.tile_px - 1)
        base_row = max(0, cy_l2 // self.tile_px - 1)
        out = []
        for row in range(4):
            for col in range(4):
                x = (base_col + col) * self.tile_px
                y = (base_row + row) * self.tile_px
                if (
                    x >= 0
                    and y >= 0
                    and x + self.tile_px <= width
                    and y + self.tile_px <= height
                ):
                    out.append((x, y))
        return out

    def _diagonal_coords(self, center: tuple[int, int]) -> list[tuple[int, int]]:
        step_l0 = self.tile_px * self.pan_step_tiles
        half = self.pan_steps // 2
        return [
            (center[0] + (i - half) * step_l0, center[1] + (i - half) * step_l0)
            for i in range(self.pan_steps)
        ]


@dataclass
class WorkloadResult:
    name: str
    target_n: int
    gate_mode: str
    comparability: str
    comparability_note: Optional[str]
    samples_us: list[int]
    error: Optional[str] = None

    @classmethod
    def new(cls, name: str) -> "WorkloadResult":
        spec = WORKLOAD_BY_NAME.get(name, WorkloadSpec(name, 0))
        return cls(
            name=name,
            target_n=spec.target_n,
            gate_mode=spec.gate_mode,
            comparability=spec.comparability,
            comparability_note=spec.comparability_note,
            samples_us=[],
        )

    @classmethod
    def with_error(cls, name: str, error: str) -> "WorkloadResult":
        result = cls.new(name)
        result.error = error
        return result

    def to_json_obj(self) -> dict:
        out = {
            "name": self.name,
            "target_n": self.target_n,
            "n": len(self.samples_us),
            "gate_mode": self.gate_mode,
            "comparability": self.comparability,
            "comparability_note": self.comparability_note,
            "samples_us": self.samples_us,
            "error": self.error,
        }
        if self.samples_us:
            p = percentiles(self.samples_us)
            out.update(p)
        return out


class IrisSlide:
    def __init__(self, path: Path, codec_module):
        self.path = path
        result = codec_module.validate_slide_path(str(path))
        if not result.success():
            raise RuntimeError(f"validation failed: {result.message}")
        slide = codec_module.open_slide(str(path))
        if not slide:
            raise RuntimeError("open_slide returned null")
        result, info = slide.get_info()
        if not result.success():
            raise RuntimeError(f"get_info failed: {result.message}")

        self._slide = slide
        self._info = info
        self._extent = info.extent
        self._layers = list(info.extent.layers)
        if not self._layers:
            raise RuntimeError("Iris slide reports no layers")
        self._level_to_iris_layer = [
            idx
            for idx, _layer in sorted(
                enumerate(self._layers), key=lambda item: float(item[1].downsample)
            )
        ]
        self.level_dims = [
            self._level_dimensions(self._layers[idx]) for idx in self._level_to_iris_layer
        ]

    def _level_dimensions(self, layer) -> tuple[int, int]:
        return (int(layer.x_tiles) * 256, int(layer.y_tiles) * 256)

    def read_tile_at_top_left(self, level: int, top_left: tuple[int, int]):
        layer_idx = self._level_to_iris_layer[level]
        layer = self._layers[layer_idx]
        col = top_left[0] // 256
        row = top_left[1] // 256
        if col < 0 or row < 0 or col >= int(layer.x_tiles) or row >= int(layer.y_tiles):
            raise IndexError(f"Iris tile out of bounds level={level} col={col} row={row}")
        return self._slide.read_slide_tile(layer_idx, int(col), int(row))

    def read_region(self, level: int, x: int, y: int, width: int, height: int):
        import numpy as np

        level_width, level_height = self.level_dims[level]
        out = np.zeros((height, width, 4), dtype=np.uint8)
        tile_px = 256
        col0 = math.floor(x / tile_px)
        row0 = math.floor(y / tile_px)
        col1 = math.floor((x + width - 1) / tile_px)
        row1 = math.floor((y + height - 1) / tile_px)

        for row in range(row0, row1 + 1):
            for col in range(col0, col1 + 1):
                tile_x = col * tile_px
                tile_y = row * tile_px
                ox0 = max(x, tile_x, 0)
                oy0 = max(y, tile_y, 0)
                ox1 = min(x + width, tile_x + tile_px, level_width)
                oy1 = min(y + height, tile_y + tile_px, level_height)
                if ox0 >= ox1 or oy0 >= oy1:
                    continue

                tile = self.read_tile_at_top_left(level, (tile_x, tile_y))
                dst_x0 = ox0 - x
                dst_y0 = oy0 - y
                src_x0 = ox0 - tile_x
                src_y0 = oy0 - tile_y
                out[dst_y0 : dst_y0 + (oy1 - oy0), dst_x0 : dst_x0 + (ox1 - ox0), :] = tile[
                    src_y0 : src_y0 + (oy1 - oy0), src_x0 : src_x0 + (ox1 - ox0), :
                ]
        return out

    def associated_names(self) -> set[str]:
        return set(str(name) for name in self._info.metadata.associated_images)

    def read_associated(self, name: str):
        return self._slide.read_associated_image(name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Iris Codec on a .iris slide")
    parser.add_argument("slide_path", nargs="?", help="Path to an encoded .iris slide")
    parser.add_argument("--self-test", action="store_true", help="Run script-only sanity checks")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return 0

    if not args.slide_path:
        print("usage: iris_bench.py <slide-path>", file=sys.stderr)
        return 2

    slide_path = Path(args.slide_path)
    if not slide_path.is_file():
        print(f"slide path is not a file: {slide_path}", file=sys.stderr)
        return 2

    selected = selected_workload()
    repeat = repeat_index()
    workload_name = selected or "single_tile_l0"

    try:
        from Iris import Codec
    except Exception as exc:
        print(
            render_run(
                str(slide_path),
                selected,
                repeat,
                [WorkloadResult.with_error(workload_name, f"Iris import failed: {exc}")],
            )
        )
        return 1

    workloads: list[WorkloadResult] = []
    if should_run_workload(selected, "cold_open"):
        cold = WorkloadResult.new("cold_open")
        for _ in range(10):
            started = time.perf_counter_ns()
            try:
                IrisSlide(slide_path, Codec)
            except Exception as exc:
                cold.error = f"open failed: {exc}"
                break
            cold.samples_us.append((time.perf_counter_ns() - started) // 1000)
        workloads.append(cold)
        if selected == "cold_open" or cold.error:
            print(render_run(str(slide_path), selected, repeat, workloads))
            return 1 if cold.error else 0

    try:
        slide = IrisSlide(slide_path, Codec)
    except Exception as exc:
        print(
            render_run(
                str(slide_path),
                selected,
                repeat,
                [WorkloadResult.with_error(workload_name, f"open failed: {exc}")],
            )
        )
        return 1

    plan = WorkloadPlan.compute(slide.level_dims)

    if should_run_workload(selected, "single_tile_l0"):
        top_left = tile_top_left_at_pixel(plan.center_l0, plan.tile_px)
        workloads.append(
            run_workload("single_tile_l0", 200, lambda: slide.read_tile_at_top_left(0, top_left))
        )

    if should_run_workload(selected, "pan_trace_l0"):
        coords = iter(plan.pan_trace_l0())
        workloads.append(
            run_workload(
                "pan_trace_l0",
                len(plan.pan_trace_l0()),
                lambda: slide.read_tile_at_top_left(0, next(coords)),
            )
        )

    if should_run_workload(selected, "pan_trace_l2"):
        coords_list = plan.pan_trace_l2()
        coords = iter(coords_list)
        workloads.append(
            run_workload(
                "pan_trace_l2",
                len(coords_list),
                lambda: slide.read_tile_at_top_left(plan.level2_idx, next(coords)),
            )
        )

    if should_run_workload(selected, "pan_trace_l2_dense"):
        coords_list = plan.pan_trace_l2_dense()
        coords = iter(coords_list)
        workloads.append(
            run_workload(
                "pan_trace_l2_dense",
                len(coords_list),
                lambda: slide.read_tile_at_top_left(plan.level2_idx, next(coords)),
            )
        )

    if should_run_workload(selected, "region_2k"):
        workloads.append(
            run_workload(
                "region_2k",
                30,
                lambda: slide.read_region(
                    0, plan.center_l0[0] - 1024, plan.center_l0[1] - 1024, 2048, 2048
                ),
            )
        )

    if should_run_workload(selected, "viewport_region_l2"):
        viewport = plan.centered_viewport_l2(1024)
        workloads.append(
            run_workload(
                "viewport_region_l2",
                30,
                lambda: slide.read_region(
                    plan.level2_idx,
                    viewport.level2_top_left[0],
                    viewport.level2_top_left[1],
                    viewport.side_px,
                    viewport.side_px,
                ),
            )
        )

    if should_run_workload(selected, "thumbnail"):
        if "thumbnail" in slide.associated_names():
            workloads.append(run_workload("thumbnail", 30, lambda: slide.read_associated("thumbnail")))
        else:
            deepest = plan.level_count - 1
            dims = slide.level_dims[deepest]
            workloads.append(
                run_workload(
                    "thumbnail",
                    30,
                    lambda: slide.read_region(deepest, 0, 0, dims[0], dims[1]),
                )
            )

    print(render_run(str(slide_path), selected, repeat, workloads))
    return 1 if any(workload.error for workload in workloads) else 0


def run_workload(name: str, target_n: int, func: Callable[[], object]) -> WorkloadResult:
    result = WorkloadResult.new(name)
    result.target_n = target_n
    for _ in range(target_n):
        started = time.perf_counter_ns()
        try:
            value = func()
            consume_result(value)
        except Exception as exc:
            result.error = str(exc)
            break
        result.samples_us.append((time.perf_counter_ns() - started) // 1000)
    return result


def consume_result(value: object) -> None:
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape) == 0:
        raise RuntimeError("Iris returned scalar data")


def render_run(
    slide_path: str,
    selected: Optional[str],
    repeat: Optional[int],
    workloads: list[WorkloadResult],
) -> str:
    return json.dumps(
        {
            "schema_version": SCHEMA_VERSION,
            "library": "iris",
            "slide_path": slide_path,
            "host": host_string(),
            "run_mode": RUN_MODE_SINGLE_WORKLOAD if selected else RUN_MODE_FULL_SUITE,
            "selected_workload": selected,
            "repeat_index": repeat,
            "peak_rss_bytes": None,
            "rss_method": None,
            "workloads": [workload.to_json_obj() for workload in workloads],
        },
        indent=2,
    )


def percentiles(samples_us: list[int]) -> dict:
    values = sorted(samples_us)
    n = len(values)

    def pick(q: float) -> int:
        rank = math.ceil(q * n)
        return values[min(n - 1, max(0, rank - 1))]

    return {
        "p50_us": pick(0.50),
        "p95_us": pick(0.95),
        "p99_us": pick(0.99),
        "max_us": values[-1],
        "mean_us": sum(values) // n,
    }


def selected_workload() -> Optional[str]:
    value = os.environ.get(SELECTED_WORKLOAD_ENV)
    if value is None or not value.strip():
        return None
    value = value.strip()
    if value not in WORKLOAD_BY_NAME:
        valid = ", ".join(spec.name for spec in WORKLOAD_SPECS)
        raise SystemExit(f"invalid {SELECTED_WORKLOAD_ENV} value {value!r}; valid workloads: {valid}")
    return value


def repeat_index() -> Optional[int]:
    value = os.environ.get(REPEAT_INDEX_ENV)
    if value is None:
        return None
    return int(value)


def should_run_workload(selected: Optional[str], workload_name: str) -> bool:
    return selected is None or selected == workload_name


def tile_top_left_at_pixel(pixel_xy: tuple[int, int], tile_px: int) -> tuple[int, int]:
    col = pixel_xy[0] // tile_px
    row = pixel_xy[1] // tile_px
    return (col * tile_px, row * tile_px)


def host_string() -> str:
    try:
        return subprocess.check_output(["uname", "-a"], text=True).strip()
    except Exception:
        return platform.platform()


def run_self_test() -> None:
    dims = [(8192, 8192), (4096, 4096), (2048, 2048)]
    plan = WorkloadPlan.compute(dims)
    assert plan.level_count == 3
    assert plan.level2_idx == 2
    assert plan.center_l0 == (4096, 4096)
    assert len(plan.pan_trace_l0()) == 32
    assert plan.pan_trace_l2_dense()[0] == (768, 768)
    assert tile_top_left_at_pixel((4099, 4100), 256) == (4096, 4096)
    result = WorkloadResult.new("single_tile_l0")
    result.samples_us = [50, 10, 99, 1, 100, 95]
    obj = result.to_json_obj()
    assert obj["p50_us"] == 50
    assert obj["p99_us"] == 100
    parsed = json.loads(render_run("/tmp/a.iris", "single_tile_l0", 2, [result]))
    assert parsed["library"] == "iris"
    assert parsed["workloads"][0]["name"] == "single_tile_l0"


if __name__ == "__main__":
    raise SystemExit(main())
