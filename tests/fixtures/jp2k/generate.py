#!/usr/bin/env python3

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WIDTH = 16
HEIGHT = 12


def run(*args: str) -> None:
    subprocess.run(args, check=True, cwd=ROOT)


def write_ppm(path: Path, pixels: list[tuple[int, int, int]], width: int, height: int) -> None:
    with path.open("wb") as fh:
        fh.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        for r, g, b in pixels:
            fh.write(bytes((r, g, b)))


def rgb_source_pixels() -> list[tuple[int, int, int]]:
    pixels: list[tuple[int, int, int]] = []
    for y in range(HEIGHT):
        for x in range(WIDTH):
            pixels.append(
                (
                    (x * 17 + y * 11) % 256,
                    (x * 7 + y * 23 + 40) % 256,
                    (x * 29 + y * 5 + 80) % 256,
                )
            )
    return pixels


def write_ycbcr_raw(path: Path, sub_x: int, sub_y: int) -> None:
    y_plane = bytearray()
    cb_plane = bytearray()
    cr_plane = bytearray()

    for y in range(HEIGHT):
        for x in range(WIDTH):
            y_plane.append((x * 13 + y * 19 + 32) % 256)

    for y in range(0, HEIGHT, sub_y):
        for x in range(0, WIDTH, sub_x):
            cb_plane.append((64 + x * 9 + y * 7) % 256)
            cr_plane.append((192 - x * 5 - y * 11) % 256)

    path.write_bytes(bytes(y_plane + cb_plane + cr_plane))


def read_ppm_pixels(path: Path) -> tuple[int, int, bytes]:
    data = path.read_bytes()
    if not data.startswith(b"P6\n"):
        raise ValueError(f"{path} is not a binary PPM")

    cursor = 3
    tokens: list[bytes] = []
    while len(tokens) < 3:
        while cursor < len(data) and chr(data[cursor]).isspace():
            cursor += 1
        if cursor < len(data) and data[cursor] == ord("#"):
            while cursor < len(data) and data[cursor] != ord("\n"):
                cursor += 1
            continue
        start = cursor
        while cursor < len(data) and not chr(data[cursor]).isspace():
            cursor += 1
        tokens.append(data[start:cursor])

    width = int(tokens[0])
    height = int(tokens[1])
    maxval = int(tokens[2])
    if maxval != 255:
        raise ValueError(f"unsupported maxval in {path}: {maxval}")

    while cursor < len(data) and chr(data[cursor]).isspace():
        cursor += 1
    return width, height, data[cursor:]


def ycbcr_triplets_to_rgb(ycbcr_data: bytes) -> list[tuple[int, int, int]]:
    if len(ycbcr_data) % 3 != 0:
        raise ValueError("YCbCr data is not pixel-interleaved")

    pixels: list[tuple[int, int, int]] = []
    for offset in range(0, len(ycbcr_data), 3):
        yy = ycbcr_data[offset]
        cb = ycbcr_data[offset + 1]
        cr = ycbcr_data[offset + 2]
        cb_off = cb - 128
        cr_off = cr - 128
        r = max(0, min(255, yy + ((1402 * cr_off) // 1000)))
        g = max(0, min(255, yy - ((344 * cb_off + 714 * cr_off) // 1000)))
        b = max(0, min(255, yy + ((1772 * cb_off) // 1000)))
        pixels.append((r, g, b))
    return pixels


def build_fixture(
    tmp: Path,
    stem: str,
    compress_args: list[str],
    upsample: bool,
    convert_ycbcr_reference: bool,
) -> None:
    j2k = tmp / f"{stem}.j2k"
    decoded = tmp / f"{stem}.ppm"

    run("opj_compress", *compress_args, "-o", str(j2k))

    decompress_args = ["opj_decompress"]
    if upsample:
        decompress_args.append("-upsample")
    decompress_args.extend(["-i", str(j2k), "-o", str(decoded)])
    run(*decompress_args)
    if convert_ycbcr_reference:
        width, height, pixels = read_ppm_pixels(decoded)
        write_ppm(ROOT / decoded.name, ycbcr_triplets_to_rgb(pixels), width, height)
    else:
        shutil.copy2(decoded, ROOT / decoded.name)
    shutil.copy2(j2k, ROOT / j2k.name)


def main() -> None:
    if shutil.which("opj_compress") is None or shutil.which("opj_decompress") is None:
        raise SystemExit("opj_compress and opj_decompress must be available on PATH")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        rgb_source = tmp / "rgb_source.ppm"
        write_ppm(rgb_source, rgb_source_pixels(), WIDTH, HEIGHT)

        write_ycbcr_raw(tmp / "ycbcr_444.raw", 1, 1)
        write_ycbcr_raw(tmp / "ycbcr_422.raw", 2, 1)
        write_ycbcr_raw(tmp / "ycbcr_420.raw", 2, 2)

        build_fixture(
            tmp,
            "rgb_nomct",
            ["-i", str(rgb_source), "-I", "-mct", "0", "-r", "5", "-n", "3"],
            upsample=False,
            convert_ycbcr_reference=False,
        )
        build_fixture(
            tmp,
            "rgb_mct",
            ["-i", str(rgb_source), "-I", "-mct", "1", "-r", "5", "-n", "3"],
            upsample=False,
            convert_ycbcr_reference=False,
        )
        build_fixture(
            tmp,
            "ycbcr_444",
            [
                "-i",
                str(tmp / "ycbcr_444.raw"),
                "-F",
                f"{WIDTH},{HEIGHT},3,8,u@1x1:1x1:1x1",
                "-I",
                "-mct",
                "0",
                "-r",
                "5",
                "-n",
                "3",
            ],
            upsample=True,
            convert_ycbcr_reference=True,
        )
        build_fixture(
            tmp,
            "ycbcr_422",
            [
                "-i",
                str(tmp / "ycbcr_422.raw"),
                "-F",
                f"{WIDTH},{HEIGHT},3,8,u@1x1:2x1:2x1",
                "-I",
                "-mct",
                "0",
                "-r",
                "5",
                "-n",
                "3",
            ],
            upsample=True,
            convert_ycbcr_reference=False,
        )
        build_fixture(
            tmp,
            "ycbcr_420",
            [
                "-i",
                str(tmp / "ycbcr_420.raw"),
                "-F",
                f"{WIDTH},{HEIGHT},3,8,u@1x1:2x2:2x2",
                "-I",
                "-mct",
                "0",
                "-r",
                "5",
                "-n",
                "3",
            ],
            upsample=True,
            convert_ycbcr_reference=False,
        )


if __name__ == "__main__":
    main()
