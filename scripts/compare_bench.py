#!/usr/bin/env python3
"""compare_bench.py — join ziggurat and openslide bench JSON outputs.

Usage:
    compare_bench.py <wsi_json> <openslide_json>

Prints a markdown measurement table to stdout: one row per workload, with
p50/p99 ratios and a verdict tag (ok / slower / hotspot / jitter).
"""

import json
import sys
from pathlib import Path


def load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def fmt_us(us):
    if us is None:
        return "—"
    if us < 1000:
        return f"{us}μs"
    if us < 100_000:
        return f"{us / 1000:.1f}ms"
    return f"{us / 1000:.0f}ms"


def verdict(ours_p50, ours_p99, theirs_p50, theirs_p99):
    if ours_p50 is None or theirs_p50 is None:
        return "n/a"
    ratio = ours_p50 / theirs_p50 if theirs_p50 else float("inf")
    ours_jitter = (ours_p99 / ours_p50) if ours_p50 else float("inf")
    theirs_jitter = (
        (theirs_p99 / theirs_p50) if (theirs_p99 and theirs_p50) else 1.0
    )
    if ratio > 2.0:
        return "HOTSPOT"
    # Flag JITTER only when ziggurat is meaningfully jitterier than openslide.
    # Two conditions must hold: (1) ziggurat absolute jitter is bad (>5x), and
    # (2) ziggurat jitter is at least 2x worse than openslide on the same path.
    # This avoids false positives on workloads where both libraries are bumpy
    # because of the underlying file format (e.g., NDPI thumbnail decode).
    if ours_jitter > 5.0 and ours_jitter > theirs_jitter * 2.0:
        return "JITTER"
    if ratio > 1.3:
        return "slower"
    if ratio < 0.8:
        return "FASTER"
    return "ok"


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    ours = load(sys.argv[1])
    theirs = load(sys.argv[2])

    ours_by = {w["name"]: w for w in ours["workloads"]}
    theirs_by = {w["name"]: w for w in theirs["workloads"]}
    names = list(ours_by.keys())  # use ziggurat ordering

    print(f"### Slide: `{ours['slide_path']}`")
    print()
    print("| workload | ziggurat p50 | ziggurat p99 | openslide p50 | openslide p99 | p50 ratio | verdict |")
    print("|---|---:|---:|---:|---:|---:|---|")
    for name in names:
        o = ours_by.get(name, {})
        t = theirs_by.get(name, {})
        ours_p50 = o.get("p50_us")
        ours_p99 = o.get("p99_us")
        theirs_p50 = t.get("p50_us")
        theirs_p99 = t.get("p99_us")
        ratio = (
            f"{ours_p50 / theirs_p50:.2f}×" if (ours_p50 and theirs_p50) else "—"
        )
        v = verdict(ours_p50, ours_p99, theirs_p50, theirs_p99)
        print(
            f"| `{name}` | {fmt_us(ours_p50)} | {fmt_us(ours_p99)} | "
            f"{fmt_us(theirs_p50)} | {fmt_us(theirs_p99)} | {ratio} | {v} |"
        )
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
