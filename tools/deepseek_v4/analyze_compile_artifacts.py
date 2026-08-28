#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Summarize repeatable DeepSeek-V4 FX and Neuron compilation artifacts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


FX_PATTERNS = {
    "linear_matmul": re.compile(r"(?:linear|mm|matmul|addmm)"),
    "einsum": re.compile(r"einsum"),
    "cache_update": re.compile(r"(?:index_copy|scatter|copy_)"),
    "mask": re.compile(r"(?:masked_fill|where|logical_|bitwise_)"),
    "routing": re.compile(r"(?:topk|scatter_add|expert|affinit)"),
}

LOG_PATTERNS = {
    "frontend_seconds": re.compile(r"(?:front[ -]?end|graph extraction)\D+([0-9.]+)\s*s", re.I),
    "backend_seconds": re.compile(r"(?:back[ -]?end|neuronx-cc)\D+([0-9.]+)\s*s", re.I),
    "engine_compile_seconds": re.compile(r"compilation:\s*([0-9.]+)\s*s"),
    "engine_init_seconds": re.compile(r"init engine .* took\s*([0-9.]+)\s*s"),
    "target_wall_seconds": re.compile(r"(?:target wall|wall time)\D+([0-9.]+)\s*s", re.I),
    "runtime_latency_seconds": re.compile(r"(?:runtime latency|generation after initialization)\D+([0-9.]+)\s*s", re.I),
    "hbm": re.compile(r"(?:hbm|neuron memory)\D+([0-9.]+\s*[KMG]i?B)", re.I),
    "peak_rss": re.compile(r"(?:peak rss|maximum resident set size)\D+([0-9.]+\s*[KMG]i?B)", re.I),
    "max_instructions": re.compile(r"max(?:imum)? instruction(?: count)?\D+(\d+)", re.I),
}

TIME_V_PEAK_RSS = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)")
TIME_V_ELAPSED = re.compile(
    r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*([0-9:.]+)"
)


def _elapsed_seconds(value: str) -> float:
    fields = [float(field) for field in value.split(":")]
    if len(fields) == 2:
        return fields[0] * 60 + fields[1]
    if len(fields) == 3:
        return fields[0] * 3600 + fields[1] * 60 + fields[2]
    raise ValueError(f"unsupported GNU time elapsed value: {value!r}")


def analyze_fx(path: Path) -> dict[str, int]:
    text = path.read_text(errors="replace")
    nodes = [line for line in text.splitlines() if re.search(r"\b(call_|placeholder|output)\w*", line)]
    result = {"total_nodes": len(nodes)}
    result.update({name: sum(bool(pattern.search(line)) for line in nodes) for name, pattern in FX_PATTERNS.items()})
    return result


def analyze_log(path: Path) -> dict[str, object]:
    text = path.read_text(errors="replace")
    result: dict[str, object] = {}
    for name, pattern in LOG_PATTERNS.items():
        matches = pattern.findall(text)
        result[name] = matches[-1] if matches else None
    result["cache_hits"] = len(re.findall(r"cache hit", text, re.I))
    result["cache_misses"] = len(re.findall(r"cache miss", text, re.I))
    cache_keys = re.findall(r"(?:cache key|model hash)\s*[:=]\s*([A-Za-z0-9_.:/+-]+)", text, re.I)
    result["cache_keys"] = sorted(set(cache_keys))
    peak_rss = TIME_V_PEAK_RSS.findall(text)
    result["time_v_peak_rss_kbytes"] = int(peak_rss[-1]) if peak_rss else None
    elapsed = TIME_V_ELAPSED.findall(text)
    result["time_v_elapsed_seconds"] = (
        _elapsed_seconds(elapsed[-1]) if elapsed else None
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fx", type=Path, action="append", default=[])
    parser.add_argument("--log", type=Path, action="append", default=[])
    parser.add_argument("--cache-root", type=Path)
    args = parser.parse_args()
    report: dict[str, object] = {
        "fx": {str(path): analyze_fx(path) for path in args.fx},
        "logs": {str(path): analyze_log(path) for path in args.log},
    }
    if args.cache_root:
        neffs = list(args.cache_root.rglob("*.neff"))
        report["neff"] = {
            "count": len(neffs),
            "total_bytes": sum(path.stat().st_size for path in neffs),
            "largest_bytes": sorted((path.stat().st_size for path in neffs), reverse=True)[:10],
        }
        report["cache"] = {
            "file_count": sum(1 for path in args.cache_root.rglob("*") if path.is_file()),
            "total_bytes": sum(path.stat().st_size for path in args.cache_root.rglob("*") if path.is_file()),
        }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
