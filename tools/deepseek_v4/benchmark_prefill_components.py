#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the V4 prefill compile bisection with one isolated cache per case.

The actual model/component command is supplied as a template because device
images use different launchers.  ``{component}``, ``{query}``, ``{cache}``, and
``{diagnostic}`` are substituted without invoking a shell.  The child should
write compiler output to stdout/stderr; GNU time supplies wall/RSS metrics and
the cache is inspected for NEFF size.  This keeps cold measurements honest and
makes the resulting JSON directly consumable by ``analyze_compile_artifacts``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path


COMPONENTS = (
    "sliding_mla",
    "hca_mla",
    "csa_indexer",
    "csa_mla",
    "moe_mapping",
    "moe_cte",
)
QUERIES = (512, 1024, 2048, 4096)


def _bytes_under(root: Path, suffix: str | None = None) -> int:
    return sum(
        path.stat().st_size
        for path in root.rglob("*")
        if path.is_file() and (suffix is None or path.suffix == suffix)
    )


def run_case(template: str, component: str, query: int, root: Path, diagnostic: str) -> dict:
    cache = root / f"{component}-q{query}-{diagnostic}"
    if cache.exists():
        shutil.rmtree(cache)
    cache.mkdir(parents=True)
    values = {
        "component": component,
        "query": str(query),
        "cache": str(cache),
        "diagnostic": diagnostic,
    }
    command = [part.format(**values) for part in shlex.split(template)]
    env = os.environ.copy()
    env["NEURON_COMPILE_CACHE_URL"] = str(cache)
    env["VLLM_NEURON_DEEPSEEK_V4_DIAGNOSTIC_IDENTITY"] = diagnostic
    started = time.monotonic()
    completed = subprocess.run(
        ["/usr/bin/time", "-v", *command], env=env, text=True, capture_output=True
    )
    elapsed = time.monotonic() - started
    log = cache / "benchmark.log"
    log_text = completed.stdout + completed.stderr
    log.write_text(log_text)

    def last_float(pattern: str) -> float | None:
        matches = re.findall(pattern, log_text, re.I)
        return float(matches[-1]) if matches else None

    rss = re.findall(r"Maximum resident set size \(kbytes\):\s*(\d+)", log_text)
    return {
        "component": component,
        "query": query,
        "diagnostic_identity": diagnostic,
        "command": command,
        "returncode": completed.returncode,
        "wall_seconds": elapsed,
        "frontend_seconds": last_float(r"(?:front[ -]?end|graph extraction)\D+([0-9.]+)\s*s"),
        "backend_seconds": last_float(r"(?:back[ -]?end|neuronx-cc)\D+([0-9.]+)\s*s"),
        "peak_rss_kbytes": int(rss[-1]) if rss else None,
        "instruction_count": last_float(r"(?:instruction count|max instructions)\D+(\d+)"),
        "temporary_hbm_bytes": last_float(r"temporary hbm(?: bytes)?\D+(\d+)"),
        "neff_bytes": _bytes_under(cache, ".neff"),
        "cache_bytes": _bytes_under(cache),
        "log": str(log),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--command-template", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--component", choices=COMPONENTS, action="append")
    parser.add_argument("--query", choices=QUERIES, type=int, action="append")
    parser.add_argument(
        "--diagnostic",
        default="none",
        help="component replaced by the launcher's opaque identity, or 'none'",
    )
    args = parser.parse_args()
    cache_root = args.cache_root or Path(tempfile.mkdtemp(prefix="dsv4-cold-"))
    cache_root.mkdir(parents=True, exist_ok=True)
    records = [
        run_case(args.command_template, component, query, cache_root, args.diagnostic)
        for component in (args.component or COMPONENTS)
        for query in (args.query or QUERIES)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"cache_root": str(cache_root), "records": records}, indent=2) + "\n")
    if any(record["returncode"] for record in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
