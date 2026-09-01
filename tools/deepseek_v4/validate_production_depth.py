#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build and optionally execute an isolated DeepSeek-V4 depth ladder.

The ladder deliberately separates depth mechanics from checkpoint-width
validation. Each rung is a config-only BF16 checkpoint, so the model loader
materializes deterministic dummy parameters without requiring every official
weight shard. Cold and warm runs share only their rung-local cache. Expensive
device execution is opt-in; the default writes a complete, reviewable plan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


RATIO_TO_LAYER_TYPE = {
    0: "sliding_attention",
    4: "compressed_sparse_attention",
    128: "heavily_compressed_attention",
}
MANIFEST_NAME = "depth-validation-manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_positive_csv(value: str, option: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise ValueError(f"{option} must be comma-separated integers") from error
    if not parsed or any(item <= 0 for item in parsed):
        raise ValueError(f"{option} entries must be positive")
    if tuple(sorted(set(parsed))) != parsed:
        raise ValueError(f"{option} entries must be unique and increasing")
    return parsed


def layer_manifest(config: dict[str, Any], depth: int) -> list[dict[str, Any]]:
    ratios = config.get("compress_ratios")
    source_depth = int(config.get("num_hidden_layers", 0))
    if not isinstance(ratios, list) or len(ratios) < source_depth:
        raise ValueError(
            "source config must contain at least one compress_ratios entry per "
            "decoder layer"
        )
    if not 1 <= depth <= source_depth:
        raise ValueError(f"depth must be between 1 and {source_depth}; got {depth}")
    hash_layers = int(config.get("num_hash_layers", 0))
    records = []
    for index, ratio in enumerate(ratios[:depth]):
        if ratio not in RATIO_TO_LAYER_TYPE:
            raise ValueError(f"layer {index} has unsupported compression ratio {ratio}")
        records.append(
            {
                "destination_layer": index,
                "source_layer": index,
                "compress_ratio": ratio,
                "attention_type": RATIO_TO_LAYER_TYPE[ratio],
                "router_type": "hash" if index < hash_layers else "topk",
            }
        )
    return records


def prepare_dummy_checkpoint(
    source: Path,
    output: Path,
    *,
    depth: int,
    experts: int,
) -> dict[str, Any]:
    """Create a config-only BF16 checkpoint and its provenance manifest."""
    source_config = source / "config.json" if source.is_dir() else source
    if not source_config.is_file():
        raise FileNotFoundError(f"source config does not exist: {source_config}")
    config = json.loads(source_config.read_text())
    layers = layer_manifest(config, depth)
    source_experts = int(config.get("n_routed_experts", 0))
    topk = int(config.get("num_experts_per_tok", 0))
    if not topk <= experts <= source_experts:
        raise ValueError(
            f"experts must be between num_experts_per_tok={topk} and "
            f"n_routed_experts={source_experts}; got {experts}"
        )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing rung: {output}")
    output.mkdir(parents=True)

    variant = dict(config)
    variant["num_hidden_layers"] = depth
    variant["compress_ratios"] = [record["compress_ratio"] for record in layers]
    variant["num_hash_layers"] = min(int(config.get("num_hash_layers", 0)), depth)
    variant["n_routed_experts"] = experts
    variant["dtype"] = "bfloat16"
    variant["torch_dtype"] = "bfloat16"
    # The depth ladder intentionally validates the current BF16 path.
    variant.pop("quantization_config", None)
    variant.pop("expert_dtype", None)
    (output / "config.json").write_text(json.dumps(variant, indent=2) + "\n")

    manifest = {
        "schema_version": 1,
        "kind": "deepseek_v4_config_only_depth_rung",
        "source_config": str(source_config.resolve()),
        "source_config_sha256": sha256_file(source_config),
        "source_depth": int(config["num_hidden_layers"]),
        "source_compress_ratio_entries": len(config["compress_ratios"]),
        "non_decoder_compress_ratio_entries": max(
            0,
            len(config["compress_ratios"])
            - int(config["num_hidden_layers"]),
        ),
        "depth": depth,
        "source_experts": source_experts,
        "experts": experts,
        "weight_mode": "deterministic_dummy",
        "dtype": "bfloat16",
        "layers": layers,
    }
    (output / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def canary_layers(depth: int, hash_layers: int) -> tuple[int, ...]:
    """Choose sparse boundaries plus both sides of the hash-router transition."""
    if depth < 1:
        raise ValueError("depth must be positive")
    selected = {0, depth - 1, depth // 4, depth // 2, (3 * depth) // 4}
    if 0 < hash_layers < depth:
        selected.update((hash_layers - 1, hash_layers))
    return tuple(sorted(selected))


def capture_modules(depth: int, hash_layers: int) -> tuple[str, ...]:
    return (
        "model.embed_tokens",
        *(f"model.layers.{layer}" for layer in canary_layers(depth, hash_layers)),
        "lm_head",
    )


def _tensor_summary(tensor) -> dict[str, Any]:
    import torch

    result: dict[str, Any] = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "numel": tensor.numel(),
        "bytes": tensor.numel() * tensor.element_size(),
    }
    if tensor.numel() == 0 or not (torch.is_floating_point(tensor) or tensor.is_complex()):
        return result
    values = tensor.detach().float().reshape(-1, tensor.shape[-1] if tensor.ndim else 1)
    finite = torch.isfinite(values)
    result["finite_fraction"] = finite.float().mean().item()
    safe = torch.where(finite, values, torch.zeros_like(values))
    row_rms = safe.square().mean(dim=-1).sqrt()
    result.update(
        {
            "mean": safe.mean().item(),
            "rms": safe.square().mean().sqrt().item(),
            "max_abs": safe.abs().max().item(),
            "row_rms_median": torch.quantile(row_rms, 0.5).item(),
            "row_rms_p95": torch.quantile(row_rms, 0.95).item(),
        }
    )
    return result


def summarize_captures(root: Path) -> dict[str, Any]:
    """Return finite-value and magnitude canaries without retaining tensors."""
    import torch

    files = sorted(root.glob("prompt_*/step_*/*_rank*.pt"))
    summaries: dict[str, Any] = {}
    for path in files:
        tensor = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(tensor, torch.Tensor):
            summaries[str(path.relative_to(root))] = {
                "unsupported_capture_type": type(tensor).__name__
            }
            continue
        summaries[str(path.relative_to(root))] = _tensor_summary(tensor)
    return {
        "capture_root": str(root),
        "capture_count": len(files),
        "captures": summaries,
    }


def capture_canaries_pass(report: dict[str, Any]) -> bool:
    captures = report.get("captures", {})
    if not captures:
        return False
    return all(
        summary.get("finite_fraction", 1.0) == 1.0
        for summary in captures.values()
        if "unsupported_capture_type" not in summary
    ) and all(
        "unsupported_capture_type" not in summary for summary in captures.values()
    )


def cache_summary(root: Path) -> dict[str, int]:
    files = [path for path in root.rglob("*") if path.is_file()] if root.exists() else []
    neffs = [path for path in files if path.suffix == ".neff"]
    return {
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "neff_count": len(neffs),
        "neff_total_bytes": sum(path.stat().st_size for path in neffs),
        "largest_neff_bytes": max((path.stat().st_size for path in neffs), default=0),
    }


def gnu_time_summary(path: Path) -> dict[str, float | int | None]:
    if not path.is_file():
        return {"elapsed_seconds": None, "peak_rss_kbytes": None}
    text = path.read_text(errors="replace")
    rss = re.findall(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
    elapsed = re.findall(
        r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*([0-9:.]+)",
        text,
    )
    elapsed_seconds = None
    if elapsed:
        fields = [float(field) for field in elapsed[-1].split(":")]
        if len(fields) == 2:
            elapsed_seconds = fields[0] * 60 + fields[1]
        elif len(fields) == 3:
            elapsed_seconds = fields[0] * 3600 + fields[1] * 60 + fields[2]
    return {
        "elapsed_seconds": elapsed_seconds,
        "peak_rss_kbytes": int(rss[-1]) if rss else None,
    }


def runtime_log_summary(path: Path) -> dict[str, Any]:
    """Extract rank-level memory and optional compiler metrics from a run log."""
    if not path.is_file():
        return {"rank_memory": {}, "instruction_count_max": None}
    log = path.read_text(errors="replace")
    rank_memory: dict[str, dict[str, float]] = {}
    memory_pattern = re.compile(
        r"Worker_TP(\d+).*Neuron HBM: ([0-9.]+) GiB used, "
        r"([0-9.]+) GiB free"
    )
    for rank, used, free in memory_pattern.findall(log):
        rank_memory[rank] = {
            "hbm_used_gib": float(used),
            "hbm_free_gib": float(free),
        }
    footprint_pattern = re.compile(
        r"Worker_TP(\d+).*rank footprint: ([0-9.]+) GiB"
    )
    for rank, footprint in footprint_pattern.findall(log):
        rank_memory.setdefault(rank, {})["parameter_footprint_gib"] = float(
            footprint
        )
    instructions = [
        int(value)
        for value in re.findall(
            r"(?:instruction count|max instructions)\D+(\d+)", log, re.I
        )
    ]
    return {
        "rank_memory": rank_memory,
        "instruction_count_max": max(instructions) if instructions else None,
    }


def neuron_inventory() -> list[dict[str, Any]]:
    try:
        result = subprocess.run(
            ["neuron-ls", "-j"], capture_output=True, text=True, check=False
        )
    except FileNotFoundError:
        return []
    if result.returncode:
        return []
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return []


def active_compilers() -> list[str]:
    try:
        result = subprocess.run(
            ["pgrep", "-af", "neuronx-cc|walrus_driver"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return []
    return [
        line
        for line in result.stdout.splitlines()
        if line and "pgrep -af" not in line
    ]


def validate_devices(devices: tuple[int, ...], tp: int, inventory) -> dict[str, Any]:
    """Validate the logical core IDs consumed by ``NEURON_VISIBLE_DEVICES``."""
    by_core = {
        int(core): item
        for item in inventory
        for core in item.get("neuroncore_ids", [])
    }
    missing = [device for device in devices if device not in by_core]
    occupied = {
        device: by_core[device].get("neuron_processes", [])
        for device in devices
        if device in by_core and by_core[device].get("neuron_processes")
    }
    capacity = len(devices) - len(missing)
    if missing:
        raise RuntimeError(f"requested logical Neuron cores do not exist: {missing}")
    if occupied:
        raise RuntimeError(
            "refusing to use occupied logical Neuron cores: "
            + ", ".join(
                f"{device} ({len(processes)} processes)"
                for device, processes in occupied.items()
            )
        )
    if capacity < tp:
        raise RuntimeError(
            f"selected devices expose {capacity} logical execution groups, below TP={tp}"
        )
    return {"logical_cores": list(devices), "logical_capacity": capacity}


def generate_command(
    *,
    python: Path,
    generator: Path,
    checkpoint: Path,
    result: Path,
    tp: int,
    ep_degree: int | None,
    max_model_len: int,
    prefill_bucket: int,
    prompt_length: int,
    max_tokens: int,
    num_gpu_blocks: int,
    gpu_memory_utilization: float,
    captures: Path | None,
    modules: tuple[str, ...],
) -> list[str]:
    command = [
        str(python),
        str(generator),
        str(checkpoint),
        "--output",
        str(result),
        "--load-format",
        "dummy",
        "--tensor-parallel-size",
        str(tp),
        "--max-num-seqs",
        "1",
        "--num-seqs-buckets",
        "1",
        "--max-model-len",
        str(max_model_len),
        "--max-num-batched-tokens",
        str(prefill_bucket),
        "--prefill-segment-buckets",
        str(prefill_bucket),
        "--num-gpu-blocks-override",
        str(num_gpu_blocks),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--prompt-length",
        str(prompt_length),
        "--max-tokens",
        str(max_tokens),
        "--first-token-probe",
        "--ignore-eos",
    ]
    if ep_degree is not None:
        command.extend(("--enable-expert-parallel", "--ep-degree", str(ep_degree)))
    if captures is not None:
        command.extend(
            ("--capture-dir", str(captures), "--capture-modules", ",".join(modules))
        )
    return command


def _run_once(command: list[str], env: dict[str, str], root: Path, label: str) -> dict[str, Any]:
    log = root / f"{label}.log"
    time_file = root / f"{label}.time"
    timed = ["/usr/bin/time", "-v", "-o", str(time_file), *command]
    started = time.monotonic()
    with log.open("w") as output:
        completed = subprocess.run(
            timed, env=env, stdout=output, stderr=subprocess.STDOUT, check=False
        )
    result = {
        "returncode": completed.returncode,
        "wall_seconds": time.monotonic() - started,
        "log": str(log),
        "time": str(time_file),
    }
    result.update(gnu_time_summary(time_file))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="official checkpoint config or directory")
    parser.add_argument("run_root", type=Path, help="new isolated output directory")
    parser.add_argument("--depths", default="3,8,16,43")
    parser.add_argument("--experts", type=int, default=32)
    parser.add_argument("--tensor-parallel-size", type=int, default=16)
    parser.add_argument("--ep-degree", type=int)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--prefill-bucket", type=int, default=512)
    parser.add_argument("--prompt-length", type=int)
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument(
        "--num-gpu-blocks-override",
        type=int,
        default=256,
        help=(
            "fixed KV block count (default: 256, enough for the Q512 "
            "heterogeneous-KV diagnostic)"
        ),
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--kv-gmu-budget-cap-fraction",
        type=float,
        default=0.3,
        help="explicit Neuron KV allocation safety cap (default: 0.3)",
    )
    parser.add_argument("--capture-canaries", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--warm", action="store_true")
    parser.add_argument(
        "--devices",
        help="comma-separated logical Neuron core IDs; required with --execute",
    )
    parser.add_argument(
        "--acknowledge-expensive-compile",
        action="store_true",
        help="required to execute any rung deeper than 16 layers",
    )
    parser.add_argument(
        "--allow-concurrent-compiler",
        action="store_true",
        help="run despite an existing neuronx-cc/walrus process (timings are contaminated)",
    )
    args = parser.parse_args()

    try:
        depths = parse_positive_csv(args.depths, "--depths")
    except ValueError as error:
        parser.error(str(error))
    if args.experts < 1 or args.tensor_parallel_size < 1:
        parser.error("--experts and --tensor-parallel-size must be positive")
    if args.ep_degree is not None and args.ep_degree < 1:
        parser.error("--ep-degree must be positive")
    if args.max_tokens < 1 or args.max_model_len < 1 or args.prefill_bucket < 1:
        parser.error("model length, prefill bucket, and max tokens must be positive")
    if args.num_gpu_blocks_override < 32:
        parser.error("--num-gpu-blocks-override must be at least 32")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be in (0, 1]")
    if not 0 < args.kv_gmu_budget_cap_fraction <= 1:
        parser.error("--kv-gmu-budget-cap-fraction must be in (0, 1]")
    prompt_length = args.prompt_length or (args.prefill_bucket - args.max_tokens)
    if prompt_length < 1:
        parser.error("prompt length must be positive")
    if prompt_length + args.max_tokens > args.max_model_len:
        parser.error("prompt length plus max tokens must fit max model length")
    if args.prefill_bucket > args.max_model_len:
        parser.error("prefill bucket cannot exceed max model length")
    if args.run_root.exists():
        parser.error(f"run root already exists: {args.run_root}")
    if args.execute and not args.devices:
        parser.error("--devices is required with --execute")
    if args.execute and max(depths) > 16 and not args.acknowledge_expensive_compile:
        parser.error(
            "--acknowledge-expensive-compile is required to execute depth >16"
        )
    devices: tuple[int, ...] = ()
    if args.devices:
        try:
            devices = parse_positive_csv(args.devices, "--devices")
        except ValueError:
            # Device zero is valid, unlike the other positive-only lists.
            try:
                devices = tuple(int(item) for item in args.devices.split(","))
            except ValueError as error:
                parser.error("--devices must be comma-separated integers")
            if (
                not devices
                or any(item < 0 for item in devices)
                or len(set(devices)) != len(devices)
            ):
                parser.error("--devices entries must be unique and non-negative")

    compilers = active_compilers()
    if args.execute and compilers and not args.allow_concurrent_compiler:
        parser.error(
            "another Neuron compiler is active; wait for it or pass "
            "--allow-concurrent-compiler to accept contaminated timings"
        )

    repo = Path(__file__).resolve().parents[2]
    generator = repo / "tools" / "deepseek_v4" / "generate_tiny.py"
    args.run_root.mkdir(parents=True)
    inventory = neuron_inventory()
    device_report = None
    if args.execute:
        device_report = validate_devices(devices, args.tensor_parallel_size, inventory)

    report: dict[str, Any] = {
        "schema_version": 1,
        "source": str(args.source.resolve()),
        "run_root": str(args.run_root.resolve()),
        "execute": args.execute,
        "warm": args.warm,
        "device_selection": device_report,
        "preexisting_compilers": compilers,
        "measurement_environment": {
            "nki_enable_trace_cache": "0",
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "kv_gmu_budget_cap_fraction": args.kv_gmu_budget_cap_fraction,
            "cache_scope": "one isolated cache per rung",
        },
        "rungs": [],
    }
    report_path = args.run_root / "ladder-report.json"

    env = os.environ.copy()
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    env["PYTHONPATH"] = str(repo)
    env["VLLM_NEURON_ENABLE_DEEPSEEK_V4"] = "1"
    env["VLLM_NEURON_VALIDATE_CACHE_METADATA"] = "1"
    env["NEURON_SKIP_EFA_AFFINITY"] = "1"
    # This SDK defaults the persistent trace cache on. A cold ladder must
    # disable it explicitly; merely removing an inherited setting is not enough.
    env["NKI_ENABLE_TRACE_CACHE"] = "0"
    env["VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION"] = str(
        args.kv_gmu_budget_cap_fraction
    )
    if devices:
        env["NEURON_VISIBLE_DEVICES"] = ",".join(str(item) for item in devices)

    failed = False
    for depth in depths:
        rung_root = args.run_root / f"depth-{depth}-experts-{args.experts}"
        checkpoint = rung_root / "checkpoint"
        cache = rung_root / "cache"
        result = rung_root / "cold-result.json"
        captures = rung_root / "captures" if args.capture_canaries else None
        manifest = prepare_dummy_checkpoint(
            args.source, checkpoint, depth=depth, experts=args.experts
        )
        cache.mkdir()
        hash_layers = sum(
            record["router_type"] == "hash" for record in manifest["layers"]
        )
        modules = capture_modules(depth, hash_layers)
        command = generate_command(
            python=Path(sys.executable),
            generator=generator,
            checkpoint=checkpoint,
            result=result,
            tp=args.tensor_parallel_size,
            ep_degree=args.ep_degree,
            max_model_len=args.max_model_len,
            prefill_bucket=args.prefill_bucket,
            prompt_length=prompt_length,
            max_tokens=args.max_tokens,
            num_gpu_blocks=args.num_gpu_blocks_override,
            gpu_memory_utilization=args.gpu_memory_utilization,
            captures=captures,
            modules=modules,
        )
        (rung_root / "command.txt").write_text(shlex.join(command) + "\n")
        rung: dict[str, Any] = {
            "depth": depth,
            "experts": args.experts,
            "checkpoint": str(checkpoint),
            "cache": str(cache),
            "command": command,
            "canary_modules": list(modules) if args.capture_canaries else [],
            "status": "planned",
        }
        report["rungs"].append(rung)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        if not args.execute:
            continue

        rung_env = dict(env)
        rung_env["VLLM_CACHE_ROOT"] = str(cache)
        cold = _run_once(command, rung_env, rung_root, "cold")
        rung["cold"] = cold
        rung["runtime_metrics"] = runtime_log_summary(Path(cold["log"]))
        rung["cache_summary"] = cache_summary(cache)
        rung["status"] = "passed" if cold["returncode"] == 0 else "failed"
        if result.is_file():
            rung["cold_result"] = json.loads(result.read_text())
        if captures is not None:
            canaries = summarize_captures(captures)
            (rung_root / "capture-canaries.json").write_text(
                json.dumps(canaries, indent=2) + "\n"
            )
            rung["capture_summary"] = str(rung_root / "capture-canaries.json")
            rung["capture_canaries_passed"] = capture_canaries_pass(canaries)
            if not rung["capture_canaries_passed"]:
                rung["status"] = "canary_failed"
                failed = True
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        if cold["returncode"] or failed:
            failed = True
            break
        if args.warm:
            warm_command = list(command)
            warm_result = rung_root / "warm-result.json"
            warm_command[warm_command.index(str(result))] = str(warm_result)
            warm = _run_once(warm_command, rung_env, rung_root, "warm")
            rung["warm"] = warm
            if warm_result.is_file():
                rung["warm_result"] = json.loads(warm_result.read_text())
            if warm["returncode"]:
                rung["status"] = "warm_failed"
                failed = True
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            if failed:
                break

    print(json.dumps(report, indent=2))
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
