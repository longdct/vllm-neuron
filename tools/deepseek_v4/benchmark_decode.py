#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Production-style streaming decode benchmark for DeepSeek-V4.

The first generated token measures prefill/TTFT.  The interval from the first
to the second generated token is reported as the prefill-to-decode transition.
Only later intervals contribute to steady-state ITL statistics.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


WORKLOADS: dict[str, list[int]] = {
    "short": [508],
    "sustained": [384],
    "batch8": [384, 320, 256, 192, 128, 96, 64, 32],
}

SAMPLING_CASES = ("greedy", "top-k", "top-p", "temperature", "mixed")


def percentile(values: list[float], quantile: float) -> float | None:
    """Return a linearly interpolated percentile without a NumPy dependency."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def distribution(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "median": percentile(values, 0.5),
        "p95": percentile(values, 0.95),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def request_metrics(trace: dict[str, Any]) -> dict[str, Any]:
    timestamps = trace["token_timestamps_seconds"]
    intervals = [right - left for left, right in zip(timestamps, timestamps[1:])]
    # Interval 0 is the graph/scheduler transition from prefill's sampled token
    # to the first decode token.  It is deliberately excluded from steady ITL.
    steady = intervals[1:]
    latency = trace["finished_seconds"] - trace["started_seconds"]
    return {
        "ttft_seconds": timestamps[0] if timestamps else None,
        "first_decode_interval_seconds": intervals[0] if intervals else None,
        "steady_itl_seconds": steady,
        "median_steady_itl_seconds": percentile(steady, 0.5),
        "p95_steady_itl_seconds": percentile(steady, 0.95),
        "request_latency_seconds": latency,
        "output_tokens": len(trace["token_ids"]),
        "output_tokens_per_second": (
            len(trace["token_ids"]) / latency if latency > 0 else None
        ),
    }


def summarize_measurements(measurements: list[dict[str, Any]]) -> dict[str, Any]:
    requests = [
        request
        for measurement in measurements
        for request in measurement["requests"]
    ]
    metrics = [request["metrics"] for request in requests]
    steady = [value for metric in metrics for value in metric["steady_itl_seconds"]]
    return {
        "repetitions": len(measurements),
        "requests": len(requests),
        "ttft_seconds": distribution(
            [
                metric["ttft_seconds"]
                for metric in metrics
                if metric["ttft_seconds"] is not None
            ]
        ),
        "first_decode_interval_seconds": distribution(
            [
                metric["first_decode_interval_seconds"]
                for metric in metrics
                if metric["first_decode_interval_seconds"] is not None
            ]
        ),
        "steady_itl_seconds": distribution(steady),
        "request_latency_seconds": distribution(
            [metric["request_latency_seconds"] for metric in metrics]
        ),
        "per_request_output_tokens_per_second": distribution(
            [metric["output_tokens_per_second"] for metric in metrics]
        ),
        "aggregate_output_tokens_per_second": distribution(
            [
                measurement["aggregate_output_tokens_per_second"]
                for measurement in measurements
            ]
        ),
    }


def parse_csv_ints(value: str) -> list[int]:
    try:
        values = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected comma-separated integers"
        ) from error
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("bucket values must be positive")
    if values != sorted(set(values)):
        raise argparse.ArgumentTypeError("bucket values must be unique and increasing")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--load-format", default="dummy")
    parser.add_argument("--tensor-parallel-size", type=int, default=64)
    parser.add_argument(
        "--enable-expert-parallel",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--ep-degree", type=int, default=4)
    parser.add_argument(
        "--quantization",
        choices=("bf16", "fp8"),
        default="bf16",
        help=(
            "Routed-expert weight storage. 'fp8' also requires "
            "UNSAFE_FP8FNCAST=1 and the neuronx-cc e4m3fn cast flag, and "
            "routes the MoE to shard_on_i with a 256-token block."
        ),
    )
    parser.add_argument(
        "--sampling-backend", choices=("cpu", "device"), default="device"
    )
    parser.add_argument(
        "--sampling-case",
        action="append",
        choices=SAMPLING_CASES,
        dest="sampling_cases",
        help="Repeat to benchmark multiple cases in one engine configuration.",
    )
    parser.add_argument(
        "--workload",
        action="append",
        choices=tuple(WORKLOADS),
        dest="workloads",
        help="Repeat to select workloads (default depends on query bucket).",
    )
    parser.add_argument("--query-bucket", type=int, choices=(512, 8192), default=512)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument(
        "--decode-context-buckets", type=parse_csv_ints, default=[4096, 8192]
    )
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--num-gpu-blocks-override", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--max-output-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--deterministic-sampling", action="store_true")
    parser.add_argument(
        "--logprobs",
        type=int,
        default=0,
        help="Request and compile full logits for logprobs when greater than zero.",
    )
    parser.add_argument("--debug-logits-dir")
    return parser


def model_environment() -> dict[str, str | None]:
    """Capture every environment variable that changes what is measured.

    The two arms of an A/B are otherwise indistinguishable in the retained
    JSON: both record the same revision and the same CLI arguments while an
    env flag silently selects a different graph.  Record the flags explicitly
    so a report proves which configuration produced it, and record ``None``
    for an unset flag rather than omitting the key, so the absence of a flag
    is evidence too.
    """
    tracked = (
        "VLLM_NEURON_DSV4_FUSED_MOE_REDUCTION",
        "VLLM_NEURON_DSV4_FIXED_CSA_SELECTION",
        "VLLM_NEURON_DSV4_NKI_COMPRESSOR",
        "VLLM_NEURON_DSV4_DYNAMIC_PAGE_LOOP",
        "NEURON_VISIBLE_DEVICES",
        "NEURON_RT_VISIBLE_CORES",
        "VLLM_NEURON_CPU_MODE",
        "VLLM_CACHE_ROOT",
    )
    return {name: os.environ.get(name) for name in tracked}


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    positive = {
        "tensor parallel size": args.tensor_parallel_size,
        "EP degree": args.ep_degree,
        "max model length": args.max_model_len,
        "block size": args.block_size,
        "KV blocks": args.num_gpu_blocks_override,
        "repetitions": args.repetitions,
        "max output tokens": args.max_output_tokens,
    }
    for name, value in positive.items():
        if value < 1:
            parser.error(f"{name} must be positive")
    if args.warmups < 0:
        parser.error("warmups cannot be negative")
    if args.logprobs < 0:
        parser.error("logprobs cannot be negative")
    if args.ep_degree > args.tensor_parallel_size:
        parser.error("EP degree cannot exceed TP for this single-host benchmark")
    if args.enable_expert_parallel and args.tensor_parallel_size % args.ep_degree:
        parser.error("EP degree must divide tensor parallel size")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("gpu memory utilization must be in (0, 1]")
    if max(args.decode_context_buckets) > args.max_model_len:
        parser.error("decode context buckets cannot exceed max model length")
    longest_request = max(
        max(WORKLOADS[workload])
        + (4 if workload == "short" else args.max_output_tokens)
        for workload in selected_workloads(args)
    )
    if longest_request > args.max_model_len:
        parser.error("max model length does not fit the benchmark workloads")
    if args.query_bucket > args.max_model_len:
        parser.error("query bucket cannot exceed max model length")


def selected_workloads(args: argparse.Namespace) -> list[str]:
    if args.workloads:
        return list(dict.fromkeys(args.workloads))
    # The short transition probe is defined as Q512.  Q8192 validation repeats
    # the sustained and ragged batch workloads only.
    if args.query_bucket == 512:
        return ["short", "sustained", "batch8"]
    return ["sustained", "batch8"]


def sequence_buckets(workloads: list[str]) -> list[int]:
    """Compile only the request-count shapes selected by this invocation.

    Batch-8 runs retain the batch-1 bucket because their active request count
    shrinks during ragged completion.  Single-request-only probes do not need
    to compile or execute the batch-8 graph during engine initialization.
    """
    return [1, 8] if "batch8" in workloads else [1]


def prompt_tokens(length: int, vocab_size: int) -> list[int]:
    usable = max(vocab_size - 1, 1)
    return [1 + (index % usable) for index in range(length)]


def sampling_kwargs(case: str, request_index: int, seed: int) -> dict[str, Any]:
    choices = {
        "greedy": {"temperature": 0.0},
        "top-k": {"temperature": 0.8, "top_k": 50},
        "top-p": {"temperature": 0.8, "top_k": 256, "top_p": 0.9},
        "temperature": {"temperature": 0.7, "top_k": 256},
    }
    if case == "mixed":
        mixed = [
            {"temperature": 0.0},
            {"temperature": 0.8, "top_k": 10},
            {"temperature": 0.8, "top_k": 50},
            {"temperature": 0.8, "top_k": 256, "top_p": 0.8},
            {"temperature": 0.8, "top_k": 256, "top_p": 0.95},
            {"temperature": 0.7, "top_k": 256},
            {"temperature": 1.2, "top_k": 256},
            {"temperature": 0.9, "top_k": 256, "top_p": 0.9},
        ]
        result = dict(mixed[request_index % len(mixed)])
    else:
        result = dict(choices[case])
    result["seed"] = seed + request_index
    return result


async def run_request(
    engine,
    sampling_params_cls,
    prompt: list[int],
    request_id: str,
    sampling: dict[str, Any],
    output_tokens: int,
    workload_started: float,
    logprobs: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    params = sampling_params_cls(
        **sampling,
        max_tokens=output_tokens,
        ignore_eos=True,
        logprobs=logprobs or None,
    )
    token_ids: list[int] = []
    timestamps: list[float] = []
    async for output in engine.generate(prompt, params, request_id):
        now = time.perf_counter()
        current = list(output.outputs[0].token_ids)
        if len(current) > len(token_ids):
            timestamps.extend([now - started] * (len(current) - len(token_ids)))
            token_ids = current
    finished = time.perf_counter()
    trace = {
        "request_id": request_id,
        "prompt_length": len(prompt),
        "sampling": sampling,
        "token_ids": token_ids,
        "started_seconds": started - workload_started,
        "finished_seconds": finished - workload_started,
        "token_timestamps_seconds": timestamps,
        "token_timestamps_from_workload_start_seconds": [
            started - workload_started + timestamp for timestamp in timestamps
        ],
    }
    trace["metrics"] = request_metrics(trace)
    return trace


async def run_once(
    engine,
    sampling_params_cls,
    workload: str,
    case: str,
    repetition: int,
    vocab_size: int,
    output_tokens: int,
    seed: int,
    logprobs: int,
) -> dict[str, Any]:
    lengths = WORKLOADS[workload]
    if case == "mixed" and len(lengths) == 1:
        raise ValueError("the mixed sampling case requires the batch8 workload")
    started = time.perf_counter()
    requests = await asyncio.gather(
        *[
            run_request(
                engine,
                sampling_params_cls,
                prompt_tokens(length, vocab_size),
                f"{workload}-{case}-{repetition}-{index}-{time.time_ns()}",
                sampling_kwargs(case, index, seed + repetition * 1000),
                4 if workload == "short" else output_tokens,
                started,
                logprobs,
            )
            for index, length in enumerate(lengths)
        ]
    )
    finished = time.perf_counter()
    total_tokens = sum(len(request["token_ids"]) for request in requests)
    elapsed = finished - started
    return {
        "repetition": repetition,
        "elapsed_seconds": elapsed,
        "total_output_tokens": total_tokens,
        "aggregate_output_tokens_per_second": total_tokens / elapsed,
        "requests": requests,
    }


def command_output(*command: str) -> str:
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=False)
    except FileNotFoundError:
        return "not-found"
    return (result.stdout + result.stderr).strip()


def checkpoint_identity(checkpoint: Path, load_format: str) -> dict[str, Any]:
    config_path = checkpoint / "config.json"
    config = json.loads(config_path.read_text())
    manifest = checkpoint / "slice-manifest.json"
    if load_format == "dummy" or not list(checkpoint.glob("*.safetensors")):
        kind = "shape-accurate-bf16-dummy"
    elif manifest.exists():
        kind = "official-derived-slice"
    else:
        kind = "checkpoint-weights"
    return {
        "path": str(checkpoint.resolve()),
        "kind": kind,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "architecture": config.get("architectures"),
        "model_depth": config.get("num_hidden_layers"),
        "expert_count": config.get("n_routed_experts", config.get("num_local_experts")),
        "vocab_size": int(config["vocab_size"]),
        "torch_dtype": config.get("torch_dtype", "bfloat16"),
        "slice_manifest": str(manifest.resolve()) if manifest.exists() else None,
    }


def cache_artifacts(cache_root: Path) -> dict[str, Any]:
    cache = cache_root / "neuron" / "compile_cache"
    files = [path for path in cache.rglob("*") if path.is_file()] if cache.exists() else []
    sizes = [path.stat().st_size for path in files]
    return {
        "path": str(cache),
        "file_count": len(files),
        "total_bytes": sum(sizes),
        "largest_file_bytes": max(sizes, default=0),
    }


async def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    # DeepSeek-V4 registration is opt-in (vllm_neuron/model/registry.py).  Without
    # the gate, `ModelRegistry.resolve_model_cls` silently returns vLLM's own
    # `DeepseekV4ForCausalLM` and every worker dies with "has no attribute
    # 'from_configs'".  This tool can only ever benchmark DeepSeek-V4, so set the
    # gate rather than make each caller remember it.
    os.environ.setdefault("VLLM_NEURON_ENABLE_DEEPSEEK_V4", "1")
    # Register TorchNeuron Native before vLLM resolves its platform plugin.
    import vllm_neuron  # noqa: F401
    import vllm.platforms as vllm_platforms

    if not vllm_platforms.current_platform.device_type:
        vllm_platforms._current_platform = None
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    identity = checkpoint_identity(args.checkpoint, args.load_format)
    workloads = selected_workloads(args)
    cases = list(dict.fromkeys(args.sampling_cases or ["greedy"]))
    if "mixed" in cases and "batch8" not in workloads:
        raise ValueError("--sampling-case mixed requires --workload batch8")

    neuron_config: dict[str, Any] = {
        "num_batched_tokens_buckets": [args.query_bucket],
        "num_seqs_buckets": sequence_buckets(workloads),
        "max_logprobs": args.logprobs,
        "on_device_sampling_config": (
            {
                "all_greedy": False,
                "max_top_k": 256,
                "deterministic": args.deterministic_sampling,
            }
            if args.sampling_backend == "device"
            else None
        ),
    }
    explicit_decode = [
        bucket for bucket in args.decode_context_buckets if bucket < args.max_model_len
    ]
    if explicit_decode:
        neuron_config["decode_context_length_buckets"] = explicit_decode
    if args.enable_expert_parallel:
        neuron_config["ep_degree"] = args.ep_degree
    neuron_config["quantization"] = args.quantization
    if args.debug_logits_dir:
        neuron_config["debug_logits_dir"] = args.debug_logits_dir

    engine_args = AsyncEngineArgs(
        model=str(args.checkpoint),
        load_format=args.load_format,
        dtype="bfloat16",
        skip_tokenizer_init=True,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_expert_parallel=args.enable_expert_parallel,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.query_bucket,
        max_num_seqs=max(len(WORKLOADS[workload]) for workload in workloads),
        block_size=args.block_size,
        enable_prefix_caching=False,
        num_gpu_blocks_override=args.num_gpu_blocks_override,
        gpu_memory_utilization=args.gpu_memory_utilization,
        async_scheduling=args.sampling_backend == "device",
        stream_interval=1,
        max_logprobs=args.logprobs,
        disable_log_stats=True,
        additional_config={"neuron_config": neuron_config},
    )
    initialization_started = time.perf_counter()
    engine = AsyncLLM.from_engine_args(engine_args)
    initialization_seconds = time.perf_counter() - initialization_started
    groups: list[dict[str, Any]] = []
    hbm: Any = None
    async_stats: Any = None
    try:
        for workload in workloads:
            for case in cases:
                if case == "mixed" and workload != "batch8":
                    continue
                for warmup in range(args.warmups):
                    await run_once(
                        engine,
                        SamplingParams,
                        workload,
                        case,
                        -(warmup + 1),
                        identity["vocab_size"],
                        args.max_output_tokens,
                        args.seed,
                        args.logprobs,
                    )
                measurements = [
                    await run_once(
                        engine,
                        SamplingParams,
                        workload,
                        case,
                        repetition,
                        identity["vocab_size"],
                        args.max_output_tokens,
                        args.seed,
                        args.logprobs,
                    )
                    for repetition in range(args.repetitions)
                ]
                groups.append(
                    {
                        "workload": workload,
                        "sampling_case": case,
                        "prompt_lengths": WORKLOADS[workload],
                        "output_tokens_per_request": (
                            4 if workload == "short" else args.max_output_tokens
                        ),
                        "measurements": measurements,
                        "summary": summarize_measurements(measurements),
                    }
                )
        try:
            hbm = await engine.collective_rpc("get_hbm_memory_stats")
        except Exception as error:  # Provenance must not invalidate a run.
            hbm = {"error": repr(error)}
        try:
            async_stats = await engine.collective_rpc("get_async_scheduling_stats")
        except Exception as error:
            async_stats = {"error": repr(error)}
    finally:
        engine.shutdown()

    cache_root = Path(os.environ.get("VLLM_CACHE_ROOT", Path.home() / ".cache" / "vllm"))
    return {
        "schema_version": 1,
        "created_unix_seconds": time.time(),
        "configuration": {
            "checkpoint": identity,
            "tensor_parallel_size": args.tensor_parallel_size,
            "expert_parallel_enabled": args.enable_expert_parallel,
            "expert_parallel_degree": args.ep_degree if args.enable_expert_parallel else 1,
            "query_bucket": args.query_bucket,
            "decode_context_buckets": args.decode_context_buckets,
            "max_model_len": args.max_model_len,
            "max_output_tokens": args.max_output_tokens,
            "block_size": args.block_size,
            "num_gpu_blocks_override": args.num_gpu_blocks_override,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "load_format": args.load_format,
            "quantization": args.quantization,
            "sampling_backend": args.sampling_backend,
            "sampling_cases": cases,
            "on_device_sampling": neuron_config["on_device_sampling_config"],
            "async_scheduling": args.sampling_backend == "device",
            "full_logits_requested": bool(args.logprobs or args.debug_logits_dir),
            "warmups_discarded": args.warmups,
            "measured_repetitions": args.repetitions,
            "cache_root": str(cache_root),
        },
        "environment": {
            "python": sys.version,
            "vllm": importlib.metadata.version("vllm"),
            "torch": importlib.metadata.version("torch"),
            "torch_neuronx": importlib.metadata.version("torch-neuronx"),
            "neuronx_cc": command_output(
                str(Path(sys.executable).with_name("neuronx-cc")), "--version"
            ),
            "git_revision": command_output("git", "rev-parse", "HEAD"),
            "git_status": command_output("git", "status", "--short"),
            "neuron_visible_devices": os.environ.get("NEURON_VISIBLE_DEVICES"),
            "model_environment": model_environment(),
        },
        "initialization_seconds": initialization_seconds,
        "hbm": hbm,
        "async_scheduling_stats": async_stats,
        "compiled_artifacts": cache_artifacts(cache_root),
        "results": groups,
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    try:
        result = asyncio.run(benchmark(args))
    except ValueError as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
