# SPDX-License-Identifier: Apache-2.0

from argparse import Namespace

import pytest

from tools.deepseek_v4.benchmark_decode import (
    build_parser,
    request_metrics,
    sequence_buckets,
    selected_workloads,
    summarize_measurements,
    validate_args,
)


def test_transition_interval_is_not_counted_as_steady_itl():
    trace = {
        "token_ids": [1, 2, 3, 4],
        "token_timestamps_seconds": [2.0, 2.8, 3.0, 3.2],
        "started_seconds": 0.1,
        "finished_seconds": 3.4,
    }
    metrics = request_metrics(trace)
    assert metrics["ttft_seconds"] == 2.0
    assert metrics["first_decode_interval_seconds"] == pytest.approx(0.8)
    assert metrics["steady_itl_seconds"] == pytest.approx([0.2, 0.2])
    assert metrics["median_steady_itl_seconds"] == pytest.approx(0.2)


def test_summary_preserves_per_request_and_aggregate_throughput():
    trace = {
        "token_ids": [1, 2, 3],
        "token_timestamps_seconds": [1.0, 1.5, 1.75],
        "started_seconds": 0.0,
        "finished_seconds": 2.0,
    }
    trace["metrics"] = request_metrics(trace)
    measurement = {
        "aggregate_output_tokens_per_second": 6.0,
        "requests": [trace],
    }
    summary = summarize_measurements([measurement])
    assert summary["steady_itl_seconds"]["median"] == 0.25
    assert summary["per_request_output_tokens_per_second"]["median"] == 1.5
    assert summary["aggregate_output_tokens_per_second"]["median"] == 6.0


def test_q8192_default_omits_the_q512_only_short_probe():
    assert selected_workloads(Namespace(workloads=None, query_bucket=512)) == [
        "short",
        "sustained",
        "batch8",
    ]
    assert selected_workloads(Namespace(workloads=None, query_bucket=8192)) == [
        "sustained",
        "batch8",
    ]


def test_short_only_probe_fits_its_fixed_four_token_output():
    parser = build_parser()
    args = parser.parse_args(
        [
            "/tmp/checkpoint",
            "--output",
            "/tmp/result.json",
            "--workload",
            "short",
            "--max-model-len",
            "512",
            "--decode-context-buckets",
            "512",
        ]
    )
    validate_args(parser, args)


def test_single_request_probe_does_not_compile_batch8_graph():
    assert sequence_buckets(["short", "sustained"]) == [1]
    assert sequence_buckets(["batch8"]) == [1, 8]
