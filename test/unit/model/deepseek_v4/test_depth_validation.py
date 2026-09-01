# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from vllm_neuron.model.deepseek_v4.model import _initialize_dummy_parameters


MODULE_PATH = (
    Path(__file__).resolve().parents[4]
    / "tools"
    / "deepseek_v4"
    / "validate_production_depth.py"
)
SPEC = importlib.util.spec_from_file_location("deepseek_depth_validation", MODULE_PATH)
depth_validation = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(depth_validation)


def _source_config(root: Path) -> Path:
    root.mkdir()
    config = {
        "architectures": ["DeepseekV4ForCausalLM"],
        "num_hidden_layers": 7,
        "num_hash_layers": 3,
        # The official raw config has one trailing entry for its MTP metadata;
        # Transformers retains only the first num_hidden_layers entries.
        "compress_ratios": [0, 0, 4, 128, 4, 128, 0, 128],
        "n_routed_experts": 256,
        "num_experts_per_tok": 6,
        "torch_dtype": "bfloat16",
        "expert_dtype": "fp4",
        "quantization_config": {"quant_method": "fp8"},
    }
    (root / "config.json").write_text(json.dumps(config))
    return root


def test_config_only_rung_preserves_layer_semantics_and_provenance(tmp_path):
    source = _source_config(tmp_path / "source")
    output = tmp_path / "rung"

    manifest = depth_validation.prepare_dummy_checkpoint(
        source, output, depth=5, experts=32
    )

    config = json.loads((output / "config.json").read_text())
    assert config["num_hidden_layers"] == 5
    assert config["compress_ratios"] == [0, 0, 4, 128, 4]
    assert config["num_hash_layers"] == 3
    assert config["n_routed_experts"] == 32
    assert "quantization_config" not in config
    assert "expert_dtype" not in config
    assert not list(output.glob("*.safetensors"))

    assert manifest["weight_mode"] == "deterministic_dummy"
    assert manifest["source_compress_ratio_entries"] == 8
    assert manifest["non_decoder_compress_ratio_entries"] == 1
    assert [layer["attention_type"] for layer in manifest["layers"]] == [
        "sliding_attention",
        "sliding_attention",
        "compressed_sparse_attention",
        "heavily_compressed_attention",
        "compressed_sparse_attention",
    ]
    assert [layer["router_type"] for layer in manifest["layers"]] == [
        "hash",
        "hash",
        "hash",
        "topk",
        "topk",
    ]
    written = json.loads(
        (output / depth_validation.MANIFEST_NAME).read_text()
    )
    assert written == manifest


def test_rung_refuses_invalid_expert_count_and_overwrite(tmp_path):
    source = _source_config(tmp_path / "source")
    with pytest.raises(ValueError, match="num_experts_per_tok"):
        depth_validation.prepare_dummy_checkpoint(
            source, tmp_path / "too-few", depth=3, experts=5
        )
    output = tmp_path / "rung"
    depth_validation.prepare_dummy_checkpoint(source, output, depth=3, experts=32)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        depth_validation.prepare_dummy_checkpoint(source, output, depth=3, experts=32)


def test_canary_layers_cover_quartiles_router_transition_and_edges():
    assert depth_validation.canary_layers(43, 3) == (0, 2, 3, 10, 21, 32, 42)
    assert depth_validation.capture_modules(3, 3) == (
        "model.embed_tokens",
        "model.layers.0",
        "model.layers.1",
        "model.layers.2",
        "lm_head",
    )


def test_capture_summary_reports_finiteness_and_row_magnitudes(tmp_path):
    assert not depth_validation.capture_canaries_pass(
        depth_validation.summarize_captures(tmp_path / "missing")
    )
    capture = tmp_path / "prompt_0" / "step_0"
    capture.mkdir(parents=True)
    torch.save(
        torch.tensor([[3.0, 4.0], [float("nan"), 0.0]]),
        capture / "model.layers.7_rank0.pt",
    )

    report = depth_validation.summarize_captures(tmp_path)

    assert report["capture_count"] == 1
    summary = report["captures"][
        "prompt_0/step_0/model.layers.7_rank0.pt"
    ]
    assert summary["shape"] == [2, 2]
    assert summary["finite_fraction"] == 0.75
    assert summary["max_abs"] == 4.0
    assert summary["row_rms_p95"] >= summary["row_rms_median"]
    assert not depth_validation.capture_canaries_pass(report)

    torch.save(
        torch.tensor([[3.0, 4.0], [1.0, 0.0]]),
        capture / "model.layers.7_rank0.pt",
    )
    assert depth_validation.capture_canaries_pass(
        depth_validation.summarize_captures(tmp_path)
    )


def test_generate_command_compiles_only_batch_one_and_requested_prefill(tmp_path):
    command = depth_validation.generate_command(
        python=Path("/venv/python"),
        generator=Path("generate_tiny.py"),
        checkpoint=tmp_path / "checkpoint",
        result=tmp_path / "result.json",
        tp=16,
        ep_degree=None,
        max_model_len=512,
        prefill_bucket=512,
        prompt_length=508,
        max_tokens=4,
        num_gpu_blocks=256,
        gpu_memory_utilization=0.9,
        captures=None,
        modules=(),
    )
    assert command[command.index("--num-seqs-buckets") + 1] == "1"
    assert command[command.index("--prefill-segment-buckets") + 1] == "512"
    assert command[command.index("--num-gpu-blocks-override") + 1] == "256"
    assert command[command.index("--gpu-memory-utilization") + 1] == "0.9"
    assert "--capture-dir" not in command


def test_device_validation_rejects_occupied_and_under_capacity():
    inventory = [
        {
            "neuron_device": 0,
            "nc_count": 4,
            "logical_neuroncore_config": 2,
            "neuroncore_ids": [0, 1, 2, 3],
            "neuron_processes": [],
        },
        {
            "neuron_device": 1,
            "nc_count": 4,
            "logical_neuroncore_config": 2,
            "neuroncore_ids": [4, 5, 6, 7],
            "neuron_processes": [{"pid": 9}],
        },
    ]
    assert depth_validation.validate_devices((0, 1), 2, inventory) == {
        "logical_cores": [0, 1],
        "logical_capacity": 2,
    }
    with pytest.raises(RuntimeError, match="occupied"):
        depth_validation.validate_devices((4,), 1, inventory)
    with pytest.raises(RuntimeError, match="below TP=3"):
        depth_validation.validate_devices((0, 1), 3, inventory)


def test_runtime_log_summary_extracts_per_rank_memory_and_instructions(tmp_path):
    log = tmp_path / "cold.log"
    log.write_text(
        "(Worker_TP0 pid=1) rank footprint: 0.973 GiB\n"
        "(Worker_TP1 pid=2) rank footprint: 1.125 GiB\n"
        "(Worker_TP0 pid=1) Neuron HBM: 1.07 GiB used, 22.93 GiB free\n"
        "(Worker_TP1 pid=2) Neuron HBM: 1.22 GiB used, 22.78 GiB free\n"
        "instruction count: 1200\nmax instructions = 1300\n"
    )

    assert depth_validation.runtime_log_summary(log) == {
        "rank_memory": {
            "0": {
                "parameter_footprint_gib": 0.973,
                "hbm_used_gib": 1.07,
                "hbm_free_gib": 22.93,
            },
            "1": {
                "parameter_footprint_gib": 1.125,
                "hbm_used_gib": 1.22,
                "hbm_free_gib": 22.78,
            },
        },
        "instruction_count_max": 1300,
    }


def test_dummy_parameter_initialization_is_repeatable_and_preserves_buffers():
    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(5, 7))
            self.register_buffer("routing", torch.arange(6))

    first, second = Module(), Module()
    _initialize_dummy_parameters(first)
    _initialize_dummy_parameters(second)

    torch.testing.assert_close(first.weight, second.weight, rtol=0, atol=0)
    assert torch.isfinite(first.weight).all()
    assert torch.count_nonzero(first.weight) > 0
    assert torch.equal(first.routing, torch.arange(6))
