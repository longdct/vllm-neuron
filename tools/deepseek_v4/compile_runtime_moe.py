#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Cold-compile the DeepSeek-V4 Q512 routed-MoE CTE component."""

from __future__ import annotations

import argparse
import importlib.util
import json
import resource
import sys
import time
from pathlib import Path

import nki.language as nl
import torch
from nkilib.core.moe.moe_cte.moe_cte import (
    ActFnType,
    ExpertAffinityScaleMode,
    MoECTEImplementation,
)


_MOE_PATH = (
    Path(__file__).resolve().parents[2]
    / "vllm_neuron"
    / "functional"
    / "moe"
    / "moe_cte.py"
)
_MOE_SPEC = importlib.util.spec_from_file_location("dsv4_standalone_moe_cte", _MOE_PATH)
assert _MOE_SPEC is not None and _MOE_SPEC.loader is not None
_MOE_MODULE = importlib.util.module_from_spec(_MOE_SPEC)
sys.modules[_MOE_SPEC.name] = _MOE_MODULE
_MOE_SPEC.loader.exec_module(_MOE_MODULE)
moe_cte = _MOE_MODULE.moe_cte


class RuntimeMoE(torch.nn.Module):
    def __init__(self, block_size: int):
        super().__init__()
        self.block_size = block_size

    def forward(self, hidden, affinities, gate_up, down, token_ids, experts):
        return moe_cte(
            hidden_states=hidden,
            expert_affinities_masked=affinities,
            gate_up_proj_weight=gate_up,
            down_proj_weight=down,
            token_position_to_id=token_ids,
            block_to_expert=experts,
            block_size=self.block_size,
            implementation=MoECTEImplementation.shard_on_block,
            activation_function=ActFnType.SiLU,
            compute_dtype=nl.bfloat16,
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            gate_clamp_upper_limit=10.0,
            gate_clamp_lower_limit=-10.0,
            up_clamp_upper_limit=10.0,
            up_clamp_lower_limit=-10.0,
            skip_token=True,
            is_tensor_update_accumulating=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query", type=int, choices=(512, 1024, 2048, 4096, 8192), default=512
    )
    parser.add_argument("--block-size", type=int, choices=(128, 256, 512), default=128)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    hidden_size, intermediate, experts, top_k = 4096, 2048, 32, 6
    block_size = args.block_size
    # The mapper reserves one partially filled block per expert in addition to
    # the densely packed routed-token blocks.
    blocks = (args.query * top_k + block_size - 1) // block_size + experts
    device = torch.device("neuron:0")
    hidden = torch.zeros(args.query, hidden_size, dtype=torch.bfloat16, device=device)
    affinities = torch.zeros(
        args.query * experts, 1, dtype=torch.bfloat16, device=device
    )
    gate_up = torch.zeros(
        experts, hidden_size, 2, intermediate, dtype=torch.bfloat16, device=device
    )
    down = torch.zeros(
        experts, intermediate, hidden_size, dtype=torch.bfloat16, device=device
    )
    token_ids = torch.full(
        (blocks * block_size,), -1, dtype=torch.int32, device=device
    )
    block_experts = torch.zeros(blocks, dtype=torch.int32, device=device)

    compiled = torch.compile(RuntimeMoE(block_size), backend="neuron", dynamic=False)
    started = time.monotonic()
    output = compiled(hidden, affinities, gate_up, down, token_ids, block_experts)
    torch.neuron.synchronize()
    elapsed = time.monotonic() - started
    record = {
        "query": args.query,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate,
        "experts": experts,
        "top_k": top_k,
        "blocks": blocks,
        "wall_seconds": elapsed,
        "peak_rss_kbytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "output_shape": list(output.shape),
    }
    text = json.dumps(record, indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
