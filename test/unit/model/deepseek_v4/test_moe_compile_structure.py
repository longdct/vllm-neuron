# SPDX-License-Identifier: Apache-2.0

import inspect

from vllm_neuron.model.deepseek_v4.model import DeepseekV4MoE


def test_q8192_moe_uses_the_compiler_safe_block512_geometry():
    source = inspect.getsource(DeepseekV4MoE._forward_nki)
    assert "moe_block_size = 512 if original_tokens > 4096 else 128" in source
    assert "block_size=moe_block_size" in source
    assert source.count("block_size=moe_block_size") == 2
