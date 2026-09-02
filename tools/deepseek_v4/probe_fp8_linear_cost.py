#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Does storing a projection in FP8 cost latency, or save it?

Attention is 2% of DeepSeek-V4's parameters, so FP8 there buys ~1% of weight
memory. Whether that is worth having depends entirely on what the dequantize
costs at batch-1 decode, which is bandwidth-bound: if the compiler fuses the
scale multiply into the weight load, FP8 reads half the bytes and should be
*faster*; if it materializes a bf16 copy first, it reads 2.5x and is slower.

Measured on trn2 (batch 1, one core, `UNSAFE_FP8FNCAST=1` plus the neuronx-cc
e4m3 cast flag)::

        projection            shape   bf16 ms    fp8 ms   ratio
              wo_a     (8192, 4096)     0.167     0.271    1.62x
              wq_b    (32768, 1024)     0.166     0.256    1.54x
               wkv      (512, 4096)     0.070     0.076    1.09x

It does not fuse. Dequantize-in-forward therefore costs about 1.6x on the
projections that matter, to save ~1% of weight memory (attention and the
shared experts are 2.01% of DeepSeek-V4's parameters). That is why milestone 5
did not wire FP8 storage into the attention path: the conversion is
numerically almost free, but spending it this way is a regression. It becomes
worth revisiting behind a kernel that consumes FP8 weights directly, the way
`model/llama3/model_static_fp8.py` drives `NF.mlp`.
"""
import time, torch

class Bf16Linear(torch.nn.Module):
    def __init__(self, w):
        super().__init__()
        self.weight = torch.nn.Parameter(w, requires_grad=False)
    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight)

class Fp8Linear(torch.nn.Module):
    def __init__(self, w8, scale):
        super().__init__()
        self.weight = torch.nn.Parameter(w8, requires_grad=False)
        self.scale = torch.nn.Parameter(scale, requires_grad=False)
    def forward(self, x):
        w = self.weight.to(torch.bfloat16) * self.scale[:, None]
        return torch.nn.functional.linear(x, w)

def bench(mod, x, dev, iters=50):
    compiled = torch.compile(mod.to(dev), backend="neuron", dynamic=False)
    xd = x.to(dev)
    for _ in range(5):
        compiled(xd)
    torch.neuron.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        compiled(xd)
    torch.neuron.synchronize()
    return (time.perf_counter() - start) / iters * 1e3

dev = torch.device("neuron:0")
gen = torch.Generator().manual_seed(0)
print(f"{'projection':>18} {'shape':>16} {'bf16 ms':>9} {'fp8 ms':>9} {'ratio':>7}")
for name, out, inp in (("wo_a", 8192, 4096), ("wq_b", 32768, 1024), ("wkv", 512, 4096)):
    w = (torch.randn((out, inp), generator=gen) * 0.02).to(torch.bfloat16)
    ref = w.float()
    peak = ref.abs().amax(dim=1).clamp(min=1e-30)
    scale = torch.exp2(torch.ceil(torch.log2(peak / 240.0)))
    w8 = (ref / scale[:, None]).to(torch.float8_e4m3fn)
    x = torch.randn((1, inp), generator=gen).to(torch.bfloat16)

    t_bf16 = bench(Bf16Linear(w), x, dev)
    t_fp8 = bench(Fp8Linear(w8, scale.to(torch.bfloat16)), x, dev)
    print(f"{name:>18} {str((out,inp)):>16} {t_bf16:9.3f} {t_fp8:9.3f} {t_fp8/t_bf16:7.2f}x")
