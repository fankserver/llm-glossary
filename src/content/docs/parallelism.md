---
title: "Multi-GPU & parallelism"
---

**[TP](https://arxiv.org/abs/1909.08053) (Tensor Parallelism)** — Splitting every layer's matrices across N GPUs, which then cooperate on every single token. Needs very fast GPU-to-GPU links. `--tensor-parallel-size`.

**[PP](https://arxiv.org/abs/1811.06965) (Pipeline Parallelism)** — Splitting the model by *layers*: GPU 1 holds the first half, GPU 2 the second; tokens flow through like an assembly line.

**DP (Data Parallelism)** — Running full copies of the model and splitting *requests* between them.

**[EP](https://arxiv.org/abs/2006.16668) (Expert Parallelism)** — For MoE models: splitting the experts across GPUs.

**`--split-mode layer / row` (llama.cpp)** — llama.cpp's multi-GPU knob: **layer** assigns whole layers to each GPU (PP-style, works fine over plain PCIe, only one GPU computes at a time), **row** splits each matrix across GPUs (TP-style, needs fast interconnect to pay off).

**[NCCL](https://developer.nvidia.com/nccl)** — NVIDIA's library for GPU-to-GPU communication (used by all the above). An **[all-reduce](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)** is its core operation: combining partial results from all GPUs (Ring/Tree = its algorithms). vLLM variants: **custom all-reduce** (fast path over NVLink peer-to-peer), **PYNCCL**, **P2P** (direct GPU-to-GPU memory access).

**torchrun** — PyTorch's launcher for multi-GPU/multi-node jobs.
