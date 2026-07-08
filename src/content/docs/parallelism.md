---
title: "Multi-GPU & parallelism"
---

**TP (Tensor Parallelism)** — Splitting every layer's matrices across N GPUs, which then cooperate on every single token. Needs very fast GPU-to-GPU links. `--tensor-parallel-size`.

**PP (Pipeline Parallelism)** — Splitting the model by *layers*: GPU 1 holds the first half, GPU 2 the second; tokens flow through like an assembly line.

**DP (Data Parallelism)** — Running full copies of the model and splitting *requests* between them.

**EP (Expert Parallelism)** — For MoE models: splitting the experts across GPUs.

**[NCCL](https://developer.nvidia.com/nccl)** — NVIDIA's library for GPU-to-GPU communication (used by all the above). An **all-reduce** is its core operation: combining partial results from all GPUs (Ring/Tree = its algorithms). vLLM variants: **custom all-reduce** (fast path over NVLink peer-to-peer), **PYNCCL**, **P2P** (direct GPU-to-GPU memory access).

**torchrun** — PyTorch's launcher for multi-GPU/multi-node jobs.
