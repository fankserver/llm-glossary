# GPU hardware

**GPU generations / architectures** — NVIDIA names generations after scientists: **Ampere** (A100) → **Hopper** (H100, GH200; compute capability **sm_90**) → **Blackwell** (B200, GB200, GB10; **sm_100/sm_12x**). Kernels are often built per-generation — e.g. native FP4 math instructions exist only on Blackwell.

**H100 NVL** — A Hopper-generation datacenter GPU variant (94 GB memory, NVLink-bridge capable).

**SM / compute capability (sm_90, SM120, SM121a…)** — NVIDIA's version number for a GPU's instruction set. Software compiled for one may not run on another.

**VRAM** — GPU memory. Everything (weights + KV cache + overhead) must fit in it.

**HBM / HBM3 (High-Bandwidth Memory)** — The very fast memory stacked on datacenter GPUs. Decode speed is usually limited by how fast weights can be read from it ("memory-bandwidth-bound" / the "memory-bandwidth floor").

**NVLink / NVLink bridge** — NVIDIA's fast GPU-to-GPU interconnect (~600 GB/s vs ~64 GB/s over **PCIe**, the standard slot connection). Determines whether tensor parallelism is fast or crippled.

**DGX Spark / GB10 / HP ZGX Nano G1n** — NVIDIA's small Grace-Blackwell desktop AI box (HP sells a variant as ZGX Nano). **Grace** = the ARM CPU part; the GPU and CPU share **unified memory** (128 GB LPDDR5X, ~273 GB/s — much slower than HBM, so decode is slower than on datacenter GPUs).

**Clustering / networking terms** (from the DGX-Spark clustering guide) — **ConnectX-7 (CX-7)**: the 200 Gb/s network card; **QSFP112/QSFP56/DAC**: port/cable types; **RoCEv2 / InfiniBand**: high-speed networking protocols for GPU clusters; **GPU Direct RDMA**: NICs writing straight into GPU memory (not supported on GB10); **C2C**: NVIDIA's chip-to-chip link between Grace and the GPU.

**Kernel** — A small program that runs on the GPU (a matrix multiply, an attention step). Fast serving is mostly about having fast kernels for your exact GPU + number format.

- **GEMM** — general matrix multiply, the workhorse kernel.
- **CUTLASS / FlashInfer / [FlashAttention](https://arxiv.org/abs/2205.14135) (flash_attn) / Marlin / Triton / TileLang / CuteDSL / TRT-LLM kernels** — kernel libraries/backends. vLLM picks among them (`--attention-backend`, `moe_backend`); mismatches cause real incompatibilities (e.g. DFlash needs FLASH_ATTN but fp8 KV needs FLASHINFER). **Marlin** also does fast on-the-fly dequantization of 4-bit weights on GPUs without native FP4.
- **PTX / NVVM / NVCC / CUDA** — NVIDIA's GPU assembly / compiler / toolchain underneath everything.

**cudagraph (CUDA graph) / `--enforce-eager`** — Recording a whole GPU step once and replaying it, removing per-step launch overhead. `--enforce-eager` disables this (slower, but works around capture bugs). **torch.compile / `-O3`** — PyTorch's compiler producing fused, optimized kernels at startup.
