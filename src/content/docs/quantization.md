---
title: "Quantization (making models smaller)"
---

**Quantization (quant)** — Storing model weights (and sometimes activations/KV cache) in fewer bits than the original 16, trading a little quality for a lot less memory and often more speed. "4-bit is not free" — quality loss is real and must be validated.

**Number formats** —
- **[BF16](https://arxiv.org/abs/1905.12322) / FP16** — 16-bit floats; the "full quality" baseline for inference.
- **[FP8](https://arxiv.org/abs/2209.05433)** — 8-bit float; halves memory vs 16-bit with usually negligible quality loss. A common default for serving big models.
- **FP4 (e2m1)** — 4-bit float; e2m1 is the bit layout (2 exponent, 1 mantissa bit). Very small, but quality risk is significant — blamed for one model's degraded output. Only 16 representable values, so FP4 is never used alone: weights are stored in small blocks that share a scale factor, and the two competing formats below differ exactly in how that block scaling works.
- **[MXFP4](https://arxiv.org/abs/2310.10537)** — the open **OCP Microscaling (MX)** standard: blocks of 32 values sharing one **E8M0** scale (8-bit, exponent-only — powers of two). Cross-vendor: supported on NVIDIA Blackwell *and* AMD hardware.
- **[NVFP4](https://arxiv.org/abs/2509.25149)** — NVIDIA's own variant, Blackwell-only: smaller blocks of 16 values with a finer FP8 (E4M3) scale, plus a per-tensor FP32 scale on top. The finer/smaller-block scaling tracks outliers better, so it usually loses less quality than MXFP4 — at slightly more storage overhead.
- **[Ternary / 1.58-bit / BitNet](https://arxiv.org/abs/2402.17764)** — the extreme end: every weight is just −1, 0, or +1 (log₂3 ≈ 1.58 bits of information). The step below is **binary / 1-bit** (weights only ±1). Requires training the model that way from the start; rewards are tiny files and multiplication-free inference.
- **[INT8 / INT4](https://arxiv.org/abs/2004.09602)** — 8-/4-bit integers.
- **W4A16 / W8A8** — shorthand: **W**eights in 4-bit, **A**ctivations in 16-bit, etc.

**[GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) quant types (Q4_0, Q4_1, Q4_K_M, Q8_0, IQ2_XS, …)** — The naming scheme for quantization levels of GGUF files (llama.cpp / Ollama). Reading the name: `Q<bits>_<variant>` — the number is bits per weight, the suffix is *how* the scaling works:
- **Q4_0 / Q5_0 / Q8_0** — the original ("legacy") scheme: weights are grouped in blocks of 32, each block stores one scale factor. Simple, slightly lossier.
- **Q4_1 / Q5_1** — same, but each block stores a scale *and* an offset (minimum), recovering a bit more accuracy for a bit more size. So Q4_1 ≈ slightly bigger + slightly better than Q4_0.
- **K-quants (Q3_K, Q4_K, Q5_K, Q6_K)** — the newer scheme: "super-blocks" with smarter, finer-grained scales, and *mixed* precision — important layers get more bits. The **_S / _M / _L** suffix (e.g. **Q4_K_M**) picks small/medium/large within that: how many layers get the higher-precision treatment. Q4_K_M is the common "sweet spot" recommendation.
- **IQ-quants (IQ2_XS, IQ3_M, …)** — "importance-matrix" quants for extreme compression (≲3-bit): calibration data decides which weights deserve precision. Usable where plain Q2/Q3 would fall apart, at some CPU speed cost. The size ladder within one bit level runs **XXS < XS < S < M < L** (bigger = more precision); **IQ4_NL** is the "non-linear" variant whose value grid is spaced to fit weight distributions better than evenly-spaced levels.
- **TQ1_0 / TQ2_0** — ternary types for [BitNet-style 1.58-bit models](https://arxiv.org/abs/2402.17764) (weights −1/0/+1, see the ternary entry above); only meaningful for models trained ternary, not a way to shrink a normal model.
- **Recipe prefixes/suffixes (UD-, _XL, …)** — a prefix like **UD-** (Unsloth Dynamic) or an extra suffix like **_XL** marks a quanter's own recipe on top of a standard type: typically dynamic per-layer bit allocation, with sensitive layers (embeddings, lm-head) kept at higher precision. The standard type after the prefix (`UD-Q4_K_XL` → Q4_K) still tells you the base format; the rest is the uploader's tuning.

Rule of thumb: higher number = better quality + bigger file; at equal bits, IQ > K > _1 > _0 in quality. Q8_0 is near-lossless; Q4_K_M is the typical quality/size compromise; below Q3 expect visible degradation.

**QAT (Quantization-Aware Training)** — The model was *trained* knowing it would be quantized, so the quantized version loses much less quality (e.g. [Gemma](https://arxiv.org/abs/2607.02770)'s QAT checkpoints). Naming pattern worth decoding: a repo called `…-qat-q4_0-unquantized` is not a contradiction — it holds the *full-precision* weights that came out of a QAT run targeting `q4_0`, published so you can do the final quantization to that format yourself.

**[Calibration](https://arxiv.org/abs/2311.09755)** — For post-training quantization: running sample data through the model to choose good scaling factors. Bad calibration → subtle quality bugs (e.g. a "miscalibrated logit tail" at temperature 1.0).

**Quantization toolkits/formats** — **[compressed-tensors](https://github.com/vllm-project/compressed-tensors) (`-ct`)** (checkpoint format used by RedHatAI et al.), **[AutoRound](https://github.com/intel/auto-round)** (Intel), **[AWQ](https://arxiv.org/abs/2306.00978)**, **[GPTQ](https://arxiv.org/abs/2210.17323)**, **[llm-compressor](https://github.com/vllm-project/llm-compressor)**, **[modelopt](https://github.com/NVIDIA/TensorRT-Model-Optimizer)** (NVIDIA TensorRT Model Optimizer — what NVIDIA's own NVFP4/FP8 checkpoints on HuggingFace are made with), **GGUF** (llama.cpp's file format — see the GGUF quant types above).
