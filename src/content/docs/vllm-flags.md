---
title: "Common vLLM flags (quick reference)"
---

| Flag | Meaning |
|---|---|
| `--tensor-parallel-size` | Split the model across N GPUs (TP) |
| `--max-model-len` | Max context length to support (bounds KV-cache size) |
| `--gpu-memory-utilization` | Fraction of GPU memory vLLM may use |
| `--max-num-seqs` | Max concurrent requests |
| `--max-num-batched-tokens` | Prefill chunk size / per-step token budget |
| `--enable-prefix-caching` | Reuse KV cache for shared prompt prefixes |
| `--kv-cache-dtype fp8` | Store KV cache in 8-bit |
| `--quantization` | Weight quantization scheme to use |
| `--speculative-config` | Enable/configure speculative decoding |
| `--enforce-eager` | Disable CUDA graphs (slower, more compatible) |
| `--trust-remote-code` | Allow running model-supplied Python code (needed by some architectures) |
| `--served-model-name` | The model name clients use in API calls |
| `--hf-overrides` | Patch fields in the model's HF config at load time |
| `--tool-call-parser` / `--reasoning-parser` | Which parser converts the model's raw markup into API fields |
| `--attention-backend` | Which attention kernel library to use |
| `--language-model-only` | Skip loading the vision part of a multimodal model |
