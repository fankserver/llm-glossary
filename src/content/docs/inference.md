---
title: "How inference works (prefill, decode, caching)"
---

**Prefill** — Phase 1 of answering a request: the engine processes the entire prompt in one (parallel) pass. Determines how long you wait for the first word. **Compute-bound**: speed scales with the GPU's raw math throughput (TFLOPS spec), because all prompt tokens are crunched in parallel.

**Decode** — Phase 2: generating the answer one token at a time. Determines how fast the words stream out. **Memory-bandwidth-bound**: for every single token, all model weights must be re-read from GPU memory, so speed scales with the memory-bandwidth spec (GB/s), not compute. This is why a GPU with fast HBM decodes much faster than a unified-memory box (see [GPU hardware](../gpu-hardware/)) even at similar compute, and why quantization speeds up decode — fewer bytes to read per token.

**[TTFT](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html) (Time To First Token)** — How long from sending the request until the first token of the answer arrives ≈ prefill time. The dominant latency for long agent-style prompts ("0.6s at empty context → 37s at 32k").

**[TPOT](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html) (Time Per Output Token)** — Average time between output tokens during decode. Inverse of decode tok/s.

**KV cache (key-value cache)** — During attention, the model computes "keys" and "values" for every token; caching them means each new token only does new work instead of recomputing the whole context. This cache is the main consumer of GPU memory at long contexts — hence all the tricks (MLA, GQA, fp8 KV) to shrink it. `--kv-cache-dtype fp8` stores it in 8-bit to halve its size.

**[KV cache quantization](https://arxiv.org/abs/2502.15075) (K vs V separately)** — The cache can be stored in fewer bits, like weights. The two halves are not equally sensitive: **keys** are used to compute attention scores, so quantizing K hurts quality more than quantizing **values** (which are only mixed into the output). llama.cpp therefore lets you set them independently — `-ctk` / `--cache-type-k` and `-ctv` / `--cache-type-v` — and a common recipe is a higher-precision K with a more aggressive V (e.g. K=q8_0, V=q4_0). vLLM's `--kv-cache-dtype` applies one dtype to both.

**[PagedAttention](https://arxiv.org/abs/2309.06180) / paged KV** — vLLM's core idea: manage KV-cache memory in fixed-size blocks (like OS memory pages) instead of one big contiguous buffer, so memory isn't wasted. `--block-size` sets the block granularity.

**[Prefix caching](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/) (APC, Automatic Prefix Caching)** — If many requests share the same beginning (e.g. the same big system prompt), the engine reuses the already-computed KV cache for that shared prefix instead of re-prefilling it. Biggest single latency lever for agent workloads ("6–21× lower TTFT"). Flag: `--enable-prefix-caching`. Doesn't work for layer types that carry running state (GDN/Mamba), since there's no per-token cache to reuse.

**[Chunked prefill](https://arxiv.org/abs/2308.16369)** — Splitting a very long prompt into fixed-size chunks (e.g. 8192 tokens) processed sequentially, so one huge request doesn't stall everyone else. Flags: `--enable-chunked-prefill`, `--max-num-batched-tokens`.

**[Continuous batching](https://www.usenix.org/conference/osdi22/presentation/yu)** — The engine dynamically packs many concurrent requests into each GPU step, adding/removing requests on the fly (rather than waiting for a batch to fill). `--max-num-seqs` caps concurrent requests (llama.cpp's counterparts: `-b` = logical batch size, `-ub` = the physical micro-batch ("ubatch") actually run per GPU pass). "**Single-stream**" benchmarks use concurrency = 1 (one request at a time) to measure best-case latency.

**[Streaming](https://platform.openai.com/docs/guides/streaming-responses) / deltas** — Sending the answer to the client incrementally as it's generated (what makes chat UIs "type"). Each increment is a *delta*. Speculative decoding can emit several tokens per delta ("multi-token deltas"), which broke older parsers. `stream_options: {include_usage}` asks for token counts at the end of a stream.

**Async scheduling** — A vLLM scheduler mode that overlaps CPU scheduling work with GPU execution for extra throughput.

**CPU offloading / hybrid CPU-GPU inference** — Running part of a too-big model from system RAM/CPU. Classically layer-granular (llama.cpp's `-ngl` = how many layers go to GPU); newer systems ([ATSInfer](https://arxiv.org/abs/2607.10183)) schedule per *tensor* and adapt to load. Always slower than all-on-GPU — PCIe bandwidth is the ceiling.

**OOM / OOMKill (Out Of Memory)** — The GPU (or the container) ran out of memory and the process died — the classic failure when the KV cache, model, and overhead don't fit. `--gpu-memory-utilization` sets what fraction of GPU memory vLLM may claim; `--max-model-len` caps context length to bound the cache; "headroom" = memory left free as safety margin.
