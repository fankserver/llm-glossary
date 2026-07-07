# Glossary

Plain-language explanations of terminology from the world of self-hosted LLM inference and serving. Written for someone **without** an LLM/GPU background — each entry says what the thing is and why it matters in practice.

Organized by topic; use Ctrl+F to look up a term. Synonyms and abbreviations are listed in parentheses.

---

## 1. LLM basics

**LLM (Large Language Model)** — A neural network trained on huge amounts of text that generates text one small piece (token) at a time. Self-hosting means running ("serving") such models on your own hardware instead of using a cloud API.

**Token** — The unit an LLM reads and writes. Roughly a word fragment: ~3–4 characters of English on average. "128K context" means the model can consider ~128,000 tokens of input at once. Speeds are measured in **tok/s** (tokens per second).

**Context / context window / context length** — The amount of text (in tokens) the model can "see" at once: the conversation so far plus its own output. Bigger contexts need more GPU memory. "@32k" in benchmarks means "with a 32,000-token prompt".

**Prompt** — The input text sent to the model (system instructions + conversation + question).

**Inference** — Running a trained model to get answers (as opposed to *training* it). "Inference engine" = the software that does this efficiently (vLLM, SGLang, llama.cpp, …).

**Serving** — Running a model behind an API so applications can send requests to it.

**Weights / parameters (params)** — The learned numbers that make up a model. "31B" = 31 billion parameters. More parameters ≈ more capable but needs more memory and compute.

**Checkpoint** — A saved copy of a model's weights, usually downloaded from HuggingFace.

**Sampling / sampling parameters** — How the model picks the next token from its probability distribution. **temperature** (higher = more random), **top_p** / **top_k** (restrict choices to the most likely tokens), **greedy** (always pick the single most likely token — deterministic).

**Logits** — The raw scores the model assigns to every possible next token before they're turned into probabilities.

---

## 2. Model architecture

**Transformer** — The standard neural-network architecture behind modern LLMs. Its core mechanism is **attention** (below). Most terms in this section are variations on it.

**Attention** — The mechanism that lets the model relate each token to every other token in the context ("which earlier words matter for the next word?"). It is expensive: cost grows with context length, which is why so many variants below exist.

**Dense model** — A "normal" model where *all* parameters are used for every token (e.g. "dense 30.7B" Gemma). Opposite of MoE.

**MoE (Mixture of Experts)** — An architecture where the model contains many small sub-networks ("experts") and a router activates only a few per token. This is why you see two sizes: "284B total / 13B active" means the model has 284B parameters on disk/in memory, but only ~13B do work per token — so it's much faster than a dense 284B model. Written as suffixes like `-A10B` ("10B active").

- **Routed / shared experts** — Routed experts are chosen per token by the router; shared experts always run.
- **Expert routing / auxiliary-loss-free routing** (`noaux_tc`, `e_score_correction_bias`, `topk_method`, `scoring_func`) — Config knobs describing *how* the router picks experts (DeepSeek-V3 style). Only matters when a bad config breaks model loading.

**MLA (Multi-head Latent Attention)** — DeepSeek's attention variant that stores a *compressed* version of the attention cache, making long contexts use far less GPU memory (e.g. 122K tokens of context in ~11 GiB instead of many times that).

**GQA (Grouped-Query Attention)** — A common attention optimization where several "query heads" share the same "key/value heads", shrinking the KV cache. Most modern non-DeepSeek models use it.

**DSA (DeepSeek Sparse Attention)** — DeepSeek's trick to make attention cheaper on long contexts: a small "**Lightning Indexer**" first picks the top-k most relevant earlier tokens, and full attention is only computed against those. The indexer keeps its own small cache.

**NSA (Native Sparse Attention)** — DeepSeek's earlier published sparse-attention design (same family of ideas as DSA): attend only to a selected subset of tokens instead of all of them, trained natively that way rather than bolted on afterwards.

**SWA (Sliding-Window Attention)** — Attention limited to a fixed window of recent tokens (e.g. `sliding_window=128`) instead of the whole context. Cheap, but the layer can't see far back; models mix SWA layers with full-attention layers.

**c4a / c128a** — Shorthand in DeepSeek-V4 for compressed global-attention layers with a per-layer KV compression ratio of 4 or 128 (higher = more compressed = cheaper).

**SSM (State-Space Model) / Mamba / Mamba-2** — An alternative to attention: instead of looking back at all tokens, the layer maintains a running "hidden state" that is updated token by token (like a summary it carries forward). Much cheaper for long contexts. "**Hybrid**" models interleave Mamba/SSM layers with attention layers.

**GDN (Gated DeltaNet)** — Qwen's hybrid linear-attention/recurrent layer type (used in ~75% of Qwen3.5/3.6 layers). Like SSMs, it carries a running state — which has practical consequences: the state can't be checkpointed per token, so features like prefix caching and speculative-decoding rollback don't work naturally with it.

**Encoder / decoder** — Two transformer flavors. *Decoders* generate text left-to-right (all chat LLMs). *Encoders* read the whole input at once and output a representation (classic for embeddings, e.g. BERT-style). Matters for embedding models: decoder-based embedders need **last-token pooling** (take the representation of the final token), encoders typically use mean/CLS pooling.

**lm-head** — The model's final layer that converts its internal representation into scores over the vocabulary (i.e. "which token comes next"). Sometimes quantized separately.

**Multimodal / vision model** — A model that accepts images as well as text. `--language-model-only` disables the vision part to save memory when only text is needed.

**YaRN** — A technique to stretch a model's context window beyond what it was trained on (e.g. extending to 1M tokens).

**CoT (Chain of Thought) / thinking / reasoning** — The model "thinks out loud" before answering, emitting reasoning inside special tags (e.g. `<think>…</think>`). Serving-side knobs: `enable_thinking`, `reasoning_effort`, `thinking_token_budget`. The **reasoning parser** (below) strips this into a separate `reasoning` field so clients don't see it mixed into the answer.

**model_type** — The architecture identifier in a HuggingFace model config (`deepseek_v3`, `qwen3_5_moe`, `nemotron_h`, …). Tells the engine which code path to load.

---

## 3. How inference works (prefill, decode, caching)

**Prefill** — Phase 1 of answering a request: the engine processes the entire prompt in one (parallel) pass. Compute-heavy; determines how long you wait for the first word.

**Decode** — Phase 2: generating the answer one token at a time. Memory-bandwidth-heavy; determines how fast the words stream out.

**TTFT (Time To First Token)** — How long from sending the request until the first token of the answer arrives ≈ prefill time. The dominant latency for long agent-style prompts ("0.6s at empty context → 37s at 32k").

**TPOT (Time Per Output Token)** — Average time between output tokens during decode. Inverse of decode tok/s.

**KV cache (key-value cache)** — During attention, the model computes "keys" and "values" for every token; caching them means each new token only does new work instead of recomputing the whole context. This cache is the main consumer of GPU memory at long contexts — hence all the tricks (MLA, GQA, fp8 KV) to shrink it. `--kv-cache-dtype fp8` stores it in 8-bit to halve its size.

**KV cache quantization (K vs V separately)** — The cache can be stored in fewer bits, like weights. The two halves are not equally sensitive: **keys** are used to compute attention scores, so quantizing K hurts quality more than quantizing **values** (which are only mixed into the output). llama.cpp therefore lets you set them independently — `-ctk` / `--cache-type-k` and `-ctv` / `--cache-type-v` — and a common recipe is a higher-precision K with a more aggressive V (e.g. K=q8_0, V=q4_0). vLLM's `--kv-cache-dtype` applies one dtype to both.

**PagedAttention / paged KV** — vLLM's core idea: manage KV-cache memory in fixed-size blocks (like OS memory pages) instead of one big contiguous buffer, so memory isn't wasted. `--block-size` sets the block granularity.

**Prefix caching (APC, Automatic Prefix Caching)** — If many requests share the same beginning (e.g. the same big system prompt), the engine reuses the already-computed KV cache for that shared prefix instead of re-prefilling it. Biggest single latency lever for agent workloads ("6–21× lower TTFT"). Flag: `--enable-prefix-caching`. Doesn't work for layer types that carry running state (GDN/Mamba), since there's no per-token cache to reuse.

**Chunked prefill** — Splitting a very long prompt into fixed-size chunks (e.g. 8192 tokens) processed sequentially, so one huge request doesn't stall everyone else. Flags: `--enable-chunked-prefill`, `--max-num-batched-tokens`.

**Continuous batching** — The engine dynamically packs many concurrent requests into each GPU step, adding/removing requests on the fly (rather than waiting for a batch to fill). `--max-num-seqs` caps concurrent requests. "**Single-stream**" benchmarks use concurrency = 1 (one request at a time) to measure best-case latency.

**Streaming / deltas** — Sending the answer to the client incrementally as it's generated (what makes chat UIs "type"). Each increment is a *delta*. Speculative decoding can emit several tokens per delta ("multi-token deltas"), which broke older parsers. `stream_options: {include_usage}` asks for token counts at the end of a stream.

**Async scheduling** — A vLLM scheduler mode that overlaps CPU scheduling work with GPU execution for extra throughput.

**OOM / OOMKill (Out Of Memory)** — The GPU (or the container) ran out of memory and the process died — the classic failure when the KV cache, model, and overhead don't fit. `--gpu-memory-utilization` sets what fraction of GPU memory vLLM may claim; `--max-model-len` caps context length to bound the cache; "headroom" = memory left free as safety margin.

---

## 4. Speculative decoding

**Speculative decoding (spec decode)** — A decode speed-up: a small, fast "**drafter**" guesses the next several tokens, then the big model verifies the whole guess in *one* pass. Correct guesses are accepted for free; wrong ones fall back to normal decoding. Output is identical to non-speculative decoding — it's purely faster. Configured with `--speculative-config`; **k** / `num_speculative_tokens` = how many tokens are drafted per step.

- **Drafter / speculator / draft model** — the small model doing the guessing. **Verifier** — the big model checking.
- **Acceptance rate / accepted-per-step / eff tok/step** — how many drafted tokens the verifier accepts on average. The whole game: higher acceptance = faster. "42% acceptance, 7.33 length" = of long drafts, on average ~7 tokens survive per step.

Common methods:

**MTP (Multi-Token Prediction)** — The drafter is a small extra layer *inside* the main model (DeepSeek ships one built-in; Gemma uses a separate `-assistant` drafter checkpoint). No separate model to manage.

**EAGLE / EAGLE3** — A popular family of trained drafter heads that reuse the big model's internal states to draft accurately.

**Medusa** — An older multi-head drafting method (referenced for comparison).

**DFlash** — A "block-diffusion" drafter (~0.8B, from z-lab): instead of guessing tokens one by one, it drafts a whole block of tokens in a single non-causal forward pass (a "denoising step"), like filling in a whole phrase at once.

**DSpark** — DeepSeek's block-parallel drafting scheme built on the model's own MTP weights (knobs like `dspark_block_size`), used with DeepSeek-V4 models.

**IndexCache / skip_topk** — An upstream vLLM optimization (tracked in upgrades.md) that reuses DSA's top-k token selection across compressed-attention layers instead of recomputing it.

**QAT-matched drafter** — A drafter quantized/trained to match the quantized main model, so acceptance doesn't drop.

---

## 5. Quantization (making models smaller)

**Quantization (quant)** — Storing model weights (and sometimes activations/KV cache) in fewer bits than the original 16, trading a little quality for a lot less memory and often more speed. "4-bit is not free" — quality loss is real and must be validated.

**Number formats** —
- **BF16 / FP16** — 16-bit floats; the "full quality" baseline for inference.
- **FP8** — 8-bit float; halves memory vs 16-bit with usually negligible quality loss. A common default for serving big models.
- **FP4 / NVFP4 / MXFP4 (e2m1)** — 4-bit float formats (NVFP4 = NVIDIA's, with per-16-value scaling; e2m1 = the bit layout). Very small, but quality risk is significant — blamed for one model's degraded output.
- **INT8 / INT4** — 8-/4-bit integers.
- **W4A16 / W8A8** — shorthand: **W**eights in 4-bit, **A**ctivations in 16-bit, etc.
- **E8M0** — an 8-bit exponent-only scale format used inside MXFP4.

**GGUF quant types (Q4_0, Q4_1, Q4_K_M, Q8_0, IQ2_XS, …)** — The naming scheme for quantization levels of GGUF files (llama.cpp / Ollama). Reading the name: `Q<bits>_<variant>` — the number is bits per weight, the suffix is *how* the scaling works:
- **Q4_0 / Q5_0 / Q8_0** — the original ("legacy") scheme: weights are grouped in blocks of 32, each block stores one scale factor. Simple, slightly lossier.
- **Q4_1 / Q5_1** — same, but each block stores a scale *and* an offset (minimum), recovering a bit more accuracy for a bit more size. So Q4_1 ≈ slightly bigger + slightly better than Q4_0.
- **K-quants (Q3_K, Q4_K, Q5_K, Q6_K)** — the newer scheme: "super-blocks" with smarter, finer-grained scales, and *mixed* precision — important layers get more bits. The **_S / _M / _L** suffix (e.g. **Q4_K_M**) picks small/medium/large within that: how many layers get the higher-precision treatment. Q4_K_M is the common "sweet spot" recommendation.
- **IQ-quants (IQ2_XS, IQ3_M, …)** — "importance-matrix" quants for extreme compression (≲3-bit): calibration data decides which weights deserve precision. Usable where plain Q2/Q3 would fall apart, at some CPU speed cost.

Rule of thumb: higher number = better quality + bigger file; at equal bits, IQ > K > _1 > _0 in quality. Q8_0 is near-lossless; Q4_K_M is the typical quality/size compromise; below Q3 expect visible degradation.

**QAT (Quantization-Aware Training)** — The model was *trained* knowing it would be quantized, so the quantized version loses much less quality (e.g. Gemma's QAT checkpoints).

**Calibration** — For post-training quantization: running sample data through the model to choose good scaling factors. Bad calibration → subtle quality bugs (e.g. a "miscalibrated logit tail" at temperature 1.0).

**Quantization toolkits/formats** — **compressed-tensors (`-ct`)** (checkpoint format used by RedHatAI et al.), **AutoRound** (Intel), **AWQ**, **GPTQ**, **llm-compressor**, **modelopt** (NVIDIA TensorRT ModelOpt), **GGUF** (llama.cpp's file format — see the GGUF quant types above).

---

## 6. Fine-tunes & community model names

HuggingFace is full of community models with long, keyword-stacked names. The pattern is always the same: **base model + size + what was done to it + how it's packaged**. The vocabulary:

**Base model** — The original model everything else is derived from. Community models almost always name their base first, because that determines architecture, tokenizer, and rough capability.

**Fine-tune / finetune** — Taking a base model and training it further on extra data to change its behavior (coding, roleplay, a specific style). Cheap compared to training from scratch; the vast majority of HuggingFace uploads are fine-tunes. Flavors you'll see in cards: **SFT** (supervised fine-tuning — train on example conversations), **DPO/RLHF/GRPO** (preference tuning — train on "this answer is better than that one").

**LoRA / QLoRA** — The cheap way to fine-tune: instead of updating all weights, train a small "adapter" bolted onto them (QLoRA = doing that on top of a quantized model, so it fits on consumer GPUs). Adapters can be shipped separately (a few hundred MB) or merged into the weights before upload.

**Instruct / -it / chat** — A model tuned to follow instructions in a conversation (vs. a raw "base"/"text" model that just continues text). Nearly everything you'd self-host for chat is an instruct model.

**Distillation (distill)** — Training a smaller/cheaper model on the *outputs* of a bigger model so it imitates it. A bigger model's name inside a fine-tune's name (e.g. "R1-Distill", or a Claude/GPT model name) means "trained on data generated by that model", **not** that it contains that model's weights.

**Merge / model merging** — Combining the weights of several fine-tunes of the same base into one model (no training involved, just arithmetic). Popular in the hobbyist scene; quality is hit-or-miss. The standard tool is **mergekit**; method names that show up in cards: **SLERP**, **TIES**, **DARE** / **DARE-TIES**, **Model Stock**, **task arithmetic** — with per-model **weight** and **density** knobs controlling how much each ingredient contributes.

**Codenames & purpose words in names** — Fantasy words in a name are pure branding — the uploader's name for their recipe, with no standard meaning. Functional words hint at the target use: **Novelist / Writer / Storyteller** = creative prose, **Coder** = programming, **RP** = roleplay. Version tags (`v0.2`, `-preview`) and context tags (`-128K`) mean what they say.

**Uncensored / abliterated / abliteration** — The model's refusal behavior ("I can't help with that") has been removed. *Abliteration* is the common technique: find the internal direction that represents refusal and delete it, no retraining needed. **Heretic** is a popular open-source tool that automates this. Side effect: can make models dumber or less stable.

**Frankenmodel / upscaling / parameter expansion** — Making a model *bigger* than its base by duplicating/stacking layers, then (sometimes) healing it with more training (e.g. a 27B base sold as "40B"). The extra parameters are not new knowledge — treat capability claims skeptically.

**Thinking / Reasoning (in a name)** — The fine-tune emits chain-of-thought before answering (see CoT, section 2). Often paired with a base that supports it natively.

**RP / roleplay models** — Fine-tunes optimized for creative writing/character roleplay (a large share of the merge/uncensored scene).

**imatrix (importance matrix)** — Calibration data used when making GGUF quants (the basis of IQ-quants, section 5): sample text is run through the model to measure which weights matter most, so those keep more precision. "IMatrix" in a repo name = the quants were made with one; the choice of calibration text is up to whoever made the quants. Other quant-related keywords in such names (custom dataset names, "MAX", etc.) are usually the uploader's own branding for their recipe, not standard terms.

**-GGUF suffix / quant repos** — A repo name ending in `-GGUF` is not a new model: it's someone's repackaging of an existing model into GGUF quant files for llama.cpp/Ollama (one file per quant level, Q4_K_M etc. — section 5).

Rule of thumb: the longer the name, the further the model is from anything validated. Base-vendor releases and QAT checkpoints have benchmarks; ten-keyword merge/uncensored/frankenmodel stacks usually have only the uploader's word.

---

## 7. Multi-GPU & parallelism

**TP (Tensor Parallelism)** — Splitting every layer's matrices across N GPUs, which then cooperate on every single token. Needs very fast GPU-to-GPU links. `--tensor-parallel-size`.

**PP (Pipeline Parallelism)** — Splitting the model by *layers*: GPU 1 holds the first half, GPU 2 the second; tokens flow through like an assembly line.

**DP (Data Parallelism)** — Running full copies of the model and splitting *requests* between them.

**EP (Expert Parallelism)** — For MoE models: splitting the experts across GPUs.

**NCCL** — NVIDIA's library for GPU-to-GPU communication (used by all the above). An **all-reduce** is its core operation: combining partial results from all GPUs (Ring/Tree = its algorithms). vLLM variants: **custom all-reduce** (fast path over NVLink peer-to-peer), **PYNCCL**, **P2P** (direct GPU-to-GPU memory access).

**torchrun** — PyTorch's launcher for multi-GPU/multi-node jobs.

---

## 8. GPU hardware

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
- **CUTLASS / FlashInfer / FlashAttention (flash_attn) / Marlin / Triton / TileLang / CuteDSL / TRT-LLM kernels** — kernel libraries/backends. vLLM picks among them (`--attention-backend`, `moe_backend`); mismatches cause real incompatibilities (e.g. DFlash needs FLASH_ATTN but fp8 KV needs FLASHINFER). **Marlin** also does fast on-the-fly dequantization of 4-bit weights on GPUs without native FP4.
- **PTX / NVVM / NVCC / CUDA** — NVIDIA's GPU assembly / compiler / toolchain underneath everything.

**cudagraph (CUDA graph) / `--enforce-eager`** — Recording a whole GPU step once and replaying it, removing per-step launch overhead. `--enforce-eager` disables this (slower, but works around capture bugs). **torch.compile / `-O3`** — PyTorch's compiler producing fused, optimized kernels at startup.

---

## 9. Tokens, templates & parsing

**Tokenizer** — Converts text ↔ tokens using the model's fixed **vocab** (vocabulary). The **detokenizer** converts output tokens back to text. `--tokenizer-mode` selects the implementation.

**Special tokens** — Non-text control tokens: **BOS** (beginning-of-sequence), **EOS** (end-of-sequence — the model emits it to stop). Bugs here are visible: a "BOS leak" prints `<｜begin▁of▁sentence｜>` into the user's answer; a wrong EOS makes the model never stop. `skip_special_tokens` controls whether they're stripped from output.

**Chat template** — A per-model recipe (written in **Jinja**, a templating language) that converts a conversation (system/user/assistant messages, tool definitions) into the exact token sequence the model was trained on. Getting this wrong silently degrades quality. `add_generation_prompt` = whether the template appends the "now the assistant speaks:" marker.

**Tool calling / function calling** — The model asking the client to run a function (e.g. "call get_weather(city=Berlin)") by emitting structured output. `tool_choice: auto/required` controls whether the model may/must call tools; `--enable-auto-tool-choice` turns support on.

**Tool-call parser / reasoning parser** — vLLM plugins (`--tool-call-parser`, `--reasoning-parser`) that recognize each model's proprietary output markup and convert it into clean OpenAI-API-style fields. Historically hand-rolled per model and fragile in streaming; the new unified **ParserEngine** replaces them.

**DSML** — DeepSeek's proprietary markup language for tool calls inside its raw output (tags like `<｜DSML｜…>`). The parser must strip it; a known bug leaked close-tags into tool arguments.

**Structured output** — Constraining the model to emit valid JSON matching a schema.

---

## 10. Embeddings & RAG

**Embedding** — A list of numbers (a vector) representing a text's *meaning*, so similar texts get nearby vectors. Produced by dedicated (small) embedding models, often cheap enough to run on CPU.

**RAG (Retrieval-Augmented Generation)** — Look up relevant documents by embedding similarity first, then paste them into the LLM's prompt so it answers from your data. The main reason embedding models are served alongside LLMs.

**Dense / sparse / multi-vector (ColBERT) embeddings** — Three retrieval-vector flavors (bge-m3 produces all three): *dense* = one meaning vector (the OpenAI `/embeddings` kind); *sparse* = keyword-weight style; *ColBERT* = one small vector per token, matched late.

**Reranking / reranker** — A second-stage model that re-scores retrieved candidates against the query for better final ordering.

**Pooling** — How per-token outputs are collapsed into one embedding vector (mean, CLS, or **last-token** for decoder-based embedders).

**Thread pinning** — On CPU inference, fixing the number of worker threads (llama.cpp `-t`, `OMP_NUM_THREADS`) to match the container's CPU limit; oversubscription tanks throughput.

---

## 11. Serving engines & runtimes

**vLLM** — The most widely used open-source LLM serving engine for GPUs. Known for PagedAttention, continuous batching, and an OpenAI-compatible API. Fast-moving; production setups often pin specific builds/nightlies.

**SGLang** — A competing GPU serving engine, notable for its speculative-decoding implementation ("Spec V2").

**llama.cpp** — A lightweight C++ inference engine, great on CPU; uses GGUF files.

**Infinity** — A Python (torch/optimum) embedding server, commonly used for CPU embedding serving.

**TEI (Text Embeddings Inference)** — HuggingFace's Rust embedding server (evaluated; MKL/ONNX issues).

**TensorRT-LLM / Ollama** — Other inference engines referenced for comparison (NVIDIA's optimized engine; a hobbyist-friendly local runner).

**OpenAI-compatible API** — The de-facto standard HTTP API shape (`/v1/chat/completions`, `/v1/embeddings`); most engines speak it so any OpenAI client works against self-hosted models.

**LiteLLM** — A proxy/router that presents many backends behind one OpenAI-style API with keys and quotas (commonly paired with chat UIs like LibreChat).

**fastsafetensors / runai_streamer / safetensors** — **safetensors** is the standard safe weight-file format; the other two are fast loaders that stream weights into GPU memory at startup (model load takes minutes otherwise).

---

## 12. Platform (PCAI / Kubernetes)

**HPE PCAI (Private Cloud AI)** — HPE's on-prem AI appliance: a managed Kubernetes cluster with a model-serving layer. Typically locked-down (limited volume mounts, pod logs, SSH).

**Kubernetes (K8s)** — The container-orchestration system underneath. Relevant vocabulary: **pod** (running container instance), **PVC** (persistent volume claim — a chunk of storage, e.g. for model weights), **RWX** (storage mountable by many pods at once), **hostPath** (mounting a node directory — often forbidden by cluster policy), **/dev/shm** (shared-memory filesystem pods use for inter-process communication), **CRD** (custom resource definition — a K8s API extension), **Kyverno** (a policy engine that enforces/mutates cluster rules).

**KServe** — The model-serving framework on PCAI. An **InferenceService (isvc)** is one deployed model (with its URL and bearer-token **JWT** auth); a **ServingRuntime** defines the container/args used to run it; **autoscaling/replicas** control how many copies run.

**MLIS** — PCAI's model-inventory API (`/api/v1/models`). Important nuance: it lists model *definitions*, not what is actually running right now.

**aioli** — Name/namespace of PCAI's serving stack.

**HF / HuggingFace** — The standard hub for downloading model weights (`HF_TOKEN` = auth token; **Xet** = its newer CDN download backend).

**Engine wedge** — Informal term for a serving incident where the engine silently stalls without crashing (typical root cause: memory swapping; related vocab: **swap-thrash**, **vm.swappiness**, **OOM-killer**, **demand-paging** — Linux memory-management behaviors that can starve the GPU process).

**frp / frpc** — A reverse-proxy tunnel tool for exposing a machine behind NAT/firewall (e.g. a home lab serving endpoint) to the internet.

---

## 13. Monitoring & benchmarking

**Prometheus / `/metrics` / scrape** — The standard metrics pipeline: services expose numbers at `/metrics`; Prometheus "scrapes" (fetches) them periodically. vLLM exposes KV-cache usage, request counts, etc. there.

**Grafana / Grafana Cloud / Alloy** — Dashboards for those metrics; **Alloy** is the collector agent that scrapes and forwards them (config blocks like `prometheus.scrape`, `bearer_token_file`).

**journald / py-spy** — Linux's system log and a Python profiler that can dump the stack of a hung process — used to diagnose the engine-wedge incident.

**llama-benchy** — The benchmark tool used for single-stream latency sweeps (TTFT and tok/s across context depths).

**Throughput vs latency** — Throughput = total tok/s the box can produce across all users; latency = how fast one user's request completes. Optimizing one often hurts the other.

**Wall-clock** — Real elapsed time (as opposed to a per-step or per-token rate).

**HumanEval / MMLU / perplexity** — Model-*quality* measures: a code-generation benchmark, a broad knowledge exam, and a statistical "how surprised is the model by real text" score (lower = better). Used to check that quantization/upgrades didn't hurt quality.

---

## 14. Common vLLM flags (quick reference)

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
