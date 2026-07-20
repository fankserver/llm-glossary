---
title: "Model architecture"
---

**[Transformer](https://arxiv.org/abs/1706.03762)** — The standard neural-network architecture behind modern LLMs. Its core mechanism is **attention** (below). Most terms in this section are variations on it.

**Attention** — The mechanism that lets the model relate each token to every other token in the context ("which earlier words matter for the next word?"). It is expensive: cost grows with context length, which is why so many variants below exist.

**Dense model** — A "normal" model where *all* parameters are used for every token (e.g. "dense 30.7B" [Gemma](https://arxiv.org/abs/2607.02770)). Opposite of MoE.

**[MoE](https://arxiv.org/abs/1701.06538) (Mixture of Experts)** — An architecture where the model contains many small sub-networks ("experts") and a router activates only a few per token. This is why you see two sizes: "284B total / 13B active" means the model has 284B parameters on disk/in memory, but only ~13B do work per token — so it's much faster than a dense 284B model. Written as suffixes like `-A10B` ("10B active").

- **[Routed / shared experts](https://arxiv.org/abs/2401.06066)** — Routed experts are chosen per token by the router; shared experts always run.
- **Expert routing / auxiliary-loss-free routing** (`noaux_tc`, `e_score_correction_bias`, `topk_method`, `scoring_func`) — Config knobs describing *how* the router picks experts (DeepSeek-V3 style). Only matters when a bad config breaks model loading. **Hash routing** = the expert is picked by token ID instead of a learned router (used in some early layers: cheap, stable per-token paths).

**[MLA](https://arxiv.org/abs/2405.04434) (Multi-head Latent Attention)** — DeepSeek's attention variant that stores a *compressed* version of the attention cache, making long contexts use far less GPU memory (e.g. 122K tokens of context in ~11 GiB instead of many times that).

**[GQA](https://arxiv.org/abs/2305.13245) (Grouped-Query Attention)** — A common attention optimization where several "query heads" share the same "key/value heads", shrinking the KV cache. Most modern non-DeepSeek models use it.

**[DSA](https://arxiv.org/abs/2512.02556) (DeepSeek Sparse Attention)** — DeepSeek's trick to make attention cheaper on long contexts: a small "**Lightning Indexer**" first picks the top-k most relevant earlier tokens, and full attention is only computed against those. The indexer keeps its own small cache.

**[NSA](https://arxiv.org/abs/2502.11089) (Native Sparse Attention)** — DeepSeek's earlier published sparse-attention design (same family of ideas as DSA): attend only to a selected subset of tokens instead of all of them, trained natively that way rather than bolted on afterwards.

**[SWA](https://arxiv.org/abs/2004.05150) (Sliding-Window Attention)** — Attention limited to a fixed window of recent tokens (e.g. `sliding_window=128`) instead of the whole context. Cheap, but the layer can't see far back; models mix SWA layers with full-attention layers.

**[CSA](https://arxiv.org/abs/2606.19348) (Compressed Sparse Attention)** — DeepSeek-V4's attention for mid-range memory: every 4 tokens are *learned-compressed* into one KV entry (the model learns how much each token contributes), then a Lightning-Indexer-style top-k selection (as in DSA) picks only the most relevant compressed blocks to attend to. Compression + sparse retrieval.

**[HCA](https://arxiv.org/abs/2606.19348) (Heavily Compressed Attention)** — The long-range companion: every 128 tokens become one KV entry (32× lighter than CSA), and the model attends *densely* over that heavily compressed memory — a cheap global view rather than precise retrieval. V4 interleaves CSA and HCA layers 1:1, each with an extra 128-token sliding-window branch so the newest tokens are never summarized away. The shorthands **c4a / c128a** refer to these compression ratios. Note: attention cost still grows quadratically, just much slower — this is not linear attention.

**[MHC](https://arxiv.org/abs/2512.24880) (Manifold-Constrained [Hyper-Connections](https://arxiv.org/abs/2409.19606))** — A change to the transformer's residual stream (the "conveyor belt" carrying information between layers): instead of one stream, several parallel streams with learned mixing before/after each layer, increasing representational capacity across depth.

**[xHC](https://arxiv.org/abs/2607.14530) (Expanded Hyper-Connections)** — Scales the residual stream to 16 parallel streams (MHC stops at ~4) by updating only a few per layer while reading from all. **xHC-Flash** = the memory-efficient deployment variant.

**SSM (State-Space Model) / [Mamba](https://arxiv.org/abs/2312.00752) / [Mamba-2](https://arxiv.org/abs/2405.21060)** — An alternative to attention: instead of looking back at all tokens, the layer maintains a running "hidden state" that is updated token by token (like a summary it carries forward). Much cheaper for long contexts. "**[Hybrid](https://arxiv.org/abs/2403.19887)**" models interleave Mamba/SSM layers with attention layers. The serving payoff: only the attention layers keep a [KV cache](../inference/), so in a mostly-Mamba hybrid (e.g. 6 attention layers out of 52) the per-sequence cache stays near-constant as context grows — decode throughput at 256K context is about the same as at 4K, where a dense model's collapses.

**[GDN](https://arxiv.org/abs/2412.06464) (Gated DeltaNet)** — Qwen's hybrid linear-attention/recurrent layer type (used in ~75% of Qwen3.5/3.6 layers). Like SSMs, it carries a running state — which has practical consequences: the state can't be checkpointed per token, so features like prefix caching and speculative-decoding rollback don't work naturally with it.

**Diffusion LM (dLLM) / text diffusion** — An alternative to left-to-right generation: the model starts from a fully masked/noisy block of text and refines it over several "denoising" steps, filling in whole passages in parallel (same idea as image diffusion). Promises much faster generation; used both for full models and as a drafting trick (see DFlash under [Speculative decoding](../speculative-decoding/)).

**Encoder / decoder** — Two transformer flavors. *Decoders* generate text left-to-right (all chat LLMs). *Encoders* read the whole input at once and output a representation (classic for embeddings, e.g. BERT-style). Matters for embedding models: decoder-based embedders need **last-token pooling** (take the representation of the final token), encoders typically use mean/CLS pooling.

**lm-head** — The model's final layer that converts its internal representation into scores over the vocabulary (i.e. "which token comes next"). Sometimes quantized separately.

**[Multimodal / vision model](https://arxiv.org/abs/2304.08485)** — A model that accepts images as well as text. `--language-model-only` disables the vision part to save memory when only text is needed.

**[YaRN](https://arxiv.org/abs/2309.00071)** — A technique to stretch a model's context window beyond what it was trained on (e.g. extending to 1M tokens).

**[CoT](https://arxiv.org/abs/2201.11903) (Chain of Thought) / thinking / reasoning** — The model "thinks out loud" before answering, emitting reasoning inside special tags (e.g. `<think>…</think>`). Serving-side knobs: `enable_thinking`, `reasoning_effort`, `thinking_token_budget`. The **reasoning parser** (below) strips this into a separate `reasoning` field so clients don't see it mixed into the answer.

**model_type** — The architecture identifier in a HuggingFace model config (`deepseek_v3`, `qwen3_5_moe`, `nemotron_h`, …). Tells the engine which code path to load.
