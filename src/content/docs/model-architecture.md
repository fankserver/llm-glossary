---
title: "Model architecture"
---

**[Transformer](https://arxiv.org/abs/1706.03762)** — The standard neural-network architecture behind modern LLMs. Its core mechanism is **attention** (below). Most terms in this section are variations on it.

**Attention** — The mechanism that lets the model relate each token to every other token in the context ("which earlier words matter for the next word?"). It is expensive: cost grows with context length, which is why so many variants below exist.

**Dense model** — A "normal" model where *all* parameters are used for every token (e.g. "dense 30.7B" Gemma). Opposite of MoE.

**MoE (Mixture of Experts)** — An architecture where the model contains many small sub-networks ("experts") and a router activates only a few per token. This is why you see two sizes: "284B total / 13B active" means the model has 284B parameters on disk/in memory, but only ~13B do work per token — so it's much faster than a dense 284B model. Written as suffixes like `-A10B` ("10B active").

- **Routed / shared experts** — Routed experts are chosen per token by the router; shared experts always run.
- **Expert routing / auxiliary-loss-free routing** (`noaux_tc`, `e_score_correction_bias`, `topk_method`, `scoring_func`) — Config knobs describing *how* the router picks experts (DeepSeek-V3 style). Only matters when a bad config breaks model loading.

**[MLA](https://arxiv.org/abs/2405.04434) (Multi-head Latent Attention)** — DeepSeek's attention variant that stores a *compressed* version of the attention cache, making long contexts use far less GPU memory (e.g. 122K tokens of context in ~11 GiB instead of many times that).

**[GQA](https://arxiv.org/abs/2305.13245) (Grouped-Query Attention)** — A common attention optimization where several "query heads" share the same "key/value heads", shrinking the KV cache. Most modern non-DeepSeek models use it.

**DSA (DeepSeek Sparse Attention)** — DeepSeek's trick to make attention cheaper on long contexts: a small "**Lightning Indexer**" first picks the top-k most relevant earlier tokens, and full attention is only computed against those. The indexer keeps its own small cache.

**[NSA](https://arxiv.org/abs/2502.11089) (Native Sparse Attention)** — DeepSeek's earlier published sparse-attention design (same family of ideas as DSA): attend only to a selected subset of tokens instead of all of them, trained natively that way rather than bolted on afterwards.

**SWA (Sliding-Window Attention)** — Attention limited to a fixed window of recent tokens (e.g. `sliding_window=128`) instead of the whole context. Cheap, but the layer can't see far back; models mix SWA layers with full-attention layers.

**c4a / c128a** — Shorthand in DeepSeek-V4 for compressed global-attention layers with a per-layer KV compression ratio of 4 or 128 (higher = more compressed = cheaper).

**SSM (State-Space Model) / [Mamba](https://arxiv.org/abs/2312.00752) / [Mamba-2](https://arxiv.org/abs/2405.21060)** — An alternative to attention: instead of looking back at all tokens, the layer maintains a running "hidden state" that is updated token by token (like a summary it carries forward). Much cheaper for long contexts. "**Hybrid**" models interleave Mamba/SSM layers with attention layers.

**[GDN](https://arxiv.org/abs/2412.06464) (Gated DeltaNet)** — Qwen's hybrid linear-attention/recurrent layer type (used in ~75% of Qwen3.5/3.6 layers). Like SSMs, it carries a running state — which has practical consequences: the state can't be checkpointed per token, so features like prefix caching and speculative-decoding rollback don't work naturally with it.

**Encoder / decoder** — Two transformer flavors. *Decoders* generate text left-to-right (all chat LLMs). *Encoders* read the whole input at once and output a representation (classic for embeddings, e.g. BERT-style). Matters for embedding models: decoder-based embedders need **last-token pooling** (take the representation of the final token), encoders typically use mean/CLS pooling.

**lm-head** — The model's final layer that converts its internal representation into scores over the vocabulary (i.e. "which token comes next"). Sometimes quantized separately.

**Multimodal / vision model** — A model that accepts images as well as text. `--language-model-only` disables the vision part to save memory when only text is needed.

**[YaRN](https://arxiv.org/abs/2309.00071)** — A technique to stretch a model's context window beyond what it was trained on (e.g. extending to 1M tokens).

**CoT (Chain of Thought) / thinking / reasoning** — The model "thinks out loud" before answering, emitting reasoning inside special tags (e.g. `<think>…</think>`). Serving-side knobs: `enable_thinking`, `reasoning_effort`, `thinking_token_budget`. The **reasoning parser** (below) strips this into a separate `reasoning` field so clients don't see it mixed into the answer.

**model_type** — The architecture identifier in a HuggingFace model config (`deepseek_v3`, `qwen3_5_moe`, `nemotron_h`, …). Tells the engine which code path to load.
