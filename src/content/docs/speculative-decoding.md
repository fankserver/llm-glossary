---
title: "Speculative decoding"
---

**[Speculative decoding](https://arxiv.org/abs/2211.17192) (spec decode)** — A decode speed-up: a small, fast "**drafter**" guesses the next several tokens, then the big model verifies the whole guess in *one* pass. Correct guesses are accepted for free; wrong ones fall back to normal decoding. Output is identical to non-speculative decoding — it's purely faster. Configured with `--speculative-config`; **k** / `num_speculative_tokens` = how many tokens are drafted per step (llama.cpp: `n_max`; its **`p_min`** stops drafting early once the drafter's confidence drops below a threshold — a simple form of adaptive draft depth, see D-Cut below).

- **Drafter / speculator / draft model** — the small model doing the guessing. **Verifier** — the big model checking.
- **Acceptance rate / accepted-per-step / eff tok/step** — how many drafted tokens the verifier accepts on average. The whole game: higher [acceptance](https://arxiv.org/abs/2302.01318) = faster. "42% acceptance, 7.33 length" = of long drafts, on average ~7 tokens survive per step.

Common methods:

**[MTP](https://arxiv.org/abs/2404.19737) (Multi-Token Prediction)** — The drafter is a small extra layer *inside* the main model (DeepSeek ships one built-in; Gemma uses a separate `-assistant` drafter checkpoint). No separate model to manage. In GGUF/llama.cpp the MTP block ships as **nextn** tensors (`--spec-type draft-mtp`) — it predicts the token *after* next from the current hidden state.

**[EAGLE](https://arxiv.org/abs/2401.15077) / [EAGLE3](https://arxiv.org/abs/2503.01840)** — A popular family of trained drafter heads that reuse the big model's internal states to draft accurately.

**[Medusa](https://arxiv.org/abs/2401.10774)** — An older multi-head drafting method (referenced for comparison).

**[DFlash](https://arxiv.org/abs/2602.06036)** — A "block-diffusion" drafter (~0.8B, from [z-lab](https://github.com/z-lab/dflash)): instead of guessing tokens one by one, it drafts a whole block of tokens in a single non-causal forward pass (a "denoising step"), like filling in a whole phrase at once.

**[DSpark](https://arxiv.org/abs/2607.05147)** — DeepSeek's block-parallel drafting scheme built on the model's own MTP weights (knobs like `dspark_block_size`), used with DeepSeek-V4 models. Its trick is *confidence scheduling*: draft further ahead when the drafter is confident, pull back when it isn't.

**[D-Cut](https://arxiv.org/abs/2607.14647)** — A fix for speculative decoding under *batching*: with many concurrent requests, verifying long drafts wastes compute on tokens that will be rejected — enough that spec decode can become slower than normal decoding. D-Cut prunes each request's verification depth adaptively (confident drafts keep more tokens, shaky ones are cut short), restoring the speed-up at high concurrency.

**IndexCache / skip_topk** — An upstream vLLM optimization (tracked in upgrades.md) that reuses DSA's top-k token selection across compressed-attention layers instead of recomputing it.

**QAT-matched drafter** — A drafter quantized/trained to match the quantized main model, so acceptance doesn't drop.
