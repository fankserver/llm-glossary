---
title: "Monitoring & benchmarking"
---

**[Prometheus](https://prometheus.io/docs/introduction/overview/) / `/metrics` / scrape** — The standard metrics pipeline: services expose numbers at `/metrics`; Prometheus "scrapes" (fetches) them periodically. vLLM exposes KV-cache usage, request counts, etc. there.

**Grafana / Grafana Cloud / Alloy** — Dashboards for those metrics; **[Alloy](https://grafana.com/docs/alloy/latest/)** is the collector agent that scrapes and forwards them (config blocks like `prometheus.scrape`, `bearer_token_file`).

**journald / [py-spy](https://github.com/benfred/py-spy)** — Linux's system log and a Python profiler that can dump the stack of a hung process — used to diagnose the engine-wedge incident.

**llama-benchy** — The benchmark tool used for single-stream latency sweeps (TTFT and tok/s across context depths).

**Throughput vs latency** — Throughput = total tok/s the box can produce across all users; latency = how fast one user's request completes. Optimizing one often hurts the other.

**Interleaved A/B (benchmark variance)** — Restart-to-restart throughput of the same setup can vary ±20% (clocks, memory layout, cache state), so "before" and "after" numbers from different server starts mostly measure noise. Valid comparisons interleave both variants on a single running instance.

**Wall-clock** — Real elapsed time (as opposed to a per-step or per-token rate).

**[HumanEval](https://arxiv.org/abs/2107.03374) / [MMLU](https://arxiv.org/abs/2009.03300) / perplexity** — Model-*quality* measures: a code-generation benchmark, a broad knowledge exam, and a statistical "how surprised is the model by real text" score (lower = better). Used to check that quantization/upgrades didn't hurt quality.

**[pass@k](https://arxiv.org/abs/2107.03374)** — "Model gets k attempts; does at least one pass?" — the standard scoring for code/math benchmarks. pass@1 = single-try accuracy; a big pass@k-vs-pass@1 gap means the model can solve it but is inconsistent.
