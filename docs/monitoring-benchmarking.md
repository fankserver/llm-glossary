# Monitoring & benchmarking

**[Prometheus](https://prometheus.io/docs/introduction/overview/) / `/metrics` / scrape** — The standard metrics pipeline: services expose numbers at `/metrics`; Prometheus "scrapes" (fetches) them periodically. vLLM exposes KV-cache usage, request counts, etc. there.

**Grafana / Grafana Cloud / Alloy** — Dashboards for those metrics; **Alloy** is the collector agent that scrapes and forwards them (config blocks like `prometheus.scrape`, `bearer_token_file`).

**journald / py-spy** — Linux's system log and a Python profiler that can dump the stack of a hung process — used to diagnose the engine-wedge incident.

**llama-benchy** — The benchmark tool used for single-stream latency sweeps (TTFT and tok/s across context depths).

**Throughput vs latency** — Throughput = total tok/s the box can produce across all users; latency = how fast one user's request completes. Optimizing one often hurts the other.

**Wall-clock** — Real elapsed time (as opposed to a per-step or per-token rate).

**HumanEval / MMLU / perplexity** — Model-*quality* measures: a code-generation benchmark, a broad knowledge exam, and a statistical "how surprised is the model by real text" score (lower = better). Used to check that quantization/upgrades didn't hurt quality.
