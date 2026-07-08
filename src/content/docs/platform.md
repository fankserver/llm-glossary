---
title: "Platform (PCAI / Kubernetes)"
---

**HPE PCAI (Private Cloud AI)** — HPE's on-prem AI appliance: a managed Kubernetes cluster with a model-serving layer. Typically locked-down (limited volume mounts, pod logs, SSH).

**Kubernetes (K8s)** — The container-orchestration system underneath. Relevant vocabulary: **pod** (running container instance), **PVC** (persistent volume claim — a chunk of storage, e.g. for model weights), **RWX** (storage mountable by many pods at once), **hostPath** (mounting a node directory — often forbidden by cluster policy), **/dev/shm** (shared-memory filesystem pods use for inter-process communication), **CRD** (custom resource definition — a K8s API extension), **Kyverno** (a policy engine that enforces/mutates cluster rules).

**[KServe](https://kserve.github.io/website/)** — The model-serving framework on PCAI. An **InferenceService (isvc)** is one deployed model (with its URL and bearer-token **JWT** auth); a **ServingRuntime** defines the container/args used to run it; **autoscaling/replicas** control how many copies run.

**MLIS** — PCAI's model-inventory API (`/api/v1/models`). Important nuance: it lists model *definitions*, not what is actually running right now.

**aioli** — Name/namespace of PCAI's serving stack.

**HF / HuggingFace** — The standard hub for downloading model weights (`HF_TOKEN` = auth token; **Xet** = its newer CDN download backend).

**Engine wedge** — Informal term for a serving incident where the engine silently stalls without crashing (typical root cause: memory swapping; related vocab: **swap-thrash**, **vm.swappiness**, **OOM-killer**, **demand-paging** — Linux memory-management behaviors that can starve the GPU process).

**frp / frpc** — A reverse-proxy tunnel tool for exposing a machine behind NAT/firewall (e.g. a home lab serving endpoint) to the internet.
