---
title: "Serving engines & runtimes"
---

**[vLLM](https://docs.vllm.ai)** — The most widely used open-source LLM serving engine for GPUs. Known for PagedAttention, continuous batching, and an OpenAI-compatible API. Fast-moving; production setups often pin specific builds/nightlies.

**[SGLang](https://docs.sglang.ai)** — A competing GPU serving engine, notable for its speculative-decoding implementation ("Spec V2").

**[llama.cpp](https://github.com/ggml-org/llama.cpp)** — A lightweight C++ inference engine, great on CPU; uses GGUF files.

**[Infinity](https://github.com/michaelfeil/infinity)** — A Python (torch/optimum) embedding server, commonly used for CPU embedding serving.

**[TEI](https://github.com/huggingface/text-embeddings-inference) (Text Embeddings Inference)** — HuggingFace's Rust embedding server (evaluated; MKL/ONNX issues).

**[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) / [Ollama](https://github.com/ollama/ollama)** — Other inference engines referenced for comparison (NVIDIA's optimized engine; a hobbyist-friendly local runner).

**[OpenAI-compatible API](https://platform.openai.com/docs/api-reference/chat)** — The de-facto standard HTTP API shape (`/v1/chat/completions`, `/v1/embeddings`); most engines speak it so any OpenAI client works against self-hosted models.

**[LiteLLM](https://docs.litellm.ai)** — A proxy/router that presents many backends behind one OpenAI-style API with keys and quotas (commonly paired with chat UIs like LibreChat).

**mmap / `--no-mmap`** — llama.cpp loads GGUFs via **memory-mapping** by default: the OS pages weights in from disk on demand, so start-up is instant and memory is shared between processes. Pitfall: if the model is larger than RAM, demand-paging thrashes and the load looks like a hang — `--no-mmap` (read the file straight into memory) fixes it.

**[fastsafetensors](https://arxiv.org/abs/2505.23072) / runai_streamer / safetensors** — **[safetensors](https://github.com/huggingface/safetensors)** is the standard safe weight-file format; the other two are fast loaders that stream weights into GPU memory at startup (model load takes minutes otherwise).
