# Serving engines & runtimes

**[vLLM](https://docs.vllm.ai)** — The most widely used open-source LLM serving engine for GPUs. Known for PagedAttention, continuous batching, and an OpenAI-compatible API. Fast-moving; production setups often pin specific builds/nightlies.

**[SGLang](https://docs.sglang.ai)** — A competing GPU serving engine, notable for its speculative-decoding implementation ("Spec V2").

**[llama.cpp](https://github.com/ggml-org/llama.cpp)** — A lightweight C++ inference engine, great on CPU; uses GGUF files.

**Infinity** — A Python (torch/optimum) embedding server, commonly used for CPU embedding serving.

**TEI (Text Embeddings Inference)** — HuggingFace's Rust embedding server (evaluated; MKL/ONNX issues).

**TensorRT-LLM / Ollama** — Other inference engines referenced for comparison (NVIDIA's optimized engine; a hobbyist-friendly local runner).

**OpenAI-compatible API** — The de-facto standard HTTP API shape (`/v1/chat/completions`, `/v1/embeddings`); most engines speak it so any OpenAI client works against self-hosted models.

**[LiteLLM](https://docs.litellm.ai)** — A proxy/router that presents many backends behind one OpenAI-style API with keys and quotas (commonly paired with chat UIs like LibreChat).

**fastsafetensors / runai_streamer / safetensors** — **safetensors** is the standard safe weight-file format; the other two are fast loaders that stream weights into GPU memory at startup (model load takes minutes otherwise).
