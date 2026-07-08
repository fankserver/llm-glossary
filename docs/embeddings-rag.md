# Embeddings & RAG

**Embedding** — A list of numbers (a vector) representing a text's *meaning*, so similar texts get nearby vectors. Produced by dedicated (small) embedding models, often cheap enough to run on CPU.

**RAG (Retrieval-Augmented Generation)** — Look up relevant documents by embedding similarity first, then paste them into the LLM's prompt so it answers from your data. The main reason embedding models are served alongside LLMs.

**Dense / sparse / multi-vector (ColBERT) embeddings** — Three retrieval-vector flavors (bge-m3 produces all three): *dense* = one meaning vector (the OpenAI `/embeddings` kind); *sparse* = keyword-weight style; *ColBERT* = one small vector per token, matched late.

**Reranking / reranker** — A second-stage model that re-scores retrieved candidates against the query for better final ordering.

**Pooling** — How per-token outputs are collapsed into one embedding vector (mean, CLS, or **last-token** for decoder-based embedders).

**Thread pinning** — On CPU inference, fixing the number of worker threads (llama.cpp `-t`, `OMP_NUM_THREADS`) to match the container's CPU limit; oversubscription tanks throughput.
