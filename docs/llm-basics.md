# LLM basics

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
