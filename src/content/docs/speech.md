---
title: "Speech (TTS / ASR)"
---

Speech models are the other thing (besides [embeddings](../embeddings-rag/)) commonly self-hosted alongside LLMs — every local voice assistant is speech-in → LLM → speech-out. This section covers the serving vocabulary, not speech science.

**TTS (Text-To-Speech; speech synthesis, voice synthesis)** — Generating spoken audio from text. Modern local TTS models are often small (1–8B) and run on the same GPU infrastructure as LLMs.

**STT / ASR (Speech-To-Text / Automatic Speech Recognition)** — Transcribing audio to text. The two abbreviations mean the same thing; papers say ASR, products say STT.

**[Whisper](https://arxiv.org/abs/2212.04356)** — OpenAI's open-weights ASR family; the de-facto standard for local transcription. Like llama.cpp for LLMs, it has optimized runtimes: **[faster-whisper](https://github.com/SYSTRAN/faster-whisper)** (CTranslate2-based) and **[whisper.cpp](https://github.com/ggml-org/whisper.cpp)** (C++, great on CPU/Apple Silicon).

**Voice cloning / zero-shot cloning / reference audio** — Making a TTS model speak in a specific voice from just a short audio sample ("reference audio"), with no training. *Zero-shot* = works for voices never seen in training. Model cards advertising "voice cloning" mean this.

**[Neural audio codec](https://arxiv.org/abs/2210.13438) / audio tokens** — The trick behind modern TTS: a codec model (EnCodec-style) compresses audio into discrete **tokens**, and the TTS model is essentially an LLM that generates those audio tokens instead of text tokens. This is why recent speech models run on LLM serving engines and inherit LLM concepts (sampling, temperature, context).

**Vocoder** — The final stage that turns an intermediate acoustic representation (audio tokens or a spectrogram) back into an actual waveform. Usually bundled inside the model; only matters when a card mentions swapping it.

**RTF (Real-Time Factor)** — The speed metric of speech serving: processing time ÷ audio duration. RTF 0.1 = generating/transcribing 10 seconds of audio per second of compute. RTF < 1 is the minimum for live use; it's the speech equivalent of tok/s.

**Streaming TTS / streaming ASR** — Emitting audio while the text is still being synthesized (or transcripts while audio is still arriving) instead of waiting for the whole input. The speech equivalent of token streaming; determines whether a voice assistant feels instant or laggy.

**Diarization** — Splitting a transcript by *who* spoke when ("speaker 1 / speaker 2"). A separate model stage on top of ASR; the `-diarize` suffix in repo names.

**Sample rate / mono 16 kHz** — The practical input gotcha: most ASR models expect a fixed format (typically mono 16 kHz WAV). Feeding 44.1 kHz stereo without resampling degrades transcription or fails outright. TTS output is commonly 24 kHz.
