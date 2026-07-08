// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

export default defineConfig({
  site: 'https://fank.github.io',
  base: '/llm-glossary',
  integrations: [
    starlight({
      title: 'LLM Serving Glossary',
      description: 'Plain-language glossary of LLM & inference-serving terminology',
      social: [
        { icon: 'github', label: 'GitHub', href: 'https://github.com/fank/llm-glossary' },
      ],
      editLink: {
        baseUrl: 'https://github.com/fank/llm-glossary/edit/main/',
      },
      lastUpdated: true,
      sidebar: [
        { label: 'Home', slug: 'index' },
        { label: 'LLM basics', slug: 'llm-basics' },
        { label: 'Model architecture', slug: 'model-architecture' },
        { label: 'How inference works', slug: 'inference' },
        { label: 'Speculative decoding', slug: 'speculative-decoding' },
        { label: 'Quantization', slug: 'quantization' },
        { label: 'Fine-tunes & model names', slug: 'finetunes-and-model-names' },
        { label: 'Multi-GPU & parallelism', slug: 'parallelism' },
        { label: 'GPU hardware', slug: 'gpu-hardware' },
        { label: 'Tokens, templates & parsing', slug: 'tokens-templates-parsing' },
        { label: 'Embeddings & RAG', slug: 'embeddings-rag' },
        { label: 'Serving engines & runtimes', slug: 'serving-engines' },
        { label: 'Platform (PCAI / Kubernetes)', slug: 'platform' },
        { label: 'Monitoring & benchmarking', slug: 'monitoring-benchmarking' },
        { label: 'Common vLLM flags', slug: 'vllm-flags' },
      ],
    }),
  ],
});
