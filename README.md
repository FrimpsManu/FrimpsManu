# Augustine Manu-Frimpong

I build LLM inference systems and do research on making language models cheaper to trust.
Computer science, with a mathematics minor, at Grambling State University. Looking for software engineering, AI infrastructure, and ML systems work.

## Selected work

**[kvserve](https://github.com/FrimpsManu/kvserve)** — LLM inference engine from first principles
Paged KV cache, continuous batching with chunked prefill, prefix caching, a Triton paged-attention kernel, CUDA graphs, and an OpenAI-compatible streaming server. Reaches 98–100% of vLLM's goodput up to 32 req/s on an RTX 4090. Every scheduling optimisation is verified token-for-token against Hugging Face `transformers`.

**[hallucination-detection-ddre](https://github.com/FrimpsManu/hallucination-detection-ddre)** — preregistered research
Replaces one step of Bayesian Sequential Estimation (Wang et al., EMNLP 2023) with direct density-ratio estimation. 70% less running time per sentence at equal detection quality on HELM, confirmed on two preregistered tests. Paper in progress.

**[patchahead](https://github.com/FrimpsManu/patchahead)** — on [PyPI](https://pypi.org/project/patchahead/)
When an API you depend on changes, PatchAhead finds the breaking code, patches a copy of your project, and runs your tests. It only calls a fix done when a test that failed before passes after. Ships as a CLI and a GitHub Action.

**[pqc-flower](https://github.com/FrimpsManu/pqc-flower)** — post-quantum federated learning
Flower with ML-DSA-65 signed model updates over hybrid ML-KEM TLS, measured against a simulated quantum attacker. Group project.

## Open source

- vLLM — [#63390](https://github.com/vllm-project/vllm/pull/63390): coalesce consecutive KV-cache `BlockRemoved` events

## Tools

Python, PyTorch, Triton, CUDA, C · FastAPI, gRPC · TypeScript and React when a project needs a front end

## Links

[LinkedIn](https://www.linkedin.com/in/augustinefrimp)
