# Third-party notices

This project is licensed under Apache-2.0 (see `LICENSE`). It depends on the components
below. None of them is copyleft and none restricts commercial use, so no obligation flows
back onto this project's own licence — the notices here exist because Apache-2.0 §4(d) asks
for attribution to travel with redistribution, and because a reader should be able to see
what was checked without re-deriving it.

## Machine-learning model

The reranker model is **downloaded at runtime** from the Hugging Face Hub and is not
vendored into this repository (see `.gitignore`). Redistributing a build that bundles the
weights would additionally require shipping the model's own licence text.

| Component | Role | Licence | Source |
|---|---|---|---|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | optional, opt-in via `rerank_model` in `config.py` | Apache-2.0 | <https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2> |
| `BAAI/bge-reranker-v2-m3` | **default local reranker** | Apache-2.0 | <https://huggingface.co/BAAI/bge-reranker-v2-m3> |
| `BAAI/bge-reranker-base` | evaluated, not used | MIT | <https://huggingface.co/BAAI/bge-reranker-base> |

The default, `bge-reranker-v2-m3`, is built on `BAAI/bge-m3`, an XLM-RoBERTa architecture.
`ms-marco-MiniLM-L-6-v2` is a BERT-based cross-encoder trained on MS MARCO by UKP Lab /
Sentence Transformers. Cite the default as:

> Chen, Xiao, Zhang, Luo, Lian, Liu. *BGE M3-Embedding: Multi-Lingual, Multi-Functionality,
> Multi-Granularity Text Embeddings Through Self-Knowledge Distillation*, 2024.
> arXiv:2402.03216

## Python libraries added for local reranking

| Component | Licence |
|---|---|
| `sentence-transformers` | Apache-2.0 |
| `transformers` | Apache-2.0 |
| `torch` | BSD-3-Clause |
| `tokenizers` | Apache-2.0 |
| `safetensors` | Apache-2.0 |
| `huggingface-hub` | Apache-2.0 |
| `scikit-learn` | BSD-3-Clause |
| `scipy`, `numpy` | BSD-3-Clause |

## Rejected on licence grounds

Recorded so the decision is not silently revisited:

**`jinaai/jina-reranker-v2-base-multilingual` — CC-BY-NC-4.0, rejected.** It is the closest
multilingual competitor and is supported out of the box by `fastembed`, which made it the
path of least resistance. Its model card states: *"This model repository is licenced for
research and evaluation purposes under CC-BY-NC-4.0. For commercial usage, please refer to
Jina AI's APIs."* The **NC** clause bars commercial use. `BAAI/bge-reranker-v2-m3` is
Apache-2.0 and carries no such restriction.

## Source data

The EU AI Act text is retrieved from EUR-Lex / CELLAR. © European Union,
<https://eur-lex.europa.eu>. Reuse of Commission documents is governed by Decision
2011/833/EU. This project stores retrieved text locally and does not redistribute it.
