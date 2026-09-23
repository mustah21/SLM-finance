# Finance LLM Project: Fine-Tuning & RAG

A two-sprint project exploring two approaches to adapting a large language model (Qwen) for the financial domain: **supervised fine-tuning** and **retrieval-augmented generation (RAG)** — with a comparative evaluation of both against each other and against a TF-IDF baseline.

## Table of Contents
- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Sprint 1: Fine-Tuning](#sprint-1-fine-tuning)
- [Sprint 2: RAG](#sprint-2-rag)
- [Evaluation](#evaluation)
- [Results](#results)
- [Discussion: RAG vs Fine-Tuning vs TF-IDF](#discussion-rag-vs-fine-tuning-vs-tf-idf)
- [License](#license)

## Overview

This project builds and compares two systems for answering financial questions grounded in company filings:

1. **Fine-tuned Qwen model** — trained on question–answer pairs to learn domain-specific response behavior.
2. **RAG pipeline** — retrieves relevant passages from a document corpus via vector search and generates grounded answers.

Both are evaluated on the same held-out test set and compared against a classical TF-IDF retrieval baseline.

## Problem Statement

> *[Fill in: e.g., "Given a financial question about a company's 10-K filing, produce an accurate, source-grounded answer."]*

**Domain:** Finance (company filings / financial statements)

## Dataset

**Source:** [sweatSmile/FinanceQA](https://huggingface.co/datasets/sweatSmile/FinanceQA) — ~4,000 financial Q&A pairs extracted from company annual reports and financial statements.

| Field | Description |
|---|---|
| `COMPANY_ID` | Company + year identifier |
| `QUERY` | Financial question |
| `ANSWER` | Extracted factual answer |
| `CONTEXT` | Supporting passage from the source filing |

The `CONTEXT` field serves as the RAG corpus source; `QUERY`/`ANSWER` pairs serve as the fine-tuning supervision signal.

## Project Structure

```
.
├── data/
│   ├── raw/                    # original dataset files
│   ├── processed/              # cleaned, deduplicated data
│   └── splits/                 # train / val / test
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preparation.ipynb
│   ├── 03_fine_tuning.ipynb
│   ├── 04_embeddings_rag.ipynb
│   └── 05_evaluation.ipynb
├── src/
│   ├── data_prep.py
│   ├── fine_tune.py
│   ├── rag_pipeline.py
│   └── evaluate.py
├── models/
│   └── qwen-finetuned-adapter/ # saved LoRA adapter weights
├── vector_store/                # FAISS/Chroma index
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone <repo-url>
cd <repo-name>
pip install -r requirements.txt
```

**requirements.txt**
```
torch
transformers
datasets
accelerate
peft
trl
bitsandbytes
sentence-transformers
faiss-cpu
chromadb
ragas
evaluate
rouge-score
scikit-learn
pandas
numpy
huggingface_hub
```

## Sprint 1: Fine-Tuning

1. **Problem & domain identification** — defined the target task and financial domain.
2. **Dataset exploration** — inspected size, label quality, field structure.
3. **Data preparation** — deduplication, cleaning, train/val/test split.
4. **Training set construction** — converted to Qwen chat-template format:
   ```json
   {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
   ```
5. **Fine-tuning** — LoRA/QLoRA fine-tuning of a Qwen model via `trl.SFTTrainer`.
6. **Save & inference** — adapter saved with `peft`, loaded via `PeftModel` for generation.
7. **Evaluation** — metrics on held-out test set (see [Evaluation](#evaluation)).

## Sprint 2: RAG

1. **Embeddings** — generated with `sentence-transformers`.
2. **Vector database** — indexed with FAISS / Chroma.
3. **RAG pipeline** — query → embed → retrieve top-k chunks → prompt LLM with context → generate answer.
4. **Retrieval + generation evaluation** — precision@k, recall@k, MRR, faithfulness, answer relevancy (via RAGAS).
5. **TF-IDF baseline** — classical sparse retrieval compared against dense embedding retrieval.

## Evaluation

| Component | Metrics |
|---|---|
| Fine-tuned model (generation) | ROUGE, BLEU, exact match / F1 |
| RAG retrieval | Precision@k, Recall@k, MRR |
| RAG generation | Faithfulness, answer relevancy, context precision/recall (RAGAS) |
| TF-IDF vs embeddings | Retrieval precision/recall comparison |

## Results

> *[Fill in after running experiments: metric tables, example outputs, before/after comparisons.]*

## Discussion: RAG vs Fine-Tuning vs TF-IDF

| | Fine-tuning | RAG | TF-IDF |
|---|---|---|---|
| Learns | Behavior/format/style | Nothing (retrieval only) | Nothing (keyword matching) |
| Knowledge updates | Requires retraining | Instant (update index) | Instant (update index) |
| Factual grounding | Weak / prone to hallucination | Strong (source-grounded) | N/A (retrieval only) |
| Semantic matching | N/A | Strong | Weak (exact-term only) |
| Setup cost | High (GPU, training time) | Moderate | Low |

**Conclusion:** *[Fill in based on your results — typically RAG suits fact-grounded QA better, fine-tuning suits consistent behavior/format, and combining both often performs best.]*

## License

*[Specify license, e.g., MIT]*
