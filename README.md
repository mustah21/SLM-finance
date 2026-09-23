# Finance LLM Project: Fine-Tuning & RAG

A two-sprint project exploring two approaches to adapting a large language model (Qwen) for the financial domain: **supervised fine-tuning** and **retrieval-augmented generation (RAG)** — with a comparative evaluation of both against each other and against a TF-IDF baseline.

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

