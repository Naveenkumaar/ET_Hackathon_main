<div align="center">

# 🏭 Industrial Knowledge Intelligence — Hybrid GraphRAG

**Answering industrial-maintenance questions by tracing relationships across documents that share no keywords with each other.**

Knowledge graph + vector search, fused via Reciprocal Rank Fusion (RRF).
Built for **ET AI Hackathon 2.0 · Problem #8: Industrial Knowledge Intelligence**.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Claude API](https://img.shields.io/badge/LLM-Claude_API-D97757)](https://www.anthropic.com/)
[![ChromaDB](https://img.shields.io/badge/Vectors-ChromaDB-4B32C3)](https://www.trychroma.com/)
[![NetworkX](https://img.shields.io/badge/Graph-NetworkX-2C5BB4)](https://networkx.org/)
[![Docker](https://img.shields.io/badge/Container-Docker-2496ED?logo=docker&logoColor=white)](ET_HACKATHLAON--main/Dockerfile)
[![CI](https://img.shields.io/badge/CI-pytest-0A9EDC)](ET_HACKATHLAON--main/.github/workflows/tests.yml)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](ET_HACKATHLAON--main/LICENSE)

**[📄 Full write-up](ET_HACKATHLAON--main/Industrial_Knowledge_Intelligence_Detailed_Document.pdf)** ·
**[🗺️ Architecture diagram](ET_HACKATHLAON--main/architecture_diagram.png)** ·
**[📊 Deck](ET_HACKATHLAON--main/Industrial_Knowledge_Intelligence_Deck.pptx)** ·
**[🎬 Demo video](ET_HACKATHLAON--main/demo_video.mp4)**

</div>

> **Note on layout:** the runnable project currently lives in the [`ET_HACKATHLAON--main/`](ET_HACKATHLAON--main/) directory. All paths below are relative to that folder unless linked otherwise.

---

## Table of contents

- [Why this exists](#why-this-exists)
- [The star demo question](#the-star-demo-question)
- [Quick start](#quick-start)
- [Architecture](#architecture)
- [Benchmark results](#benchmark-results)
- [Repo structure](#repo-structure)
- [Dataset](#dataset)
- [Engineering notes](#engineering-notes)

---

## Why this exists

Industrial failures are almost never recorded in one place. The early warning sits in an
inspection report, the reason nobody acted on it sits in a maintenance log, the alarm sits in a
sensor trend, the failure itself sits in an incident report, and what *should* have happened sits
in a procedure document. **None of these documents share keywords** — so pure keyword or vector
search can only ever find the one document you already knew to look for.

This system builds a **knowledge graph** across the whole corpus (entities merged by canonical ID,
so "Pump P-204" and "P-204" become one node) and **fuses graph traversal with semantic vector
search via RRF**. That lets it recover the *chain* — the warning three weeks earlier that a
keyword search has no way to reach.

---

## The star demo question

> **"Was there any early warning before the Pump P-204 failure, and what should have been done according to procedure?"**

This is the flagship test of the whole hybrid approach. The answer requires connecting **five
documents that share no common keywords**:

| # | Document | What it contributes |
|---|----------|---------------------|
| 1 | Inspection report | Elevated bearing vibration flagged **three weeks** before failure |
| 2 | Maintenance log | Recommended bearing replacement **deferred** — parts shortage |
| 3 | Vibration trend log | Alarm threshold **crossed five days out** |
| 4 | Incident report | Records the resulting failure |
| 5 | Procedure document | Specifies what **should** have happened instead |

Pure keyword or vector search returns only document #4. **Graph traversal is what recovers the
earlier warning chain.**

---

## Quick start

```bash
cd ET_HACKATHLAON--main
pip install -r requirements.txt

# System dependency: Tesseract OCR (only needed when (re)building the vector
# store, since 3 corpus PDFs are scanned images):
#   macOS:   brew install tesseract
#   Debian:  apt install tesseract-ocr
#   Windows: https://github.com/UB-Mannheim/tesseract/wiki

# macOS / Linux
export ANTHROPIC_API_KEY="your-key"
# Windows PowerShell
$env:ANTHROPIC_API_KEY="your-key"

streamlit run app.py
```

The first run downloads the `BAAI/bge-small-en-v1.5` embedding model (~130 MB) and builds the
vector store / knowledge graph from the corpus if they don't already exist under `data/` (no API
key needed for the build itself).

**Run the tests:**

```bash
python -m pytest tests/
```

The suite validates the eval dataset, graph invariants, chunking, citation filtering, and ranking
determinism. CI runs it on every push (`.github/workflows/tests.yml`).

**Run in Docker** (embedding model and indexes baked into the image at build time):

```bash
docker build -t ikig .
docker run -p 8501:8501 -e ANTHROPIC_API_KEY="your-key" ikig
```

---

## Architecture

Seven stages, each a standalone module in `ingest/`:

| Stage | Module | Role |
|-------|--------|------|
| 1 | `loaders.py` | PyMuPDF text extraction, with **pytesseract OCR fallback** for scanned pages |
| 2 | `extractor.py` | Claude API extracts **typed entities & relationships** per document, cached to disk |
| 3 | `graph_builder.py` | Assembles a **NetworkX graph** across the corpus, merging entities by canonical ID |
| 4 | `vector_builder.py` | Embeds each page into a persistent **ChromaDB** collection |
| 5 | `retriever.py` | **RRF fusion** — combines semantic search with graph traversal |
| 6 | `synthesis.py` | Claude API turns retrieved docs into an answer with **citations + confidence** |
| 7 | `app.py` | The user-facing **Streamlit** chat UI |

`evaluate.py` is an eighth piece, sitting outside the numbered pipeline: a benchmark harness
scoring retrieval quality against hand-labeled ground truth.

---

## Benchmark results

Measured **17 Jul 2026** after the chunking/embedding upgrade (windowed page chunks +
`bge-small-en-v1.5`), verified **byte-identical across three consecutive runs**.

| Metric | Hybrid | Vector-only |
|--------|:------:|:-----------:|
| **Star demo Q01 — recall@12** | **1.00** | 0.80 |
| Overall recall@12 (8 questions) | **0.83** | 0.81 |
| Overall recall@5 (8 questions) | 0.70 | **0.76** |
| Control (keyword-answerable) questions | 1.00 | 1.00 |
| Entity extraction **Macro-F1** | **0.8411** | — |

*(Entity F1 excludes DATE, which has zero labeled instances in the ground-truth set.)*

**On the star question, the hybrid retrieves the full five-document chain (1.00 vs 0.80)** — the two
documents the vector baseline misses (VS-204 and M-118, whose text never mentions "P-204") are
supplied by graph traversal.

**Honest reading.** Upgrading the embedding model lifted the *vector baseline* substantially
(0.69 → 0.81 @12), so the hybrid's aggregate margin is now small — and at the stricter @5 cutoff,
hop-0 graph noise costs a little precision on single-document questions. **The graph earns its
place on the multi-hop chain question this system was built for (Q01: 1.00 vs 0.80), not on
aggregate averages.** Q02 remains capped by hop-0 tie-flooding (root-cause trace in `CLAUDE.md`).

---

## Repo structure

```
ET_HACKATHLAON--main/
  ingest/          the 7-stage pipeline (see Architecture above)
  data/corpus/     the locked document corpus (synthetic, real OEM manuals, scanned)
  data/eval/       benchmark questions, ground truth, results, FailureSensorIQ reference data
  app.py           Streamlit UI — the live app
  evaluate.py      benchmark harness
  scripts/         one-off scripts that built the locked dataset — not part of running the app
                   (generate_corpus.py, make_scanned.py, download_failuresensoriq.py)
  tests/           pytest suite run in CI
```

---

## Dataset

The corpus is a locked mix of:

- a **synthetic P-204 pump near-miss chain** (11 PDFs, planted for the star demo),
- **3 real OEM pump manuals**,
- **3 degraded scans** for OCR testing, and
- IBM's **FailureSensorIQ** dataset (CC-BY-4.0) as reference evaluation data.

Full provenance, sources, and structure: [`data/DATASET_README.md`](ET_HACKATHLAON--main/data/DATASET_README.md).
FailureSensorIQ-specific reference material (citation, subsets, benchmark details):
[`data/eval/FAILURESENSORIQ_NOTES.md`](ET_HACKATHLAON--main/data/eval/FAILURESENSORIQ_NOTES.md).

---

## Engineering notes

- **Grounded citations.** Model citations are validated against the retrieved context before
  rendering — a hallucinated doc reference can never display as a source.
- **Explainability.** Each answer includes a *"Why these documents"* expander showing the
  knowledge-graph paths (works offline; all JS inlined).
- **Caching.** Answers are cached in `.synthesis_cache/`; repeat questions skip the API and show a
  ⚡ badge.
- **Observability.** Every query appends a JSON trace line to `logs/query_traces.jsonl` (latency
  split, token usage, rankings, citations).
- **Retrieval robustness.** Entity matching is alias- and punctuation-tolerant ("p204" finds
  P-204); scanned/clean duplicate documents collapse to one top-k slot; follow-up questions reuse
  the conversation so *"was it acted upon?"* resolves against the previous answer.
- **Configurable.** Model names and top-k are overridable via `SYNTHESIS_MODEL`,
  `EXTRACTION_MODEL`, and `RETRIEVAL_TOP_K` env vars. `evaluate.py` reports recall at both K=5 and
  K=12 — on a 17-document corpus the @5 figure is the meaningful one.

---

<div align="center">

Built for **ET AI Hackathon 2.0** · Problem #8 · Industrial Knowledge Intelligence
[MIT License](ET_HACKATHLAON--main/LICENSE)

</div>
