# RAG Pipeline & Evaluation

Build a working Retrieval-Augmented Generation (RAG) pipeline over a document
knowledge base, and measure it properly. The emphasis is as much on evaluation
and evidence as on the pipeline itself: the project must produce numbers,
compare configurations, and justify the results.

---

## Objective

- Ingest a document corpus and make it retrievable.
- Build retrieval (vector + lexical + hybrid) and a grounded generation step.
- Evaluate retrieval and answer quality with standard metrics.
- Run experiments comparing configurations and report the results honestly.
- Serve it over a simple API, reproducible via Docker Compose.

---

## Knowledge base

**Primary corpus: SciFact (from the BEIR benchmark).**

- Hugging Face: `BeIR/scifact` (corpus, queries, and `qrels`).
- ~5K documents, 300 test queries - small enough to iterate quickly on
  free/local hardware.
- Ships with a `qrels` relevance-judgments file (TSV: `query-id`, `corpus-id`,
  `score`), so retrieval metrics are objective and require no manual labeling.
- Load with the `beir` library:
  ```python
  from beir.datasets.data_loader import GenericDataLoader
  corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
  ```
- Evaluate only on the provided query set; metrics are computed against the
  `qrels` for those queries.

**Allowed alternatives (all BEIR, all with qrels):** `nfcorpus` (~3.6K docs,
smaller/faster) or `fiqa` (~57K docs, more realistic scale).

**Optional real-PDF corpus:** Cornell **arXiv dataset** on Kaggle (~1.7M
articles, metadata + full-text PDFs, free). Use only if PDF ingestion is in
scope; it has no `qrels`, so pair it with a small hand-built gold set (10–20
questions with known-relevant papers).

> BEIR corpora are JSONL passage collections, not document files. The arXiv
> option is what exercises `.pdf` ingestion/extraction.

---

## Scope

### Required

| # | Deliverable |
| --- | --- |
| R1 | Load corpus, queries, and qrels; chunk documents; persist with metadata (idempotent re-runs) |
| R2 | Vector retrieval: embeddings in pgvector, ANN index (HNSW/IVFFlat) |
| R3 | Lexical retrieval: PostgreSQL FTS (`tsvector`, `ts_rank_cd`) and/or BM25 via `rank-bm25` |
| R4 | Hybrid retrieval: Reciprocal Rank Fusion of vector + lexical, toggleable |
| R5 | Generation: local LLM answers the query grounded in retrieved context, with a clear prompt template |
| R6 | FastAPI endpoint `/query` returning the answer plus the retrieved sources |
| R7 | Retrieval metrics against qrels: Recall@k, nDCG@k, MRR, MAP |
| R8 | Answer metrics: faithfulness + answer relevancy (Ragas or a documented LLM-as-Judge) on a sample |
| R9 | Experiment report: a metrics table comparing configurations, with interpretation |
| R10 | Docker Compose (app, Postgres+pgvector) brings the system up with a seed step |
| R11 | README documenting architecture, setup, design decisions, results, and limitations |

### Acceptance criteria

- **R1** - Re-running ingestion produces no duplicate chunks; chunk metadata
  (source id, offsets) is persisted.
- **R2** - Top-k nearest neighbors with similarity scores; `EXPLAIN` shows the
  ANN index is used, not a sequential scan.
- **R3** - Returns top-k with scores; FTS column is GIN-indexed.
- **R4** - RRF fusion (`k = 60`); fusion is toggleable via config.
- **R5** - Answers are grounded in retrieved context; the prompt template is in
  the repo.
- **R6** - `/query` returns the answer and the list of source chunks used.
- **R7** - Metrics are computed from the `qrels`, not eyeballed; values are
  reproducible.
- **R8** - Faithfulness and answer relevancy reported on a sample, with the
  method documented.
- **R9** - The comparison table is filled with real numbers plus a short written
  interpretation.
- **R10** - `docker compose up` on a clean checkout reaches a working state,
  including seed data.
- **R11** - README is complete and the system is reproducible from it alone.

### Optional

Cross-encoder reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`), an exact-match
response cache in Redis, query rewriting, or answer streaming over SSE.

### Out of scope

Authentication, UI beyond Swagger/`curl`, model fine-tuning, graph RAG, agent
frameworks, multi-tenancy, horizontal scaling.

---

## Metrics

**Retrieval quality** (per query, averaged; computed against qrels):
- **Recall@k** - fraction of relevant documents found in the top-k.
- **Precision@k** - fraction of top-k that are relevant.
- **nDCG@k** - rank-weighted, using graded relevance from qrels.
- **MRR** - reciprocal rank of the first relevant document.
- **MAP** - mean average precision.

**Answer quality** (on a sample, LLM-judged):
- **Faithfulness** - is the answer grounded in retrieved context?
- **Answer relevancy** - does the answer address the query?

**Operational** (report, don't optimize): end-to-end latency, tokens used.

**Experiment report (R9)** - vary retrieval mode and one other axis (chunk size
or top-k), and fill:

| Config | Recall@10 | nDCG@10 | MRR | Faithfulness | Answer rel. |
| --- | --- | --- | --- | --- | --- |
| vector only | | | | | |
| lexical / BM25 only | | | | | |
| hybrid (RRF) | | | | | |
| hybrid + rerank *(if done)* | | | | | |

Follow the table with a short explanation of what moved the numbers and why.

---

### Everything free
No paid APIs or card-required tiers. The system must run from a clean checkout
with freely-downloadable weights.

| Need | Free option |
| --- | --- |
| LLM | Ollama (`llama3.1:8b` / `qwen2.5:7b`); Groq free tier as fallback |
| Embeddings | `sentence-transformers` (`BAAI/bge-small-en-v1.5`) or Ollama `nomic-embed-text` |
| Vector + lexical | PostgreSQL + pgvector |
| BM25 | `rank-bm25` |
| API | FastAPI |
| Evaluation | `beir` metrics and/or Ragas (OSS) |
| Optional rerank | `cross-encoder/ms-marco-MiniLM-L-6-v2` |

### Correctness notes
- PostgreSQL native FTS is **not** BM25. `ts_rank_cd` is cover-density ranking.
  A "BM25" result must come from an actual BM25 implementation (`rank-bm25`),
  not `ts_rank_cd` relabeled.
- Retrieval metrics must be computed against the `qrels`, not estimated by
  inspection.
