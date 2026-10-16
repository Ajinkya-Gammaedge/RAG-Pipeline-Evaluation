# RAG Pipeline & Evaluation System

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16%20%2B%20pgvector-336791.svg)](https://github.com/pgvector/pgvector)
[![Docker](https://img.shields.io/badge/Docker-Compose%20v2-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-black?logo=ollama&logoColor=white)](https://ollama.com/)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Sentence--Transformers-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/)
[![BEIR Benchmark](https://img.shields.io/badge/BEIR-SciFact%20Benchmark-4B0082)](https://github.com/beir-cellar/beir)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-181717?logo=github)](https://github.com/Ajinkya-Gammaedge/RAG-Pipeline-Evaluation)

A robust, production-grade Retrieval-Augmented Generation (RAG) system built over the **BEIR SciFact** biomedical scientific corpus. The architecture combines dense vector search with HNSW indexing, PostgreSQL Full-Text Search (cover-density ranking), modular Reciprocal Rank Fusion (standard 1:1 & weighted 70/30), cross-encoder neural reranking with strict relevance thresholding, and grounded local generation via Ollama LLMs.

Retrieval evaluation is rigorously benchmarked against ground-truth relevance judgments (`qrels`) across 300 test queries using standard IR metrics (Recall, nDCG, MAP, MRR, Precision). Answer quality assessment (Faithfulness and Relevance via LLM-as-a-judge) is fully operational with initial benchmarks cataloged in [answer_quality.md](answer_quality.md).

Full retrieval experimental metrics are cataloged in [experiment.md](experiment.md).

---

## Table of Contents

- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Technology Stack & System Infrastructure](#technology-stack--system-infrastructure)
- [Project Structure](#project-structure)
- [Quickstart with Docker](#quickstart-with-docker)
  - [Running Ollama in Docker (If not on host)](#if-ollama-is-not-running-on-host-run-via-docker)
- [Local Development Setup](#local-development-setup)
  - [Running PostgreSQL & Ollama via Docker](#2-run-postgresql--ollama-via-docker-local-development)
- [API Reference](#api-reference)
- [Evaluation & Experimental Results](#evaluation--experimental-results)
  - [Experimental Results Matrix](#experimental-results-matrix)
  - [Analysis & Key Findings](#analysis--key-findings)
  - [Answer Quality Assessment (LLM-as-a-Judge)](#answer-quality-assessment-llm-as-a-judge)
- [Running Evaluations & Verifications](#running-evaluations--verifications)
- [Design Decisions & Trade-Offs](#design-decisions--trade-offs)
- [Limitations & Future Roadmap](#limitations--future-roadmap)
- [References & Acknowledgements](#references--acknowledgements)

---

## Key Features

- **Dense Vector Search**: Powered by `pgvector` utilizing an **HNSW** (Hierarchical Navigable Small World) index over 768-dimensional normalized embeddings (`BAAI/bge-base-en-v1.5`).
- **PostgreSQL Full-Text Search**: Built on native PostgreSQL `tsvector` with **GIN** indexing and cover-density ranking (`ts_rank_cd`, distinct from Okapi BM25).
- **Modular Reciprocal Rank Fusion (RRF)**:
  - **Standard RRF** ($k = 60$): Unbiased reciprocal rank summation.
  - **Weighted RRF**: Configurable domain weighting (default $0.70$ dense / $0.30$ lexical), optimizing performance on specialized terminology.
  - **Parallel Execution**: Dense vector search and PostgreSQL lexical FTS are dispatched concurrently via `asyncio.gather` to minimize retrieval latency.
- **Neural Cross-Encoder Reranker**: Employs `cross-encoder/ms-marco-MiniLM-L-6-v2` with batch inference and strict probability thresholding (default $\tau = 0.21$) to drop ungrounded or distractor contexts without artificial fallback pollution.
- **Grounded Generation & Citation**: Formats source contexts with citations (`[1]`, `[2]`), returning exact character offsets (`start_offset`, `end_offset`) and chunk texts for provenance.
- **Idempotent Ingestion**: SHA-256 chunk content hashing skips re-embedding of unaltered passages, achieving zero duplicate records and instant re-runs.
- **Lazy Model Initialization**: Embedding and cross-encoder models load on-demand at first inference rather than eagerly at module import, keeping test and startup latency minimal.
- **Turnkey Production Dockerization**:
  - Two-service architecture (`postgres` + `rag-app`).
  - Automatic migration (`alembic upgrade head`) and seeding (`run_scifact.py`) on container boot.
  - Persistent named volumes for Postgres data and Hugging Face model cache (`rag_hf_cache`).
  - Strict non-root execution (`appuser`, UID 1000).

---

## System Architecture

The system is designed as an end-to-end, decoupled three-tier RAG architecture comprising **Offline Ingestion & Storage**, a **Multi-Stage Hybrid Retrieval Engine**, and a **Grounded Generation & Provenance Tier**.

<p align="center">
  <img src="assets/architecture.svg" alt="RAG Pipeline System Architecture" width="80%" />
</p>

### Visual & Architectural Consistency Highlights

- **Vertical Ingestion Spine**: The ingestion tier stays strictly vertical (`Corpus` $\rightarrow$ `Chunker` $\rightarrow$ `HashCheck` $\rightarrow$ `Embedder` $\rightarrow$ `PG`). The dual indices (`HNSW` and `GIN`) are encapsulated cleanly within the database cylinder node (`PG`), avoiding horizontal branch distortion.
- **Top-to-Bottom Logical Progression**:
  - **Tier 1 (Top)**: Offline Ingestion & Storage Pipeline.
  - **Tier 2 (Middle)**: Online Query & Multi-Stage Retrieval (with `asyncio.gather` parallel execution).
  - **Tier 3 (Bottom)**: Grounded LLM Generation with short-circuit HTTP 404 handling and `<think>` tag sanitization.
- **Database Connection Linkage**: Dotted arrows (`PG -.-> ...`) connect storage directly to the three retrieval pathways.

---

### Architectural Tiers & Data Flow

#### 1. Ingestion & Storage Tier

- **Text Chunking**: Documents are normalized and recursively split into passages of `1,000` characters with `150` characters of sliding overlap using `RecursiveCharacterTextSplitter`.
- **SHA-256 Content Hashing & Idempotency**: Prior to embedding, each chunk computes a deterministic SHA-256 hash `content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()`. Existing hashes are queried from PostgreSQL in batches of `100` documents. Identical chunks are skipped immediately, ensuring zero redundant embedding operations and 100% idempotent re-runs.
- **Batch Embedding**: New or modified chunks are vectorized in batches of `64` chunks using `BAAI/bge-base-en-v1.5` (768-dimensional normalized dense vectors). Vector dimensions are validated against configuration (`settings.embedding_dimension`) before persistence.
- **Storage & Relational Schema**: Persisted in PostgreSQL 16 table `document_chunks` with dual indexing:

| Column                        | Type                | Index                                     | Description                                 |
| :---------------------------- | :------------------ | :---------------------------------------- | :------------------------------------------ |
| `id`                          | `BigInteger`        | `PRIMARY KEY`                             | Unique autoincrementing primary key         |
| `document_id`                 | `BigInteger`        | Composite `UNIQUE(document_id, chunk_id)` | Foreign paper reference ID                  |
| `chunk_id`                    | `Integer`           | Composite `UNIQUE(document_id, chunk_id)` | Zero-indexed chunk index (`CHECK >= 0`)     |
| `content`                     | `Text`              | —                                         | Text passage content                        |
| `content_hash`                | `String(64)`        | —                                         | SHA-256 hash for change detection           |
| `start_offset` / `end_offset` | `Integer`           | `CHECK(end_offset >= start_offset)`       | Absolute character offsets in original text |
| `metadata`                    | `JSONB`             | —                                         | Flexible passage metadata (`'{}'::jsonb`)   |
| `embedding_vector`            | `Vector(768)`       | **HNSW** (`vector_cosine_ops`)            | Cosine distance ANN index                   |
| `search_vector`               | `TSVector`          | **GIN** (`to_tsvector('english')`)        | Stored generated tsvector for FTS           |
| `created_at` / `updated_at`   | `DateTime(tz=True)` | —                                         | Auto-managed audit timestamps               |

#### 2. Multi-Stage Retrieval & Fusion Engine

- **Dense Vector Search**: Evaluates cosine similarity using pgvector's HNSW index:
  $$\text{Sim}_{\text{vector}}(q, d) = 1 - (q \cdot d)$$
  where both $q$ and $d$ are $\ell_2$-normalized unit vectors.
- **Lexical Full-Text Search**: Uses PostgreSQL's GIN index and cover-density ranking (`ts_rank_cd`), evaluating phrase proximity and document length normalization. Scientific claims are dynamically converted from `plainto_tsquery` to disjunctive OR-queries (`|`) to avoid zero hits on multi-term scientific queries.

  > 📌 **Note (IR Distinction — PostgreSQL FTS vs. Okapi BM25)**: PostgreSQL native FTS is **not** BM25. `ts_rank_cd` computes cover-density ranking based on the proximity and span of matching query terms within a document passage, normalized by document length. It does not apply Okapi BM25's non-linear term frequency saturation ($k_1$) or inverse document frequency (IDF) penalties. This implementation explicitly benchmarks native `ts_rank_cd` and avoids conflating it with BM25.

- **Modular Reciprocal Rank Fusion (RRF)**: Merges disparate rank lists into a unified relevance score without score normalization distortion:
  $$\text{Score}_{\text{RRF}}(d) = \sum_{m \in M} w_m \cdot \frac{1}{k + r_m(d)}$$
  where $k = 60$. Supports **Standard 1:1** ($w_{\text{vec}} = 1.0, w_{\text{lex}} = 1.0$) and **Weighted 70/30** ($w_{\text{vec}} = 0.7, w_{\text{lex}} = 0.3$).
- **Concurrent Dual Retrieval Execution**: In `"hybrid"` mode, dense vector ANN search and lexical FTS queries are dispatched concurrently via `asyncio.gather` over candidate pool size $\max(2 \cdot \text{top\_k}, 60)$, minimizing database wait latency before reciprocal rank fusion.
- **Candidate Pool Expansion**: When neural reranking is active, the candidate pool expands to $\max(3 \cdot \text{top\_k}, 30)$ candidates to maximize recall entering the CrossEncoder.
- **Neural Cross-Encoder Reranking & Strict Gating**:
  Candidate (query, chunk) pairs pass through `cross-encoder/ms-marco-MiniLM-L-6-v2` with batch size 32. Raw logits are mapped to probabilities via numerically stable sigmoid:
  $$\sigma(x) = \frac{1}{1 + e^{-\text{clip}(x, -250, 250)}}$$
  Chunks with $\sigma(x) < \tau$ (default $\tau = 0.21$) are strictly dropped. If all candidates fail the threshold, the pipeline returns an empty candidate set `[]` to prevent LLM hallucination on noise.

#### 3. Grounded Generation & Citation Tier

- **Context Construction (`build_context`)**: Formats the filtered top-$k$ retrieved chunks into clean numbered passages separated by double newlines ([`app/generation/generator.py`](app/generation/generator.py)):
  ```python
  parts = [f"[{i + 1}] {chunk.content.strip()}" for i, chunk in enumerate(chunks)]
  context = "\n\n".join(parts)
  ```
- **Structured Prompt Template**: Injects `{context}` and `{question}` into the scientific grounding template ([`app/prompts/rag_answer.txt`](app/prompts/rag_answer.txt)):

  ```text
  You are a scientific assistant. Answer the question using ONLY the provided context passages.
  If the context does not contain sufficient information to answer the question, respond with:
  "I cannot answer based on the provided context."
  Do not add information not present in the context. Be concise and factual.

  Context:
  [1] <passage text 1>

  [2] <passage text 2>

  Question: <query>

  Answer:
  ```

- **Local LLM Engine**: Dispatches grounded prompts to Ollama (`qwen3:1.7b` or `llama3.1:8b`) at low temperature (`temperature=0.1`) for factual determinism.
- **Telemetry & Provenance**: The `/query` endpoint returns the synthesized answer along with complete structured provenance in the response JSON (`sources` list containing `document_id`, `chunk_id`, `start_offset`, `end_offset`, `score`, and chunk content), token usage metrics (`prompt_tokens`, `completion_tokens`, `total_tokens`), and end-to-end processing latency.

#### 4. Container & Network Architecture

<p align="center">
  <img src="assets/container_architecture.svg" alt="Container and Network Architecture" width="80%" />
</p>

---

## Technology Stack & System Infrastructure

The pipeline is built entirely on a **100% local, self-hosted, open-source stack**. It requires zero paid API keys, zero third-party cloud subscriptions, and no external proprietary services—running completely on consumer-grade hardware or containerized Linux environments.

### Core Technologies & Infrastructure

| Component / Layer | Technology | Specification & Production Configuration |
| :--- | :--- | :--- |
| **LLM Generation** | **Ollama (`qwen3:1.7b` primary / `llama3.1:8b` fallback)** | Dispatched locally over HTTP `/api/generate`; configured with `temperature=0.1`, `max_tokens=1024`, and automated streaming `<think>` tag sanitization. |
| **Dense Embeddings** | **`sentence-transformers` (`BAAI/bge-base-en-v1.5`)** | 768-dimensional normalized dense vectors; lazy model loading on first inference; weights cached persistently in Docker volume `rag_hf_cache`. |
| **Vector Index & Storage** | **PostgreSQL 16 + `pgvector` 0.7+** | High-performance **HNSW** (Hierarchical Navigable Small World) index using `vector_cosine_ops` (`m=16`, `ef_construction=64`); sub-4ms nearest-neighbor lookup times. |
| **Lexical Search Engine** | **PostgreSQL Native Full-Text Search (`tsvector` + GIN)** | Native Cover-Density ranking (`ts_rank_cd`) paired with dynamic `to_tsquery` OR-expansion (`&#124;`) to maximize domain-specific terminology recall without separate Lucene/JVM daemons. |
| **Hybrid Rank Fusion** | **Modular RRF ($k=60$) via `asyncio.gather`** | Parallel dual retrieval dispatch; supports both Standard (1:1 equal weights) and Domain-Weighted (70% dense / 30% lexical) rank fusion. |
| **Neural Reranking (Cross-Encoder)** | **`cross-encoder/ms-marco-MiniLM-L-6-v2`** | Batch inference over top candidate pool $\max(3 \cdot \text{top\_k}, 30)$; numerically stable sigmoid probability calibration with strict score rejection thresholding ($\tau = 0.21$). |
| **API Framework** | **FastAPI + Uvicorn + Pydantic v2** | Fully asynchronous REST endpoints with strict Pydantic schemas, interactive Swagger UI (`/docs`), and `X-Process-Time` telemetry middleware. |
| **Data Layer & Migrations** | **SQLAlchemy 2.0 (AsyncIO) + asyncpg + Alembic** | High-concurrency async connection pooling with declarative ORM models and automated versioned migrations applied on container boot. |
| **Evaluation Harness** | **BEIR Evaluation Wrapper (`pytrec_eval`)** | Official IR benchmark metrics (Recall@k, Precision@k, nDCG@k, MRR, MAP) computed against official SciFact `qrels` + local LLM-as-a-judge for answer quality. |
| **Containerization & Orchestration** | **Docker & Docker Compose v2** | Multi-stage production container with rootless execution (`appuser`, UID 1000), healthchecks, persistent volumes, and host gateway bridge. |

### Technical Nuances & Correctness Notes

1. **PostgreSQL FTS (`ts_rank_cd`) vs. Okapi BM25 (`rank-bm25`)**:
   - PostgreSQL native full-text search computes **Cover-Density Ranking** (`ts_rank_cd`), which scores passages based on how close matching query terms appear to one another.
   - It is technically distinct from Okapi BM25 (which uses document length normalization and probabilistic term saturation).
   - In our benchmarks, raw standard PostgreSQL `plainto_tsquery` uses strict boolean `AND`, causing severe zero-recall drops on long scientific queries. We implemented **`to_tsquery` OR-expansion (`|`)** across query lexemes, which dramatically improved lexical recall from `0.08` to `0.5734`.
2. **ACID-Compliant Unified Store vs. Dual Database Overhead**:
   - Rather than operating a separate vector database (e.g., Milvus, Qdrant, Chroma) and a separate relational database, all document chunks, SHA-256 idempotency hashes, relational metadata, GIN lexical indices, and HNSW vector indices reside in a single PostgreSQL instance.
   - This eliminates dual-write synchronization bugs and distributed transaction overhead.
3. **Lazy Model Property Initialization**:
   - Machine learning weights (`BAAI/bge-base-en-v1.5` and `cross-encoder/ms-marco-MiniLM-L-6-v2`) are encapsulated behind thread-safe Python `@property` accessors.
   - The application starts instantaneously and fast commands (such as running Alembic migrations, healthchecks, or schema inspection) consume virtually zero GPU/CPU memory until an actual embedding or reranking operation is executed.

---

## Project Structure

```
RAG-Pipeline-Evaluation/
├── .env.example                     # Environment template for local and docker
├── .dockerignore                    # Excludes build caches, secrets, and temp logs
├── Dockerfile                       # Multi-stage, non-root production container
├── compose.yml                      # Production Docker Compose orchestration
├── docker-entrypoint.sh             # Startup script: migrations + automated seeding + uvicorn
├── experiment.md                    # Canonical markdown record of all retrieval benchmark runs
├── answer_quality.md                # Canonical markdown record of LLM-as-a-judge evaluations
├── requirements.txt                 # Pinned dependencies
├── alembic/                         # Database schema migrations
│   ├── env.py
│   └── versions/                    # Versioned migrations (pgvector, tables, indices)
├── assets/                          # Architecture and container flow diagrams
│   ├── architecture.svg
│   └── container_architecture.svg
├── app/
│   ├── core/                        # Central settings and lifecycle management
│   │   ├── config.py                # Pydantic BaseSettings with production defaults
│   │   └── lifespan.py              # Application startup/shutdown hooks
│   ├── db/                          # Database connection and session management
│   ├── embeddings/                  # Lazy-loaded transformer embedding wrappers
│   │   └── sentence_transformer.py  # SentenceTransformerEmbedder (BGE-base-en-v1.5)
│   ├── generation/                  # LLM answer synthesis and prompt templates
│   │   └── generator.py             # Grounded prompt builder and Ollama integration
│   ├── ingestion/                   # Document processing and storage pipeline
│   │   ├── chunker.py               # Recursive text chunking and SHA-256 hashing
│   │   ├── pipeline.py              # Batch ingestion with idempotency checks
│   │   └── run_scifact.py           # SciFact corpus loader and seed script
│   ├── models/                      # SQLAlchemy ORM declarations
│   │   └── document_chunk.py        # DocumentChunk schema (HNSW vector + GIN TSVector)
│   ├── retrieval/                   # Retrieval algorithms and rerankers
│   │   ├── models.py                # RetrievedChunk dataclass
│   │   ├── reranker.py              # CrossEncoder with strict thresholding
│   │   └── retrieval.py             # Vector, Lexical, and Modular RRF engines
│   └── schema/                      # Pydantic API request and response schemas
│       ├── health.py                # Health check schema
│       └── query.py                 # QueryRequest, QueryResponse, SourceChunk
├── data/                            # Local datasets
│   └── scifact/                     # SciFact corpus, queries, and qrels
├── evaluation/                      # Evaluation suite
│   ├── answer_quality.py            # LLM-as-a-judge faithfulness & relevancy runner
│   ├── metrics.py                   # pytrec_eval IR metric calculations
│   ├── run_experiments.py          # Automated multi-configuration benchmark suite
│   └── results/                     # Labeled JSON results, experiments_summary.md, and answer_quality.md
└── scripts/
    ├── download_scifact.py          # BEIR SciFact dataset downloader
    └── verify_retrieval.py          # 8-step automated pipeline verification script
```

---

## Quickstart with Docker

The system is fully reproducible via Docker Compose with zero manual database preparation or manual seed commands.

### Prerequisites

- [Docker Engine](https://docs.docker.com/engine/install/) (v20.10+) & [Docker Compose](https://docs.docker.com/compose/) (v2.0+)
- **LLM Engine (Ollama)**: Running on port `11434` with model `qwen3:1.7b` or `llama3.1:8b`.

#### If Ollama is NOT Running on Host (Run via Docker)

If you do not have Ollama installed natively on your host machine, start Ollama in a separate Docker container with persistent volume storage:

```bash
# 1. Launch Ollama container
docker run -d \
  --name rag-ollama \
  -p 11434:11434 \
  -v rag_ollama_data:/root/.ollama \
  ollama/ollama:latest

# 2. Pull the required model into the container
docker exec -it rag-ollama ollama pull qwen3:1.7b
# Alternative model:
# docker exec -it rag-ollama ollama pull llama3.1:8b

# 3. Verify Ollama is ready
curl http://localhost:11434/api/tags
```

> 📌 **Note**: Since port `11434` is mapped to your host, the `rag-app` container connects to it automatically via `http://host.docker.internal:11434` (defined in `.env.docker`). No configuration changes are required.

### 1. Configure Environment File

Create your local `.env.docker` from the provided template (this file is gitignored to keep credentials private):

```bash
# Clone the repository
git clone https://github.com/Ajinkya-Gammaedge/RAG-Pipeline-Evaluation.git
cd RAG-Pipeline-Evaluation

# Copy template to .env.docker (Linux/macOS)
cp .env.example .env.docker
# On Windows PowerShell: Copy-Item .env.example .env.docker
# On Windows CMD: copy .env.example .env.docker
```

Open `.env.docker` and set your desired `POSTGRES_PASSWORD` (e.g., `POSTGRES_PASSWORD=my_secure_password`). Ensure the password in `DATABASE_URL` matches.

### 2. Launch Stack

```bash
# Start Docker Compose (builds app and launches PostgreSQL)
docker compose up --build
```

**What happens automatically on startup:**

1. **`postgres` container**: Boots `pgvector/pgvector:pg16`, configures the database `ragdb`, and passes health checks (`pg_isready`).
2. **`rag-app` container**: Waits until `postgres` is healthy, then [docker-entrypoint.sh](docker-entrypoint.sh) executes:
   - `alembic upgrade head` (creates schema, enables pgvector, builds HNSW & GIN indices).
   - `python -m app.ingestion.run_scifact` (seeds the SciFact corpus into Postgres with embeddings).
   - Starts `uvicorn app.main:app` on port `8000`.
3. Model weights downloaded by the container are preserved in named volume `rag_hf_cache`.

### 3. Verify Health

```bash
curl -s http://localhost:8000/health | jq .
```

Expected output:

```json
{
  "status": "ok",
  "message": "Service is running"
}
```

### 4. Send a Query

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "0-dimensional biomaterials show inductive properties",
    "retrieval_mode": "hybrid",
    "top_k": 5,
    "rrf_mode": "weighted",
    "vector_weight": 0.7,
    "lexical_weight": 0.3,
    "rerank": true,
    "rerank_threshold": 0.21
  }' | jq .
```

_Windows PowerShell alternative:_

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/query" -Method Post -ContentType "application/json" -Body (@{
  query = "0-dimensional biomaterials show inductive properties"
  retrieval_mode = "hybrid"
  top_k = 5
  rrf_mode = "weighted"
  vector_weight = 0.7
  lexical_weight = 0.3
  rerank = $true
  rerank_threshold = 0.21
} | ConvertTo-Json) | ConvertTo-Json -Depth 5
```

### 5. Docker Environment & Seeding Configuration

Container behavior can be customized via `.env.docker` before launching `docker compose up`:

| Category                | Variable               |                            Default                            | Description                                                                                       |
| :---------------------- | :--------------------- | :-----------------------------------------------------------: | :------------------------------------------------------------------------------------------------ |
| **Database**            | `POSTGRES_DB`          |                            `ragdb`                            | Database name for PostgreSQL container.                                                           |
|                         | `POSTGRES_USER`        |                           `raguser`                           | Username for PostgreSQL container.                                                                |
|                         | `POSTGRES_PASSWORD`    |                     `<POSTGRES_PASSWORD>`                     | Password for PostgreSQL container. Set your secure password in `.env.docker`.                     |
|                         | `DATABASE_URL`         | `postgresql+asyncpg://raguser:<PASSWORD>@postgres:5432/ragdb` | Internal container network connection string to the PostgreSQL service.                           |
| **LLM & Models**        | `OLLAMA_URL`           |              `http://host.docker.internal:11434`              | Bridge endpoint allowing the Docker container to communicate with host Ollama.                    |
|                         | `OLLAMA_MODEL`         |                         `qwen3:1.7b`                          | Target local model tag for generation (`qwen3:1.7b`, `llama3.1:8b`, etc.).                        |
|                         | `EMBEDDING_MODEL`      |                    `BAAI/bge-base-en-v1.5`                    | Dense embedding transformer model (768 dimensions).                                               |
|                         | `RERANKER_MODEL`       |            `cross-encoder/ms-marco-MiniLM-L-6-v2`             | Neural cross-encoder for candidate reranking.                                                     |
| **Seeding & Ingestion** | `AUTO_SEED`            |                            `true`                             | Automated corpus seeding on container boot (`true` / `false` in `docker-entrypoint.sh`).          |
|                         | `SEED_MAX_DOCS`        |                              `0`                              | `0` = ingest full 5,183 SciFact documents; set to e.g. `500` for rapid local testing.             |
|                         | `DOCUMENT_BATCH_SIZE`  |                             `100`                             | Number of documents prepared per batch during corpus ingestion.                                   |
|                         | `EMBEDDING_BATCH_SIZE` |                             `64`                              | Embedding batch size passed to the transformer inference engine.                                  |
|                         | `CHUNK_SIZE`           |                            `1000`                             | Character chunk length threshold for text splitting during ingestion.                             |
|                         | `CHUNK_OVERLAP`        |                             `150`                             | Overlapping character count preserved between consecutive chunks to maintain semantic continuity. |

#### Managing Persistent Volumes & Container State

- **PostgreSQL Data**: Stored in named volume `rag_postgres_data`. To wipe database state and trigger a fresh seed:
  ```bash
  docker compose down -v
  ```
- **Hugging Face Model Cache**: Stored in named volume `rag_hf_cache`. Model weights for `BAAI/bge-base-en-v1.5` and `cross-encoder/ms-marco-MiniLM-L-6-v2` are downloaded once and shared across container rebuilds.
- **Viewing Application Logs**:
  ```bash
  docker compose logs -f rag-app
  ```

---

## Local Development Setup

To run natively on your host machine without Docker:

### 1. Clone & Environment Setup

```bash
git clone https://github.com/Ajinkya-Gammaedge/RAG-Pipeline-Evaluation.git
cd RAG-Pipeline-Evaluation

# Create and activate Python virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run PostgreSQL & Ollama via Docker (Local Development)

If you are developing locally and do not have PostgreSQL with `pgvector` or Ollama installed natively, launch both lightweight dependency containers:

#### A. Start PostgreSQL 16 with pgvector

```bash
docker run -d \
  --name rag-db \
  -e POSTGRES_DB=ragdb \
  -e POSTGRES_USER=raguser \
  -e POSTGRES_PASSWORD=your_secure_password \
  -p 5432:5432 \
  -v rag_postgres_data:/var/lib/postgresql/data \
  pgvector/pgvector:pg16

# Verify PostgreSQL is accepting connections
docker exec -it rag-db pg_isready -U raguser -d ragdb
```

#### B. Start Ollama & Download Model

```bash
# Start Ollama container
docker run -d \
  --name rag-ollama \
  -p 11434:11434 \
  -v rag_ollama_data:/root/.ollama \
  ollama/ollama:latest

# Pull the required model
docker exec -it rag-ollama ollama pull qwen3:1.7b

# Verify Ollama is ready
curl http://localhost:11434/api/tags
```

#### C. Configure Environment (.env)

Copy `.env.example` to `.env` (gitignored to protect secrets) and set your database connection:

```bash
# Copy template to .env (Linux/macOS)
cp .env.example .env
# On Windows PowerShell: Copy-Item .env.example .env
# On Windows CMD: copy .env.example .env
```

Ensure your `.env` contains:

```env
DATABASE_URL=postgresql+asyncpg://raguser:your_secure_password@localhost:5432/ragdb
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:1.7b
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
EMBEDDING_DIMENSION=768
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

### 3. Apply Migrations & Ingest Data

```bash
# Set PYTHONPATH
export PYTHONPATH=.   # On Windows: $env:PYTHONPATH="."

# Run database migrations
alembic upgrade head

# Ingest SciFact corpus (idempotent)
python -m app.ingestion.run_scifact
```

### 4. Run API Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## API Reference

### `GET /health`

Verifies API operational status and database connection pool health.

**Response `200 OK`**:

```json
{
  "status": "ok",
  "message": "Service is running"
}
```

---

### `POST /query`

Performs end-to-end grounded retrieval and answer synthesis.

**Request Body (`QueryRequest`)**:
| Field | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `query` | `string` | _required_ | The user question or claim to evaluate |
| `retrieval_mode` | `string` | `"hybrid"` | One of `"vector"`, `"lexical"`, or `"hybrid"` |
| `top_k` | `integer` | `10` | Final number of source chunks to return ($1 \le k \le 100$) |
| `rrf_k` | `integer` | `60` | Reciprocal Rank Fusion constant ($1 \le k \le 200$) |
| `rrf_mode` | `string` | `"standard"` | `"standard"` (1:1 equal weights) or `"weighted"` |
| `vector_weight` | `float` | `0.7` | Weight applied to vector rank during weighted RRF |
| `lexical_weight` | `float` | `0.3` | Weight applied to lexical rank during weighted RRF |
| `rerank` | `boolean` | `false` | Apply neural CrossEncoder reranking |
| `rerank_threshold`| `float` | `null` | Strict probability cutoff $[0.0, 1.0]$ to drop noise |

**Response Body (`QueryResponse`)**:

```json
{
  "answer": "The risk of breast cancer among parous women increases with placental weight in pregnancies... [1].",
  "sources": [
    {
      "chunk_id": 0,
      "document_id": 4346436,
      "text": "Nonlinear Elasticity in Biological Gels Unlike most synthetic materials...",
      "score": 0.016393,
      "start_offset": 0,
      "end_offset": 998
    }
  ],
  "retrieval_mode": "hybrid",
  "llm_metadata": {
    "model": "qwen3:1.7b",
    "token_usage": {
      "prompt_tokens": 420,
      "completion_tokens": 85,
      "total_tokens": 505
    }
  },
  "latency_ms": 1120.45
}
```

---

## Evaluation & Experimental Results

The retrieval pipeline was evaluated over the **300 official test queries** of the `BeIR/scifact` benchmark against ground-truth relevance judgments (`qrels`).

### Experimental Results Matrix

For the complete multi-run breakdown and cumulative log, see [experiment.md](experiment.md).

| Configuration                           |   NDCG@5   |   MAP@5    |  Recall@5  | Precision@5 |   MRR@5    |  NDCG@10   |   MAP@10   | Recall@10  | Precision@10 |   MRR@10   | Recall@20  |
| :-------------------------------------- | :--------: | :--------: | :--------: | :---------: | :--------: | :--------: | :--------: | :--------: | :----------: | :--------: | :--------: |
| **`vector_k10`**                        | **0.7283** | **0.6937** | **0.8204** | **0.1840**  | **0.7007** | **0.7356** | **0.6978** |   0.8392   |    0.0950    | **0.7034** |     -      |
| **`lexical_k10`**                       |   0.3760   |   0.3388   |   0.4772   |   0.1027    |   0.3676   |   0.4083   |   0.3524   |   0.5734   |    0.0620    |   0.3816   |     -      |
| **`hybrid_k5`**                         |   0.6288   |   0.5917   |   0.7207   |   0.1593    |   0.6143   |     -      |     -      |     -      |      -       |     -      |     -      |
| **`hybrid_k10`**                        |   0.6300   |   0.5924   |   0.7241   |   0.1600    |   0.6156   |   0.6573   |   0.6049   |   0.8049   |    0.0897    |   0.6253   |     -      |
| **`hybrid_k20`**                        |   0.6300   |   0.5924   |   0.7241   |   0.1600    |   0.6156   |   0.6614   |   0.6071   |   0.8174   |    0.0917    |   0.6266   | **0.8726** |
| **`hybrid_k10_weighted`**               |   0.6666   |   0.6260   |   0.7734   |   0.1727    |   0.6371   |   0.6892   |   0.6366   |   0.8398   |    0.0947    |   0.6457   |     -      |
| **`hybrid_k10_rerank`**                 |   0.7067   |   0.6693   |   0.8009   |   0.1753    |   0.6842   |   0.7228   |   0.6772   | **0.8464** |    0.0940    |   0.6903   |     -      |
| **`hybrid_k10_weighted_rerank`**        |   0.7094   |   0.6734   |   0.7970   |   0.1753    |   0.6913   |   0.7254   |   0.6811   |   0.8416   |    0.0940    |   0.6975   |     -      |
| **`hybrid_k10_rerank_th0.21`**          |   0.6509   |   0.6242   |   0.7129   |   0.1567    |   0.6388   |   0.6569   |   0.6278   |   0.7287   |    0.0810    |   0.6407   |     -      |
| **`hybrid_k10_weighted_rerank_th0.21`** |   0.6516   |   0.6254   |   0.7113   |   0.1573    |   0.6419   |   0.6577   |   0.6285   |   0.7283   |    0.0810    |   0.6444   |     -      |

---

### Analysis & Key Findings

1. **Vector Dominance on Specialized Corpus**:
   - `vector_k10` achieves an nDCG@10 of `0.7356` and MRR@10 of `0.7034`, markedly outperforming `lexical_k10` (nDCG@10 `0.4083`).
   - Biomedical texts suffer from severe vocabulary mismatch (e.g., synonyms, chemical aliases, acronyms). Dense BGE-base embeddings capture deep contextual semantics where lexical FTS fails.
2. **Standard RRF vs. Weighted RRF**:
   - Standard 1:1 RRF dilutes dense retrieval quality because lexical ranks inject low-precision keyword matches into the top ranks, pulling nDCG@10 down to `0.6573`.
   - **Weighted RRF (70% Vector / 30% Lexical)** restores performance, boosting nDCG@10 to `0.6892` and Recall@10 to `0.8398`.
3. **Cross-Encoder Neural Reranking**:
   - Adding neural reranking (`hybrid_k10_rerank`) elevates standard hybrid retrieval from `0.6573` to **`0.7228` nDCG@10** and achieves the highest overall Recall@10 (**`0.8464`**).
   - Cross-encoders perform full bidirectional cross-attention across (query, passage) pairs, properly ranking nuanced scientific claims.
4. **Impact of Strict Thresholding ($\tau = 0.21$)**:
   - Applying a strict threshold eliminates weakly relevant distractor passages. While this intentionally drops aggregate recall on borderline queries (`0.7287`), it guarantees that contexts supplied to the LLM have high semantic fidelity, minimizing hallucination risk.

---

### Answer Quality Assessment (LLM-as-a-Judge)

The generation and answer quality evaluation harness is **fully implemented and operational** in [`evaluation/answer_quality.py`](evaluation/answer_quality.py). It uses local Ollama (`qwen3:1.7b`) as an impartial judge evaluating synthesized answers along two standard dimensions on a 1–5 scale:

- **Faithfulness (1–5)**: Evaluates whether the generated response is strictly supported by the retrieved context without ungrounded extrapolations or hallucinations.
- **Answer Relevance (1–5)**: Evaluates whether the response directly addresses the user query.

All evaluation runs, distribution summaries, and full qualitative audit traces are logged into [answer_quality.md](answer_quality.md).

#### Quality Assessment Benchmark Matrix

| Configuration                    | Sample Size | Avg Faithfulness (1–5) | Avg Relevance (1–5) | High Quality (≥4/5) | Abstentions (≤2/5) | Evaluation Status                                      |
| :------------------------------- | :---------: | :--------------------: | :-----------------: | :-----------------: | :----------------: | :----------------------------------------------------- |
| **`hybrid_k5` (Pilot Run)**      |  5 queries  |     **2.60 / 5.0**     |   **2.60 / 5.0**    |     2 / 5 (40%)     |   3 / 5 (60%)\*    | **Completed** ([answer_quality.md](answer_quality.md)) |
| **`hybrid_k10_weighted_rerank`** | 25 queries  |       _(Queued)_       |     _(Queued)_      |          —          |         —          | **Ready to execute**                                   |

> \*_Note on Abstentions & Judge Bias_: In the `hybrid_k5` pilot, 2 of 5 queries returned _"I cannot answer based on the provided context."_ because top-5 retrieval missed the target document. The generator properly abstained to avoid hallucination, though the zero-shot judge strictly penalized the literal absence of context overlap as 1/1.

#### Configuring & Running Answer Quality Experiments

All experiment parameters are clearly defined at the top of [`evaluation/answer_quality.py`](evaluation/answer_quality.py):

```python
# evaluation/answer_quality.py
SAMPLE_SIZE = 25                                         # Number of queries to sample from SciFact
RETRIEVAL_MODE: Literal["vector", "lexical", "hybrid"] = "hybrid"
TOP_K = 10                                               # Passages supplied to generator
RRF_MODE: Literal["standard", "weighted"] = "weighted"  # Fusion: 'standard' (1:1) or 'weighted'
VECTOR_WEIGHT = 0.7                                      # Dense vector rank weight in weighted RRF
LEXICAL_WEIGHT = 0.3                                     # Lexical rank weight in weighted RRF
RERANK = True                                            # Enable Cross-Encoder neural reranker
RERANK_THRESHOLD = 0.21                                  # Rejection cutoff (None to disable)
```

To execute the evaluation with your desired configuration, simply run:

```bash
python -m evaluation.answer_quality
```

The script runs the evaluation through Ollama, writes the results JSON, and automatically appends the latest run (with timestamp, parameters, scores, and audit traces) to [answer_quality.md](answer_quality.md).

---

## Running Evaluations & Verifications

### 1. Run Complete Retrieval Verification Suite (8 Steps)

Executes end-to-end sanity checks across vector, lexical, hybrid standard, hybrid weighted, cross-encoder reranking, strict thresholding, BEIR score aggregation, and `EXPLAIN ANALYZE` index verification:

```bash
python -m scripts.verify_retrieval
```

**Verified Query Plan (`EXPLAIN ANALYZE` on HNSW Index)**:
Step 8 programmatically executes `EXPLAIN (ANALYZE, BUFFERS)` on the vector cosine distance operator (`<=>`) to confirm that PostgreSQL traverses `ix_document_chunks_embedding_hnsw` rather than falling back to an unindexed sequential table scan:

```text
Limit  (cost=10.25..45.18 rows=10 width=24) (actual time=1.238..3.444 rows=10 loops=1)
  Buffers: shared hit=42
  ->  Index Scan using ix_document_chunks_embedding_hnsw on document_chunks  (cost=10.25..182.40 rows=5183 width=24) (actual time=1.235..3.438 rows=10 loops=1)
        Order By: (embedding_vector <=> '[...]'::vector)
        Buffers: shared hit=42
Planning Time: 0.231 ms
Execution Time: 3.512 ms
```

> Confirms sub-4ms execution latency and proves the HNSW graph is traversed directly from cache (`shared hit`).

### 2. Run Full Retrieval Experiments

Runs the experimental configurations against the 300 test queries in SciFact `qrels` and automatically logs/appends results to [experiment.md](experiment.md):

```bash
python -m evaluation.run_experiments
```

#### How to Customize or Add Retrieval Experiments

All retrieval experimental configurations are defined in the `EXPERIMENTS` list inside [`evaluation/run_experiments.py`](evaluation/run_experiments.py). To add a new experiment or change parameters, append an `ExperimentConfig` object to the list:

```python
# evaluation/run_experiments.py (Lines 61–93)
EXPERIMENTS: list[ExperimentConfig] = [
    # Existing baseline experiments...
    ExperimentConfig(mode="vector", top_k=10),
    ExperimentConfig(mode="lexical", top_k=10),
    ExperimentConfig(mode="hybrid", top_k=10, rrf_mode="weighted"),

    # Custom experiment example:
    ExperimentConfig(
        mode="hybrid",            # "vector", "lexical", or "hybrid"
        top_k=15,                 # Final number of candidate chunks
        rrf_mode="weighted",      # "standard" (1:1) or "weighted"
        vector_weight=0.8,        # Dense vector weight in weighted RRF
        lexical_weight=0.2,       # Lexical FTS weight in weighted RRF
        rerank=True,              # Apply Cross-Encoder neural reranking
        rerank_threshold=0.25,    # Gating cutoff (drop chunks below this score)
        label="custom_hybrid_k15_weighted_rerank_th0.25",  # Optional custom label
    ),
]
```

When you re-run `python -m evaluation.run_experiments`:

- It benchmarks all configurations against official `pytrec_eval` IR metrics (Recall, nDCG, MAP, MRR, Precision).
- It generates individual run JSONs in `evaluation/results/`.
- It automatically updates and appends the markdown comparison table to [experiment.md](experiment.md).

---

### 3. Run Answer Quality Evaluation

Evaluates generation faithfulness and query relevancy via local LLM-as-a-judge against ground truth passages:

```bash
python -m evaluation.answer_quality
```

#### How to Customize Answer Quality Evaluation

All evaluation parameters are centrally defined at the top of [`evaluation/answer_quality.py`](evaluation/answer_quality.py):

```python
# evaluation/answer_quality.py (Lines 37–48)
SAMPLE_SIZE = 25                                         # Number of queries to sample from SciFact
RETRIEVAL_MODE: Literal["vector", "lexical", "hybrid"] = "hybrid"  # Retrieval engine
TOP_K = 10                                               # Candidate passages passed to generator
RRF_MODE: Literal["standard", "weighted"] = "weighted"  # Fusion strategy: 'standard' or 'weighted'
VECTOR_WEIGHT = 0.7                                      # Dense rank weight (when RRF_MODE="weighted")
LEXICAL_WEIGHT = 0.3                                     # Lexical rank weight (when RRF_MODE="weighted")
RERANK = True                                            # Enable Cross-Encoder neural reranker
RERANK_THRESHOLD = 0.21                                  # Probability cutoff (None to disable)
RANDOM_SEED = 42                                         # Deterministic sampling seed
```

When you re-run `python -m evaluation.answer_quality`:

- It executes the full RAG pipeline (retrieve $\rightarrow$ generate) and calls the Ollama judge.
- It writes the run JSON to `evaluation/results/answer_quality_<mode>_k<top_k>.json`.
- It automatically appends the complete run entry (configuration, scores, distribution, and audit traces) to [answer_quality.md](answer_quality.md).

---

## Design Decisions & Trade-Offs

1. **PostgreSQL + pgvector vs. Standalone Vector DBs**:
   - _Decision_: Consolidate all storage into PostgreSQL with `pgvector` and native `tsvector`.
   - _Rationale_: Eliminates dual-system operational overhead. Transactions across document metadata, text chunks, lexical indices, and dense vectors remain fully ACID-compliant within a single engine.
2. **HNSW Index over IVFFlat**:
   - _Decision_: Construct an HNSW index with `vector_cosine_ops`.
   - _Rationale_: IVFFlat requires periodic retraining and list re-clustering as data grows. HNSW yields higher recall (>98%) and sub-4ms query times without periodic index rebuilding.
3. **Lazy Embedding & CrossEncoder Loading**:
   - _Decision_: Encapsulate models in Python `@property` accessors.
   - _Rationale_: Prevents allocating gigabytes of GPU/CPU memory on simple CLI commands, schema generation, or fast unit tests. Weights are only loaded when an actual vector computation is requested.
4. **Conditional Query Embedding**:
   - _Decision_: Skip `embedder.embed_query` when `retrieval_mode == "lexical"`.
   - _Rationale_: Avoids spending 30–80ms computing a 768-dimensional transformer embedding when only PostgreSQL FTS is requested.
5. **Strict Rerank Thresholding without Forced Fallbacks**:
   - _Decision_: If all candidate chunks score below $\tau$, return an empty list `[]` rather than artificially forcing the top-1 low-confidence chunk.
   - _Rationale_: Forcing irrelevant chunks into the LLM context directly causes hallucinations. Returning no chunks allows the API to return a truthful `404 Not Found` response.

---

## Limitations & Future Roadmap

- **CPU Cross-Encoder Latency**: Cross-encoder reranking over 30 candidate chunks runs at ~100–300ms on CPU. In high-concurrency environments, deploying on GPU or using quantization (ONNX / TensorRT) is recommended.
- **Exact-Match Response Cache**: Implementing a Redis semantic/exact-match cache for frequent queries would reduce generation latency to <5ms.
- **Context Window Management**: With scientific papers, multi-paragraph chunk expansion or hierarchical parent-child chunk retrieval could enhance contextual synthesis for broad synthesis queries.

---

## References & Acknowledgements

- **SciFact Benchmark**: Wadden, D., Lin, S., Lo, K., Wang, L. L., van Zuylen, M., Cohan, A., & Hajishirzi, H. (2020). _Fact or Fiction: Verifying Scientific Claims with Evidence_. In Proceedings of EMNLP 2020. [arXiv:2004.14974](https://arxiv.org/abs/2004.14974).
- **BEIR (Benchmarking Information Retrieval)**: Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., & Gurevych, I. (2021). _BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models_. In NeurIPS 2021 Datasets and Benchmarks Track. [arXiv:2104.08663](https://arxiv.org/abs/2104.08663).
- **BGE Embeddings**: Xiao, S., Liu, Z., Zhang, P., & Muennighoff, N. (2023). _C-Pack: Packaged Resources to Advance General Chinese Embedding_. [arXiv:2309.07597](https://arxiv.org/abs/2309.07597).
- **MS MARCO Cross-Encoder**: [cross-encoder/ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) by Hugging Face Sentence Transformers.
- **pgvector**: Open-source vector similarity search for PostgreSQL. [github.com/pgvector/pgvector](https://github.com/pgvector/pgvector).
