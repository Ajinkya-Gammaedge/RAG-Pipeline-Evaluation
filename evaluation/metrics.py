"""
Retrieval Evaluation

Evaluates retrieval modes (vector, lexical, hybrid, rerank) against SciFact
test qrels using BEIR's EvaluateRetrieval wrapper (pytrec_eval under the hood).

Metrics reported dynamically up to top_k:
  - top_k=5:   @5 (nDCG@5, MAP@5, Recall@5, P@5, MRR@5)
  - top_k=10:  @5, @10
  - top_k=20:  @5, @10, @20

Run:
    python -m evaluation.metrics
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from beir.retrieval.evaluation import EvaluateRetrieval

from app.db import session_local
from app.embeddings.sentence_transformer import embedder
from app.retrieval.retrieval import retrieve, aggregate_document_scores
from scripts.download_scifact import load_scifact_data

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger("beir").setLevel(logging.ERROR)

STANDARD_CUTOFFS = [5, 10, 20]
RESULTS_DIR = Path(__file__).resolve().parent / "results"


async def run_retrieval_for_all_queries(
    queries: dict[str, str],
    mode: Literal["vector", "lexical", "hybrid"],
    top_k: int,
    rrf_k: int = 60,
    query_vectors: list[list[float]] | None = None,
    rerank: bool = False,
    rerank_threshold: float | None = None,
    rrf_mode: Literal["standard", "weighted"] = "standard",
    vector_weight: float = 0.7,
    lexical_weight: float = 0.3,
) -> dict[str, dict[str, float]]:
    """
    Run retrieval for every query and return results in BEIR format:
        { query_id: { corpus_doc_id: score, ... }, ... }

    Retrieves top_k chunks per query using the retrieve() pipeline,
    then aggregates chunk scores to document level for BEIR evaluation.
    """

    results: dict[str, dict[str, float]] = {}

    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]

    if query_vectors is None:
        if mode in ("vector", "hybrid"):
            logger.info("Embedding %d queries for mode=%s ...", len(query_texts), mode)
            query_vectors = await embedder.embed_documents(query_texts)
        else:
            query_vectors = [None] * len(query_texts)
    else:
        if mode == "lexical":
            query_vectors = [None] * len(query_texts)

    logger.info(
        "Running retrieval (mode=%s, top_k=%d, rrf_mode=%s) ...", mode, top_k, rrf_mode
    )
    async with session_local() as session:
        for qid, text, qvec in zip(query_ids, query_texts, query_vectors):
            chunks = await retrieve(
                session=session,
                query=text,
                top_k=top_k,
                mode=mode,
                rrf_k=rrf_k,
                query_vector=qvec,
                rerank=rerank,
                rerank_threshold=rerank_threshold,
                rrf_mode=rrf_mode,
                vector_weight=vector_weight,
                lexical_weight=lexical_weight,
            )
            # aggregate chunk scores to document level and retain top_k unique documents
            doc_scores = aggregate_document_scores(chunks)
            sorted_docs = dict(
                sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[
                    :top_k
                ]
            )
            results[qid] = sorted_docs

    return results


async def evaluate_retrieval(
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    mode: Literal["vector", "lexical", "hybrid"] = "hybrid",
    top_k: int = 10,
    rrf_k: int = 60,
    query_vectors: list[list[float]] | None = None,
    save_results: bool = True,
    label: str | None = None,
    eval_k_values: list[int] | None = None,
    rerank: bool = False,
    rerank_threshold: float | None = None,
    rrf_mode: Literal["standard", "weighted"] = "standard",
    vector_weight: float = 0.7,
    lexical_weight: float = 0.3,
) -> dict[str, float]:
    """
    Dynamically evaluates cutoffs up to and including top_k:
      - top_k=5  -> evaluates @5
      - top_k=10 -> evaluates @5, @10
      - top_k=20 -> evaluates @5, @10, @20
    """

    t0 = time.perf_counter()
    results = await run_retrieval_for_all_queries(
        queries=queries,
        mode=mode,
        top_k=top_k,
        rrf_k=rrf_k,
        query_vectors=query_vectors,
        rerank=rerank,
        rerank_threshold=rerank_threshold,
        rrf_mode=rrf_mode,
        vector_weight=vector_weight,
        lexical_weight=lexical_weight,
    )
    elapsed = time.perf_counter() - t0
    logger.info("Retrieval done in %.1f s", elapsed)

    # determine evaluation cutoffs: standard tiers <= top_k
    if eval_k_values is not None:
        cutoffs = sorted(eval_k_values)
    else:
        cutoffs = sorted(
            list(set([k for k in STANDARD_CUTOFFS if k <= top_k] or [top_k]))
        )

    evaluator = EvaluateRetrieval()

    ndcg, map_, recall, precision = evaluator.evaluate(
        qrels,
        results,
        k_values=cutoffs,
    )
    mrr = evaluator.evaluate_custom(qrels, results, k_values=cutoffs, metric="mrr")

    scores: dict[str, float] = {}

    for k in cutoffs:
        scores[f"ndcg@{k}"] = round(float(ndcg[f"NDCG@{k}"]), 5)
        scores[f"map@{k}"] = round(float(map_[f"MAP@{k}"]), 5)
        scores[f"recall@{k}"] = round(float(recall[f"Recall@{k}"]), 5)
        scores[f"precision@{k}"] = round(float(precision[f"P@{k}"]), 5)
        scores[f"mrr@{k}"] = round(float(mrr[f"MRR@{k}"]), 5)

    exp_label = label or f"{mode}_k{top_k}"
    logger.info("Results (%s, evaluated at %s):", exp_label, cutoffs)
    for metric, value in scores.items():
        logger.info("  %-20s %.4f", metric, value)

    if save_results:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        iso_timestamp = datetime.now(timezone.utc).isoformat()
        payload = {
            "label": exp_label,
            "mode": mode,
            "top_k": top_k,
            "rrf_k": rrf_k,
            "rrf_mode": rrf_mode,
            "vector_weight": vector_weight if rrf_mode == "weighted" else 1.0,
            "lexical_weight": lexical_weight if rrf_mode == "weighted" else 1.0,
            "rerank": rerank,
            "rerank_threshold": rerank_threshold,
            "num_queries": len(queries),
            "timestamp": iso_timestamp,
            "scores": scores,
        }

        labeled_file = RESULTS_DIR / f"retrieval_{exp_label}.json"
        labeled_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        logger.info("Results saved to %s", labeled_file.name)

    return scores


async def main() -> None:
    """Evaluate all three retrieval modes and print a summary table"""

    logger.info("Loading SciFact test split ...")
    _, queries, qrels = load_scifact_data(split="test")

    modes: list[Literal["vector", "lexical", "hybrid"]] = [
        "vector",
        "lexical",
        "hybrid",
    ]

    all_scores: dict[str, dict[str, float]] = {}

    for mode in modes:
        all_scores[mode] = await evaluate_retrieval(
            queries=queries,
            qrels=qrels,
            mode=mode,
            top_k=10,
            rrf_k=60,
        )

    # Print summary table
    metrics = list(next(iter(all_scores.values())).keys())
    header = f"{'Mode':<12}" + "".join(f"{m:>14}" for m in metrics)
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for mode, scores in all_scores.items():
        row = f"{mode:<12}" + "".join(
            f"{scores[m]:>14.4f}" if m in scores else f"{'-':>14}" for m in metrics
        )
        print(row)
    print("=" * len(header) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
