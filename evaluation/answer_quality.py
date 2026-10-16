"""
Answer Quality Evaluation

Samples 25 queries from the SciFact test set, runs the full RAG pipeline
(retrieve → generate), then uses Ollama as an LLM judge to score each answer
on faithfulness and relevance (1 - 5 scale)

Outputs average scores and provides concrete examples of good and bad generations

Run:
    python -m evaluation.answer_quality
"""

import asyncio
import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path

import httpx

from app.core.config import settings
from app.db.session import session_local
from app.embeddings.sentence_transformer import embedder
from app.generation.generator import generate_answer
from app.retrieval.retrieval import retrieve
from scripts.download_scifact import load_scifact_data
from typing import Literal

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

SAMPLE_SIZE = 25
RANDOM_SEED = 42
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# judge prompt — instructs the LLM to score on faithfulness + relevance
JUDGE_PROMPT = """
You are an expert evaluator of question-answering systems.

Given a QUESTION, a CONTEXT (the retrieved passages), and an ANSWER, score the answer on two dimensions:

1. Faithfulness (1-5): Is the answer fully supported by the context? 
   5 = entirely grounded, no hallucinations.
   1 = answer contradicts or ignores the context.

2. Relevance (1-5): Does the answer directly address the question?
   5 = directly and completely answers the question.
   1 = completely off-topic.

Respond ONLY with a JSON object in this exact format (no extra text):
{{"faithfulness": <int>, "relevance": <int>, "reason": "<one sentence>"}}

QUESTION: {question}

CONTEXT:
{context}

ANSWER:
{answer}
""".strip()


async def judge_answer(
    question: str,
    context: str,
    answer: str,
    client: httpx.AsyncClient,
) -> dict:
    """Call Ollama to judge a single answer. Returns scores dict."""

    prompt = (
        JUDGE_PROMPT.replace("{question}", question)
        .replace("{context}", context)
        .replace("{answer}", answer)
    )

    payload = {
        "model": settings.ollama_model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {"temperature": 0.0, "num_predict": 512},
    }

    try:
        resp = await client.post(f"{settings.ollama_url}/api/generate", json=payload)
        resp.raise_for_status()
        data = resp.json()
        raw = data.get("response", "").strip() or data.get("thinking", "").strip()

        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError(f"No JSON found in judge response: {raw!r}")
        scores = json.loads(raw[start:end])
        return {
            "faithfulness": int(scores.get("faithfulness", 0)),
            "relevance": int(scores.get("relevance", 0)),
            "reason": str(scores.get("reason", "")),
        }
    except Exception as exc:
        logger.warning("Judge call failed: %s", exc)
        return {"faithfulness": 0, "relevance": 0, "reason": f"error: {exc}"}


async def evaluate_answer_quality(
    sample_size: int = SAMPLE_SIZE,
    retrieval_mode: str = "hybrid",
    top_k: int = 5,
    seed: int = RANDOM_SEED,
    save_results: bool = True,
    rerank: bool = False,
    rerank_threshold: float | None = None,
    rrf_mode: Literal["standard", "weighted"] = "standard",
) -> dict:
    """
    Sample queries with verified ground truth, run RAG, and judge each answer.

    Returns:
        {
            "avg_faithfulness": float,
            "avg_relevance": float,
            "good_examples": [...],
            "bad_examples": [...],
            "sample_size": int,
            "details": [ ... ]
        }
    """

    logger.info("Loading SciFact test split ...")
    _, queries, qrels = load_scifact_data(split="test")

    # filter to queries that have ground-truth positive evidence in test set
    eligible_qids = sorted(
        [
            qid
            for qid, rels in qrels.items()
            if sum(rels.values()) > 0 and qid in queries
        ]
    )

    if not eligible_qids:
        eligible_qids = sorted(queries.keys())

    rng = random.Random(seed)
    sampled_ids = rng.sample(eligible_qids, k=min(sample_size, len(eligible_qids)))
    logger.info(
        "Sampled %d queries with verified ground truth (seed=%d)",
        len(sampled_ids),
        seed,
    )

    details: list[dict] = []

    async with httpx.AsyncClient(timeout=None) as client:  # TODO: Add timeout latter
        async with session_local() as session:
            for i, qid in enumerate(sampled_ids, 1):
                question = queries[qid]
                logger.info("[%d/%d] qid=%s", i, len(sampled_ids), qid)

                # retrieve + generate
                query_vector = await embedder.embed_query(question)
                chunks = await retrieve(
                    session=session,
                    query=question,
                    top_k=top_k,
                    mode=retrieval_mode,
                    rrf_k=60,
                    query_vector=query_vector,
                    rerank=rerank,
                    rerank_threshold=rerank_threshold,
                    rrf_mode=rrf_mode,
                )

                if not chunks:
                    logger.warning("No chunks retrieved for qid=%s", qid)
                    continue

                gen_result = await generate_answer(query=question, chunks=chunks)
                context = "\n\n".join(c.content for c in chunks)

                # judge the answer
                scores = await judge_answer(
                    question=question,
                    context=context,
                    answer=gen_result.answer,
                    client=client,
                )

                details.append(
                    {
                        "query_id": qid,
                        "query": question,
                        "answer": gen_result.answer,
                        **scores,
                    }
                )

    valid = [detail for detail in details if detail["faithfulness"] > 0]

    avg_faithfulness = (
        sum(detail["faithfulness"] for detail in valid) / len(valid) if valid else 0.0
    )

    avg_relevance = (
        sum(detail["relevance"] for detail in valid) / len(valid) if valid else 0.0
    )

    good_examples = [
        detail
        for detail in valid
        if detail["faithfulness"] >= 4 and detail["relevance"] >= 4
    ][:3]

    bad_examples = [
        detail
        for detail in valid
        if detail["faithfulness"] <= 2 or detail["relevance"] <= 2
    ][:3]

    summary = {
        "retrieval_mode": retrieval_mode,
        "rrf_mode": rrf_mode if retrieval_mode == "hybrid" else None,
        "top_k": top_k,
        "rerank": rerank,
        "rerank_threshold": rerank_threshold,
        "sample_size": len(details),
        "avg_faithfulness": round(avg_faithfulness, 3),
        "avg_relevance": round(avg_relevance, 3),
        "good_examples": good_examples,
        "bad_examples": bad_examples,
        "details": details,
    }

    logger.info(
        "Answer Quality | avg_faithfulness=%.3f avg_relevance=%.3f (n=%d)",
        avg_faithfulness,
        avg_relevance,
        len(details),
    )

    if save_results:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        summary["timestamp"] = datetime.now(timezone.utc).isoformat()

        suffix = ""
        if retrieval_mode == "hybrid" and rrf_mode == "weighted":
            suffix += "_weighted"
        if rerank:
            suffix += "_rerank" + (
                f"_th{rerank_threshold}" if rerank_threshold is not None else ""
            )
        answer_quality_json = (
            RESULTS_DIR / f"answer_quality_{retrieval_mode}_k{top_k}{suffix}.json"
        )
        answer_quality_json.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("Results saved to %s ", answer_quality_json.name)

    return summary


if __name__ == "__main__":
    asyncio.run(evaluate_answer_quality())
