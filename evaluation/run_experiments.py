"""
Retrieval Experiments

Saves each run's scores and markdown summary to evaluation/results/

Experiment matrix:
  - Modes:   vector, lexical, hybrid
  - top_k:   10          (all modes)
  - hybrid:  k=5, k=20   (additional hybrid variants)
  - rerank:  hybrid k=10 with/without relevance threshold

Run:
    python -m evaluation.run_experiments
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from app.embeddings.sentence_transformer import embedder
from evaluation.metrics import evaluate_retrieval
from scripts.download_scifact import load_scifact_data

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger("beir").setLevel(logging.ERROR)

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(__file__).resolve().parent / "results"


@dataclass
class ExperimentConfig:
    mode: Literal["vector", "lexical", "hybrid"]
    top_k: int
    rrf_k: int = 60
    rrf_mode: Literal["standard", "weighted"] = "standard"
    vector_weight: float = 0.7
    lexical_weight: float = 0.3
    rerank: bool = False
    rerank_threshold: float | None = None
    label: str = ""

    def __post_init__(self) -> None:
        if not self.label:
            parts = [f"{self.mode}_k{self.top_k}"]
            if self.mode == "hybrid" and self.rrf_mode == "weighted":
                parts.append("weighted")
            if self.rerank:
                parts.append("rerank")
            if self.rerank_threshold is not None:
                parts.append(f"th{self.rerank_threshold}")
            self.label = "_".join(parts)


# Experiment matrix — covers baseline variants, modular standard/weighted RRF, and reranking with/without threshold
EXPERIMENTS: list[ExperimentConfig] = [
    ExperimentConfig(mode="vector", top_k=10),
    ExperimentConfig(mode="lexical", top_k=10),
    ExperimentConfig(mode="hybrid", top_k=5, label="hybrid_k5"),
    ExperimentConfig(mode="hybrid", top_k=10, label="hybrid_k10"),
    ExperimentConfig(mode="hybrid", top_k=20, label="hybrid_k20"),
    ExperimentConfig(
        mode="hybrid", top_k=10, rrf_mode="weighted", label="hybrid_k10_weighted"
    ),
    ExperimentConfig(mode="hybrid", top_k=10, rerank=True, label="hybrid_k10_rerank"),
    ExperimentConfig(
        mode="hybrid",
        top_k=10,
        rrf_mode="weighted",
        rerank=True,
        label="hybrid_k10_weighted_rerank",
    ),
    ExperimentConfig(
        mode="hybrid",
        top_k=10,
        rerank=True,
        rerank_threshold=0.21,
        label="hybrid_k10_rerank_th0.21",
    ),
    ExperimentConfig(
        mode="hybrid",
        top_k=10,
        rrf_mode="weighted",
        rerank=True,
        rerank_threshold=0.21,
        label="hybrid_k10_weighted_rerank_th0.21",
    ),
]


async def run_all_experiments() -> dict[str, dict[str, float]]:
    """
    Run every experiment in the matrix
    """
    logger.info("Loading SciFact test split ...")
    _, queries, qrels = load_scifact_data(split="test")

    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]

    logger.info(
        "Pre-embedding all %d queries once for experiment matrix ...", len(query_texts)
    )
    query_vectors = await embedder.embed_documents(query_texts)

    all_results: dict[str, dict[str, float]] = {}

    for exp in EXPERIMENTS:
        logger.info(
            "--- Running experiment: %s (mode=%s, top_k=%d, rrf_k=%d, rrf_mode=%s, rerank=%s, threshold=%s) ---",
            exp.label,
            exp.mode,
            exp.top_k,
            exp.rrf_k,
            exp.rrf_mode,
            exp.rerank,
            exp.rerank_threshold,
        )
        scores = await evaluate_retrieval(
            queries=queries,
            qrels=qrels,
            mode=exp.mode,
            top_k=exp.top_k,
            rrf_k=exp.rrf_k,
            query_vectors=query_vectors,
            save_results=True,
            label=exp.label,
            rerank=exp.rerank,
            rerank_threshold=exp.rerank_threshold,
            rrf_mode=exp.rrf_mode,
            vector_weight=exp.vector_weight,
            lexical_weight=exp.lexical_weight,
        )
        all_results[exp.label] = scores

    return all_results


def _collect_metric_columns(all_results: dict[str, dict[str, float]]) -> list[str]:
    """Collect unique metric keys across all experiments in a clean, consistent order."""
    metric_order = ["ndcg", "map", "recall", "precision", "mrr"]
    cutoffs = [5, 10, 20]
    standard_columns = [f"{m}@{k}" for k in cutoffs for m in metric_order]

    present_keys = set()
    for s in all_results.values():
        present_keys.update(s.keys())

    ordered = [col for col in standard_columns if col in present_keys]
    ordered += [col for col in sorted(present_keys) if col not in standard_columns]
    return ordered


def format_summary_markdown(all_results: dict[str, dict[str, float]]) -> str:
    """Format experiment results into a clean markdown table."""
    if not all_results:
        return ""

    metrics = _collect_metric_columns(all_results)
    lines = [
        f"### Run Generated on: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "| Experiment | " + " | ".join(m.upper() for m in metrics) + " |",
        "| :--- | " + " | ".join(":---:" for _ in metrics) + " |",
    ]

    for label, scores in all_results.items():
        row_vals = [f"{scores[m]:.4f}" if m in scores else "-" for m in metrics]
        row = f"| **{label}** | " + " | ".join(row_vals) + " |"
        lines.append(row)

    lines.append("")
    return "\n".join(lines)


def print_summary_table(all_results: dict[str, dict[str, float]]) -> None:
    """Print a formatted console comparison table."""
    if not all_results:
        return

    metrics = _collect_metric_columns(all_results)
    col_w = 14
    label_w = max(max(len(l) for l in all_results.keys()) + 2, 18)

    header = f"{'Experiment':<{label_w}}" + "".join(f"{m:>{col_w}}" for m in metrics)
    sep = "=" * len(header)

    print(f"\n{sep}")
    print(header)
    print(sep)
    for label, scores in all_results.items():
        row_vals = [
            f"{scores[m]:>{col_w}.4f}" if m in scores else f"{'-':>{col_w}}"
            for m in metrics
        ]
        row = f"{label:<{label_w}}" + "".join(row_vals)
        print(row)
    print(f"{sep}\n")


async def main() -> None:
    all_results = await run_all_experiments()
    print_summary_table(all_results)
    md_content = format_summary_markdown(all_results)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    experiments_summary_json = RESULTS_DIR / "experiments_summary.json"
    merged_results: dict[str, dict[str, float]] = {}
    if experiments_summary_json.exists():
        try:
            existing_data = json.loads(
                experiments_summary_json.read_text(encoding="utf-8")
            )
            if isinstance(existing_data, dict) and "all_results" in existing_data:
                merged_results = existing_data["all_results"]
        except Exception as e:
            logger.warning("Could not read existing experiments_summary.json: %s", e)

    merged_results.update(all_results)

    data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "all_results": merged_results,
    }
    experiments_summary_json.write_text(json.dumps(data, indent=2), encoding="utf-8")

    experiments_summary_md = RESULTS_DIR / "experiments_summary.md"
    if experiments_summary_md.exists() and experiments_summary_md.stat().st_size > 0:
        existing_md = experiments_summary_md.read_text(encoding="utf-8").strip()
        updated_md = f"{existing_md}\n\n---\n\n{md_content}\n"
    else:
        title_header = "# Retrieval Evaluation Experiments Summary\n\n"
        updated_md = f"{title_header}{md_content}\n"

    experiments_summary_md.write_text(updated_md, encoding="utf-8")
    root_experiment_md = ROOT_DIR / "experiment.md"
    root_experiment_md.write_text(updated_md, encoding="utf-8")

    logger.info(
        "Combined summaries saved to %s (merged), %s (appended), and %s",
        experiments_summary_json.name,
        experiments_summary_md.name,
        root_experiment_md.name,
    )


if __name__ == "__main__":
    asyncio.run(main())
