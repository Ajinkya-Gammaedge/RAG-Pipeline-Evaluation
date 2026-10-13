from pathlib import Path
import logging
from typing import Any
from beir import util
from beir.datasets.data_loader import GenericDataLoader

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
SCIFACT_DIR = DATA_DIR / "scifact"
SCIFACT_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
)


def download_scifact(data_dir: Path | str = DATA_DIR) -> Path:
    """Download and extract SciFact dataset if not already present."""
    target_dir = Path(data_dir) / "scifact"
    corpus_file = target_dir / "corpus.jsonl"

    if not corpus_file.exists():
        logger.info("Downloading SciFact to %s...", data_dir)
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        util.download_and_unzip(SCIFACT_URL, str(data_dir))
    return target_dir


def load_scifact_data(
    data_path: Path | str | None = None,
    split: str = "test",
) -> tuple[dict[str, dict[str, Any]], dict[str, str], dict[str, dict[str, int]]]:
    """Load SciFact corpus, queries, and qrels."""
    if data_path is None:
        target_path = download_scifact(DATA_DIR)
    else:
        target_path = Path(data_path)
        if not (target_path / "corpus.jsonl").exists():
            target_path = download_scifact(target_path.parent)

    corpus, queries, qrels = GenericDataLoader(data_folder=str(target_path)).load(
        split=split
    )
    return corpus, queries, qrels


def load_scifact_corpus(
    data_path: Path | str | None = None,
    split: str = "test",
) -> dict[str, dict[str, Any]]:
    """Load and return only the SciFact corpus."""
    corpus, _, _ = load_scifact_data(data_path=data_path, split=split)
    return corpus


if __name__ == "__main__":
    corpus, queries, qrels = load_scifact_data()
    logger.info(
        f"SciFact loaded successfully: {len(corpus)} documents, {len(queries)} queries, {len(qrels)} qrels"
    )
