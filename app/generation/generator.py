import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import httpx

from app.core.config import settings
from app.retrieval.models import RetrievedChunk

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "rag_answer.txt"

PROMPT_TEMPLATE: str = PROMPT_PATH.read_text(encoding="utf-8")


class GenerationError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class GenerationResult:
    answer: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


def build_context(chunks: Sequence[RetrievedChunk]) -> str:
    if not chunks:
        return ""
    parts = [f"[{i + 1}] {chunk.content.strip()}" for i, chunk in enumerate(chunks)]
    return "\n\n".join(parts)


def clean_response(raw_text: str) -> str:
    if "<think>" in raw_text and "</think>" in raw_text:
        parts = raw_text.split("</think>", 1)
        return parts[1].strip()
    return raw_text.strip()


async def generate_answer(
    query: str,
    chunks: Sequence[RetrievedChunk],
    temperature: float = 0.1,
    max_tokens: int = 1024,
) -> GenerationResult:
    context = build_context(chunks)

    prompt = PROMPT_TEMPLATE.replace("{context}", context).replace(
        "{question}", query.strip()
    )

    payload = {
        "model": settings.ollama_model,
        "prompt": prompt,
        "stream": False,
        "think": True,
        "keep_alive": "20m",
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    try:
        async with httpx.AsyncClient(
            timeout=None  # TODO: set timeout latter
        ) as client:
            response = await client.post(
                f"{settings.ollama_url}/api/generate",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
        logger.error("Failed to connect to Ollama at %s: %s", settings.ollama_url, exc)
        raise GenerationError(
            f"Ollama service unavailable at {settings.ollama_url}"
        ) from exc
    # except httpx.TimeoutException as exc:
    #     logger.error("Ollama request timed out after 120s: %s", exc)
    #     raise GenerationError("Ollama generation timed out") from exc
    except Exception as exc:
        logger.error("Unexpected error during Ollama generation: %s", exc)
        raise GenerationError(f"Generation failed: {exc}") from exc

    raw_answer: str = data.get("response", "")
    if not raw_answer and data.get("thinking"):
        raw_answer = data.get("thinking", "")

    answer = clean_response(raw_answer)
    if not answer:
        raise GenerationError("Failed to generate answer")

    model_used: str = data.get("model", settings.ollama_model)
    prompt_tokens: int = int(data.get("prompt_eval_count") or 0)
    completion_tokens: int = int(data.get("eval_count") or 0)
    total_tokens = prompt_tokens + completion_tokens

    logger.info(
        "Generated answer via %s | prompt_tokens=%d completion_tokens=%d total_tokens=%d",
        model_used,
        prompt_tokens,
        completion_tokens,
        total_tokens,
    )

    return GenerationResult(
        answer=answer,
        model=model_used,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )
