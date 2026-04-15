from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from rapidfuzz.fuzz import token_set_ratio

from app.exceptions import AppError
from app.logging_utils import configure_logging
from app.rag import RAGService

configure_logging()
logger = logging.getLogger(__name__)


def evaluate(dataset_path: str, top_k: int) -> dict:
    rag = RAGService()

    try:
        data = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Failed to load dataset: {dataset_path}") from exc

    total = len(data)
    if total == 0:
        return {"total": 0, "average_similarity": 0.0}

    scores: list[float] = []
    for idx, row in enumerate(data):
        try:
            pred = rag.answer(question=row["query"], top_k=top_k)
            similarity = token_set_ratio(pred.answer, row["expected_answer"]) / 100.0
            scores.append(similarity)
        except (KeyError, AppError) as exc:
            logger.warning("Skipping row %s due to error: %s", idx, exc)

    if not scores:
        return {"total": total, "evaluated": 0, "average_similarity": 0.0, "scores": []}

    avg = sum(scores) / len(scores)
    return {
        "total": total,
        "evaluated": len(scores),
        "average_similarity": round(avg, 4),
        "scores": scores,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG evaluation script")
    parser.add_argument("--dataset", required=True, help="JSON file: [{'query','expected_answer'}]")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    result = evaluate(args.dataset, args.top_k)
    print(json.dumps(result, indent=2))
