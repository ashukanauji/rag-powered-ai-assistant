from __future__ import annotations

import argparse
import json
from pathlib import Path

from rapidfuzz.fuzz import token_set_ratio

from app.rag import RAGService


def evaluate(dataset_path: str, top_k: int) -> dict:
    rag = RAGService()
    data = json.loads(Path(dataset_path).read_text(encoding="utf-8"))

    total = len(data)
    if total == 0:
        return {"total": 0, "average_similarity": 0.0}

    scores: list[float] = []
    for row in data:
        pred = rag.answer(question=row["query"], top_k=top_k)
        similarity = token_set_ratio(pred.answer, row["expected_answer"]) / 100.0
        scores.append(similarity)

    avg = sum(scores) / len(scores)
    return {"total": total, "average_similarity": round(avg, 4), "scores": scores}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG evaluation script")
    parser.add_argument("--dataset", required=True, help="JSON file: [{'query','expected_answer'}]")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    result = evaluate(args.dataset, args.top_k)
    print(json.dumps(result, indent=2))
