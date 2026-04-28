from __future__ import annotations

import json
import math
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


_ARTICLES = {"a", "an", "the"}
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _normalize(text: str) -> str:
    text = text.lower().translate(_PUNCT_TABLE)
    tokens = [tok for tok in text.split() if tok not in _ARTICLES]
    return " ".join(tokens)


def _tokenize(text: str) -> list[str]:
    normalized = _normalize(text)
    if not normalized:
        return []
    return normalized.split()


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = _tokenize(prediction)
    gold_tokens = _tokenize(ground_truth)

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def bleu1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = _tokenize(prediction)
    gold_tokens = _tokenize(ground_truth)

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens:
        return 0.0

    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)
    overlap = sum((pred_counts & gold_counts).values())
    precision = overlap / len(pred_tokens)

    pred_len = len(pred_tokens)
    gold_len = len(gold_tokens)
    brevity_penalty = 1.0 if pred_len > gold_len else math.exp(1 - (gold_len / pred_len))

    return brevity_penalty * precision


def _parse_prediction_answer(prediction: str) -> str:
    stripped = prediction.strip()
    if not stripped:
        return ""

    # Handle fenced JSON if present.
    stripped = re.sub(r"^```(?:json)?\\s*|\\s*```$", "", stripped, flags=re.IGNORECASE)
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            answer = parsed.get("answer", "")
            return str(answer).strip()
    except json.JSONDecodeError:
        pass

    return stripped


def _load_json_objects(raw_text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    idx = 0
    n = len(raw_text)
    records: list[dict[str, Any]] = []

    while idx < n:
        while idx < n and raw_text[idx].isspace():
            idx += 1
        if idx >= n:
            break

        obj, next_idx = decoder.raw_decode(raw_text, idx)
        if isinstance(obj, dict):
            records.append(obj)
        idx = next_idx

    return records


def evaluate_results_file(results_path: Path, grades_path: Path | None = None) -> dict[str, Any]:
    raw_text = results_path.read_text(encoding="utf-8")
    rows = _load_json_objects(raw_text)

    if grades_path is None:
        grades_path = results_path.parent / "grades.txt"

    lines: list[str] = ["question_index\tcategory\tf1\tbleu1"]
    category_totals: dict[int, dict[str, float]] = defaultdict(
        lambda: {"f1": 0.0, "bleu1": 0.0, "count": 0.0}
    )

    for row in rows:
        question_index = int(row.get("question_index", -1))
        category = int(row.get("category", -1))
        ground_truth = str(row.get("ground_truth_answer", ""))
        prediction = str(row.get("prediction", ""))
        predicted_answer = _parse_prediction_answer(prediction)

        f1 = f1_score(predicted_answer, ground_truth)
        bleu1 = bleu1_score(predicted_answer, ground_truth)

        if 1 <= category <= 5:
            category_totals[category]["f1"] += f1
            category_totals[category]["bleu1"] += bleu1
            category_totals[category]["count"] += 1

        lines.append(f"{question_index}\t{category}\t{f1:.4f}\t{bleu1:.4f}")

    category_averages: dict[int, dict[str, float]] = {}
    lines.append("")
    lines.append("category	avg_f1	avg_bleu1	count")
    for category in range(1, 6):
        total = category_totals[category]
        count = int(total["count"])
        avg_f1 = (total["f1"] / count) if count else 0.0
        avg_bleu1 = (total["bleu1"] / count) if count else 0.0
        category_averages[category] = {
            "avg_f1": avg_f1,
            "avg_bleu1": avg_bleu1,
            "count": count,
        }
        lines.append(f"CATEGORY_{category}\t{avg_f1:.4f}\t{avg_bleu1:.4f}\t{count}")

    grades_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "results_path": str(results_path),
        "grades_path": str(grades_path),
        "total_rows": len(rows),
        "averages_by_category": category_averages,
    }
