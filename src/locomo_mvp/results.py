from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ResultRecord:
    run_id: str
    timestamp: str
    sample_id: str
    question_index: int
    category: int | str | None
    question: str
    ground_truth_answer: str
    evidence: list[str]
    prompt: str
    model: str
    ollama_base_url: str
    prediction: str
    error: str | None


class ResultWriter:
    def __init__(self, output_dir: Path, run_id: str | None = None) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.run_id = run_id or f"locomo_mvp_run_{ts}"
        self.jsonl_path = self.output_dir / f"{self.run_id}.jsonl"
        self.summary_path = self.output_dir / f"{self.run_id}_summary.json"

    def append_result(self, record: ResultRecord) -> None:
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    def write_summary(self, summary: dict[str, Any]) -> None:
        with self.summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            f.write("\n")
