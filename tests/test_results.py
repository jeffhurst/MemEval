import json
from pathlib import Path

from locomo_mvp.results import ResultRecord, ResultWriter


def test_result_writer_outputs_files(tmp_path: Path) -> None:
    writer = ResultWriter(tmp_path, run_id="locomo_mvp_run_20260101_000000")
    writer.append_result(
        ResultRecord(
            run_id=writer.run_id,
            timestamp="2026-01-01T00:00:00+00:00",
            sample_id="sample_1",
            question_index=0,
            category=1,
            question="Q",
            ground_truth_answer="A",
            evidence=["D1:1"],
            prompt="P",
            model="m",
            ollama_base_url="u",
            prediction="pred",
            error=None,
        )
    )
    writer.write_summary({"run_id": writer.run_id, "total": 1})

    assert writer.jsonl_path.exists()
    assert writer.summary_path.exists()
    line = writer.jsonl_path.read_text(encoding="utf-8").strip().splitlines()[0]
    parsed = json.loads(line)
    assert parsed["sample_id"] == "sample_1"
