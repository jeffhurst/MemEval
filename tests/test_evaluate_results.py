from pathlib import Path

from locomo_mvp.evaluate_results import evaluate_results_file


def test_evaluate_results_writes_category_averages(tmp_path: Path) -> None:
    results_path = tmp_path / "sample.jsonl"
    results_path.write_text(
        '{"question_index": 0, "category": 1, "ground_truth_answer": "7 May 2023", "prediction": "{\\"answer\\": \\"May 7, 2023\\"}"}\n'
        '{"question_index": 1, "category": 2, "ground_truth_answer": "2022", "prediction": "{\\"answer\\": \\"Last year\\"}"}\n',
        encoding="utf-8",
    )

    summary = evaluate_results_file(results_path)

    grades_path = tmp_path / "grades.txt"
    assert grades_path.exists()
    text = grades_path.read_text(encoding="utf-8")
    assert "question_index\tcategory\tf1\tbleu1" in text
    assert "CATEGORY_1" in text
    assert "CATEGORY_2" in text
    assert "CATEGORY_5" in text
    assert summary["total_rows"] == 2
    assert len(summary["averages_by_category"]) == 5
