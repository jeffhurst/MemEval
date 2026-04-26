from pathlib import Path

from locomo_mvp.dataset import iter_qa_items, load_locomo_dataset


def _write_fixture(path: Path) -> None:
    path.write_text(
        """
[
  {
    "sample_id": "sample_1",
    "conversation": {
      "speaker_a": "A",
      "speaker_b": "B",
      "session_1_date_time": "2023-01-01",
      "session_1": [{"speaker": "A", "dia_id": "D1:1", "text": "hello"}]
    },
    "qa": [
      {"question": "Q1", "answer": "A1", "category": 1, "evidence": ["D1:1"]},
      {"question": "Q2", "answer": "A2", "category": 2}
    ]
  },
  {
    "sample_id": "sample_2",
    "conversation": {"session_1": []},
    "qa": [{"question": "Q3", "answer": "A3", "category": 2}]
  }
]
        """.strip(),
        encoding="utf-8",
    )


def test_loader_and_filters(tmp_path: Path) -> None:
    data = tmp_path / "tiny.json"
    _write_fixture(data)

    samples = load_locomo_dataset(data)
    assert len(samples) == 2
    assert samples[0].speaker_a == "A"
    assert len(samples[0].qa) == 2

    assert len(list(iter_qa_items(samples, max_samples=1))) == 2
    assert len(list(iter_qa_items(samples, max_questions=2))) == 2
    assert len(list(iter_qa_items(samples, sample_id="sample_2"))) == 1
    assert len(list(iter_qa_items(samples, category=2))) == 2
