import json
from pathlib import Path

from locomo_mvp.runner import RunOptions, run_evaluation


def _dataset(path: Path) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "sample_id": "sample_1",
                    "conversation": {"speaker_a": "A", "speaker_b": "B", "session_1": []},
                    "qa": [{"question": "Q1", "answer": "A1", "category": 1}],
                }
            ]
        ),
        encoding="utf-8",
    )


def test_runner_dry_run(tmp_path: Path) -> None:
    data = tmp_path / "data.json"
    _dataset(data)
    summary = run_evaluation(
        RunOptions(
            data_path=data,
            output_dir=tmp_path / "out",
            ollama_url="http://localhost:11434",
            model="gemma4:e4b",
            dry_run=True,
        )
    )
    assert summary["total_questions_attempted"] == 1
    assert summary["total_errors"] == 0


def test_runner_mocked_ollama(tmp_path: Path, monkeypatch) -> None:
    data = tmp_path / "data.json"
    _dataset(data)

    monkeypatch.setattr("locomo_mvp.ollama_client.OllamaClient.generate", lambda self, prompt: "mocked")

    summary = run_evaluation(
        RunOptions(
            data_path=data,
            output_dir=tmp_path / "out2",
            ollama_url="http://localhost:11434",
            model="gemma4:e4b",
        )
    )
    assert summary["total_successful_model_calls"] == 1
    assert summary["total_errors"] == 0
