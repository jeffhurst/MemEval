import json
from pathlib import Path
import types
import sys

from locomo_mvp.runner import RunOptions, memorize, ponder, remember, run_evaluation, wipe_memory_artifacts


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


def test_memorize_writes_vectors_and_clears_memory_file(tmp_path: Path, monkeypatch) -> None:
    data = tmp_path / "data.json"
    data.write_text(
        json.dumps(
            [
                {
                    "sample_id": "sample_1",
                    "conversation": {
                        "speaker_a": "A",
                        "speaker_b": "B",
                        "session_1": [
                            {"speaker": "A", "dia_id": "D1:1", "text": "I like tea."},
                            {"speaker": "B", "dia_id": "D1:2", "text": "Let's meet Monday."},
                            {"speaker": "A", "dia_id": "D1:3", "text": "Great idea."},
                        ],
                    },
                    "qa": [],
                }
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("locomo_mvp.ollama_client.OllamaClient.generate", lambda self, prompt: "- memory fact")
    captured: dict[str, object] = {}

    def _fake_save(records, chroma_dir):
        captured["records"] = records
        captured["chroma_dir"] = chroma_dir
        return len(records)

    monkeypatch.setattr("locomo_mvp.runner._save_memories_to_chromadb", _fake_save)
    monkeypatch.chdir(tmp_path)

    summary = memorize(
        RunOptions(
            data_path=data,
            output_dir=tmp_path / "out",
            ollama_url="http://localhost:11434",
            model="gemma4:e4b",
            max_samples=1,
        )
    )

    memory_file = tmp_path / "memory" / "memory.txt"
    assert memory_file.exists()
    content = memory_file.read_text(encoding="utf-8")
    assert content == ""
    assert summary["total_dialog_chunks_processed"] == 2
    assert summary["total_vector_documents"] > 0
    assert str(captured["chroma_dir"]).endswith("memory/chromadb")
    assert any(record["source"] == "dialogue_context" for record in captured["records"])
    assert any(record["source"] == "memory_notes" for record in captured["records"])
    assert any(record["source"] == "pondered_ideas" for record in captured["records"])


def test_ponder_generates_ideas_and_wipes_memory(tmp_path: Path, monkeypatch) -> None:
    memory_file = tmp_path / "memory.txt"
    memory_file.write_text("- idea 1\n- idea 2\n", encoding="utf-8")
    monkeypatch.setattr(
        "locomo_mvp.ollama_client.OllamaClient.generate",
        lambda self, prompt: "New idea one. New idea two.",
    )

    from locomo_mvp.ollama_client import OllamaClient

    client = OllamaClient("http://localhost:11434", "gemma4:e4b")
    _, ideas = ponder(
        memory_path=memory_file,
        client=client,
        dry_run=False,
        sample_id="sample_1",
        session_name="session_1",
        session_date="2026-01-01",
        speaker_a="A",
        speaker_b="B",
    )

    assert ideas == ["New idea one.", "New idea two."]
    assert memory_file.read_text(encoding="utf-8") == ""


def test_wipe_memory_artifacts(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    (memory_dir / "memory.txt").write_text("test", encoding="utf-8")
    chroma_dir = memory_dir / "chromadb"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    (chroma_dir / "file.bin").write_text("x", encoding="utf-8")

    result = wipe_memory_artifacts(tmp_path)

    assert result == {"memory_deleted": True, "chromadb_deleted": True}
    assert not (memory_dir / "memory.txt").exists()
    assert not chroma_dir.exists()


def test_remember_queries_chromadb_and_returns_documents(tmp_path: Path, monkeypatch) -> None:
    chroma_dir = tmp_path / "memory" / "chromadb"
    chroma_dir.mkdir(parents=True, exist_ok=True)

    class _FakeCollection:
        def query(self, query_texts, n_results):
            assert query_texts == ["what do i like?"]
            assert n_results == 2
            return {"documents": [["I like tea.", "We meet Monday."]]}

    class _FakeClient:
        def __init__(self, path):
            assert path == str(chroma_dir)

        def get_collection(self, name, embedding_function):
            assert name == "memory"
            assert embedding_function is not None
            return _FakeCollection()

    fake_chromadb = types.ModuleType("chromadb")
    fake_chromadb.PersistentClient = _FakeClient
    fake_utils = types.ModuleType("chromadb.utils")
    fake_embedding_functions = types.ModuleType("chromadb.utils.embedding_functions")
    fake_embedding_functions.SentenceTransformerEmbeddingFunction = lambda model_name: object()
    fake_utils.embedding_functions = fake_embedding_functions

    monkeypatch.setitem(sys.modules, "chromadb", fake_chromadb)
    monkeypatch.setitem(sys.modules, "chromadb.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "chromadb.utils.embedding_functions", fake_embedding_functions)

    docs = remember(2, "what do i like?", chroma_dir=chroma_dir)
    assert docs == ["I like tea.", "We meet Monday."]
