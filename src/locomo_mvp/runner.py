from __future__ import annotations

import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from locomo_mvp.dataset import (
    ConversationTurn,
    format_conversation_for_print,
    format_qa_for_print,
    iter_qa_items,
    load_locomo_dataset,
)
from locomo_mvp.ollama_client import OllamaClient, OllamaError
from locomo_mvp.prompts import build_question_only_prompt
from locomo_mvp.results import ResultRecord, ResultWriter

with open("./prompt/memorize_prompt.txt", "r", encoding="utf-8") as f:
    MEMORIZE_PROMPT = f.read()

with open("./prompt/answer_prompt.txt", "r", encoding="utf-8") as f:
    ANSWER_PROMPT = f.read()


@dataclass(slots=True)
class RunOptions:
    data_path: Path
    output_dir: Path
    ollama_url: str
    model: str
    max_samples: int | None = None
    max_questions: int | None = None
    sample_id: str | None = None
    category: int | str | None = None
    dry_run: bool = False
    no_save: bool = False
    hide_conversation: bool = False
    show_prompt: bool = False
    request_timeout_seconds: int = 300


def wipe_memory_artifacts(base_dir: Path | None = None) -> dict[str, bool]:
    root = base_dir or Path.cwd()
    memory_path = root / "memory" / "memory.txt"
    chroma_path = root / "memory" / "chromadb"

    memory_deleted = False
    chroma_deleted = False

    if memory_path.exists():
        memory_path.unlink()
        memory_deleted = True

    if chroma_path.exists():
        shutil.rmtree(chroma_path)
        chroma_deleted = True

    return {
        "memory_deleted": memory_deleted,
        "chromadb_deleted": chroma_deleted,
    }


def _iter_turn_chunks(
    turns: list[ConversationTurn], chunk_size: int = 2
) -> Iterator[list[ConversationTurn]]:
    for idx in range(0, len(turns), chunk_size):
        yield turns[idx : idx + chunk_size]


def _build_memory_prompt(
    sample_id: str,
    session_name: str,
    session_date: str,
    speaker_a: str,
    speaker_b: str,
    turns: list[ConversationTurn],
) -> str:
    rendered_turns: list[str] = []
    for turn in turns:
        dia = f"{turn.dia_id} " if turn.dia_id else ""
        rendered_turns.append(f"- {dia}{turn.speaker}: {turn.text}")

    return (
        "\n"
        f"Sample ID: {sample_id}\n"
        f"Speakers: {speaker_a} / {speaker_b}\n"
        f"Session: {session_name}\n"
        f"Session date: {session_date}\n"
        "Dialogue chunk:\n"
        f"{chr(10).join(rendered_turns)}\n"
    )


def _split_into_sentences(text: str) -> list[str]:
    normalized = " ".join(text.replace("\n", " ").split())
    if not normalized:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def _save_memories_to_chromadb(
    records: list[dict[str, str]],
    chroma_dir: Path,
    model_name: str = "all-MiniLM-L6-v2",
) -> int:
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name=model_name)
    collection = client.get_or_create_collection(
        name="memory", embedding_function=embedding_fn
    )

    ids = [record["id"] for record in records]
    documents = [record["document"] for record in records]
    metadatas = [
        {"source": record["source"], "sample_id": record["sample_id"]}
        for record in records
    ]
    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    return len(records)


def remember(how_many: int, question: str, chroma_dir: Path | None = None) -> list[str]:
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    if how_many <= 0:
        raise ValueError("how_many must be greater than 0")

    search_dir = chroma_dir or (Path("memory") / "chromadb")
    if not search_dir.exists():
        print(f"[ERROR] ChromaDB path does not exist: {search_dir}")
        return []

    client = chromadb.PersistentClient(path=str(search_dir))
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = client.get_collection(name="memory", embedding_function=embedding_fn)
    results = collection.query(query_texts=[question], n_results=how_many)
    documents = results.get("documents", [[]])[0] if results else []

    return documents


def _format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, remaining_seconds = divmod(seconds, 60)
    hours, remaining_minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{remaining_minutes:02d}:{remaining_seconds:02d}"
    return f"{remaining_minutes:02d}:{remaining_seconds:02d}"


def _build_ponder_prompt(
    sample_id: str,
    session_name: str,
    session_date: str,
    speaker_a: str,
    speaker_b: str,
    memory_text: str,
) -> str:
    return (
        "You are reviewing session memory notes from a dialogue.\n"
        "Inspect all ideas, reason about how they connect, and infer additional atomic ideas.\n"
        "Only output new atomic ideas as bullet points, one idea per bullet.\n\n"
        f"Sample ID: {sample_id}\n"
        f"Speakers: {speaker_a} / {speaker_b}\n"
        f"Session: {session_name}\n"
        f"Session date: {session_date}\n"
        "Session memory notes:\n"
        f"{memory_text}\n"
    )


def ponder(
    *,
    memory_path: Path,
    client: OllamaClient,
    dry_run: bool,
    sample_id: str,
    session_name: str,
    session_date: str,
    speaker_a: str,
    speaker_b: str,
) -> tuple[str, list[str]]:
    memory_text = (
        memory_path.read_text(encoding="utf-8").strip() if memory_path.exists() else ""
    )
    if not memory_text:
        memory_path.write_text("", encoding="utf-8")
        return "", []

    prompt = _build_ponder_prompt(
        sample_id=sample_id,
        session_name=session_name,
        session_date=session_date,
        speaker_a=speaker_a,
        speaker_b=speaker_b,
        memory_text=memory_text,
    )

    if dry_run:
        pondered_text = "[DRY RUN - no Ollama call made]"
    else:
        pondered_text = client.generate(prompt)

    ideas = _split_into_sentences(pondered_text)
    memory_path.write_text("", encoding="utf-8")
    return pondered_text, ideas


def run_evaluation(options: RunOptions) -> dict:
    started_at = datetime.now(timezone.utc)
    samples = load_locomo_dataset(options.data_path)
    client = OllamaClient(options.ollama_url, options.model, options.request_timeout_seconds)

    writer = None if options.no_save else ResultWriter(options.output_dir)

    attempted = 0
    successes = 0
    errors = 0
    seen_samples: set[str] = set()

    for example in iter_qa_items(
        samples,
        sample_id=options.sample_id,
        max_samples=options.max_samples,
        max_questions=options.max_questions,
        category=options.category,
    ):
        if example.sample_id not in seen_samples:
            seen_samples.add(example.sample_id)
            print("=" * 80)
            print(f"SAMPLE: {example.sample_id}")
            print(f"SPEAKERS: {example.sample.speaker_a} / {example.sample.speaker_b}")
            print("=" * 80)
            if not options.hide_conversation:
                print()
                print(format_conversation_for_print(example.sample))

        attempted += 1
        prompt = ANSWER_PROMPT + "\n\n"
        prompt += f"CONTEXTUAL MEMORIES: [\n"
        question = build_question_only_prompt(example.question)
        contextual_memories = remember(40, question)
        for m in contextual_memories:
            prompt += f"  {m}\n"
        prompt += f"]\n\n"
        prompt += question

        print("-" * 80)
        print(format_qa_for_print(example))
        if options.show_prompt:
            print("\nPrompt sent to model:\n")
            print(prompt)

        prediction = ""
        err: str | None = None

        if options.dry_run:
            prediction = "[DRY RUN - no Ollama call made]"
            successes += 1
        else:
            try:
                prediction = client.generate(prompt)
                successes += 1
            except OllamaError as exc:
                err = str(exc)
                prediction = ""
                errors += 1

        print("\nModel answer:\n")
        print(prediction if prediction else f"[ERROR] {err}")
        print("-" * 80)

        if writer:
            writer.append_result(
                ResultRecord(
                    run_id=writer.run_id,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    sample_id=example.sample_id,
                    question_index=example.question_index,
                    category=example.category,
                    question=example.question,
                    ground_truth_answer=example.answer,
                    evidence=example.evidence,
                    prompt=prompt,
                    model=options.model,
                    ollama_base_url=options.ollama_url,
                    prediction=prediction,
                    error=err,
                )
            )

    ended_at = datetime.now(timezone.utc)
    summary = {
        "run_id": writer.run_id if writer else None,
        "start_time": started_at.isoformat(),
        "end_time": ended_at.isoformat(),
        "dataset_path": str(options.data_path),
        "model": options.model,
        "ollama_url": options.ollama_url,
        "total_samples_loaded": len(samples),
        "total_questions_attempted": attempted,
        "total_successful_model_calls": successes,
        "total_errors": errors,
        "output_jsonl_path": str(writer.jsonl_path) if writer else None,
    }

    if writer:
        writer.write_summary(summary)

    return summary


def memorize(options: RunOptions) -> dict:
    started_at = datetime.now(timezone.utc)
    samples = load_locomo_dataset(options.data_path)
    client = OllamaClient(options.ollama_url, options.model, options.request_timeout_seconds)

    memory_path = Path("memory") / "memory.txt"
    chroma_dir = Path("memory") / "chromadb"
    memory_path.parent.mkdir(parents=True, exist_ok=True)

    selected_samples = (
        samples[: options.max_samples] if options.max_samples is not None else samples
    )
    total_chunks = sum(
        (len(sample.sessions.get(session_name, [])) + 1) // 2
        for sample in selected_samples
        for session_name in sorted(sample.sessions.keys(), key=lambda s: (len(s), s))
    )

    processed_samples = 0
    processed_chunks = 0
    errors = 0
    vector_entries: list[dict[str, str]] = []
    memorize_started = time.time()

    for sample in selected_samples:
        processed_samples += 1
        session_names = sorted(sample.sessions.keys(), key=lambda s: (len(s), s))

        for session_name in session_names:
            memory_path.write_text("", encoding="utf-8")
            session_date = sample.session_dates.get(session_name, "unknown date")
            turns = sample.sessions.get(session_name, [])

            for chunk in _iter_turn_chunks(turns, chunk_size=2):
                processed_chunks += 1
                prompt = MEMORIZE_PROMPT + "\n\n"
                prompt += _build_memory_prompt(
                    sample_id=sample.sample_id,
                    session_name=session_name,
                    session_date=session_date,
                    speaker_a=sample.speaker_a,
                    speaker_b=sample.speaker_b,
                    turns=chunk,
                )

                if options.dry_run:
                    memory_text = "[DRY RUN - no Ollama call made]"
                else:
                    try:
                        print()
                        print(prompt)
                        memory_text = client.generate(prompt)
                    except OllamaError as exc:
                        errors += 1
                        memory_text = f"[ERROR] {exc}"

                chunk_context = " ".join(
                    f"{turn.speaker}: {turn.text}" for turn in chunk
                )
                for idx, sentence in enumerate(_split_into_sentences(chunk_context)):
                    vector_entries.append(
                        {
                            "id": (
                                f"{sample.sample_id}|{session_name}|chunk{processed_chunks}|"
                                f"ctx|{idx}"
                            ),
                            "document": sentence,
                            "source": "dialogue_context",
                            "sample_id": sample.sample_id,
                        }
                    )

                for idx, sentence in enumerate(_split_into_sentences(memory_text)):
                    vector_entries.append(
                        {
                            "id": (
                                f"{sample.sample_id}|{session_name}|chunk{processed_chunks}|"
                                f"notes|{idx}"
                            ),
                            "document": sentence,
                            "source": "memory_notes",
                            "sample_id": sample.sample_id,
                        }
                    )

                with memory_path.open("a", encoding="utf-8") as memory_file:
                    memory_file.write(
                        f"[{datetime.now(timezone.utc).isoformat()}] "
                        f"sample={sample.sample_id} session={session_name}\n"
                    )
                    memory_file.write(f"{memory_text}\n\n")
                print(memory_text)

                elapsed = time.time() - memorize_started
                avg_per_chunk = elapsed / processed_chunks if processed_chunks else 0
                remaining_chunks = max(total_chunks - processed_chunks, 0)
                eta = _format_eta(avg_per_chunk * remaining_chunks)
                print(
                    f"\rMemorizing: {processed_chunks}/{total_chunks} chunks | ETA {eta}",
                    end="",
                    flush=True,
                )

            try:
                pondered_text, ideas = ponder(
                    memory_path=memory_path,
                    client=client,
                    dry_run=options.dry_run,
                    sample_id=sample.sample_id,
                    session_name=session_name,
                    session_date=session_date,
                    speaker_a=sample.speaker_a,
                    speaker_b=sample.speaker_b,
                )
            except OllamaError as exc:
                errors += 1
                pondered_text, ideas = f"[ERROR] {exc}", []

            if pondered_text:
                print(
                    f"\nPondered ideas ({sample.sample_id} / {session_name}):\n{pondered_text}"
                )

            for idx, idea in enumerate(ideas):
                vector_entries.append(
                    {
                        "id": f"{sample.sample_id}|{session_name}|ponder|{idx}",
                        "document": idea,
                        "source": "pondered_ideas",
                        "sample_id": sample.sample_id,
                    }
                )

    print()

    vector_count = 0
    if vector_entries:
        vector_count = _save_memories_to_chromadb(vector_entries, chroma_dir)

    ended_at = datetime.now(timezone.utc)
    return {
        "start_time": started_at.isoformat(),
        "end_time": ended_at.isoformat(),
        "dataset_path": str(options.data_path),
        "model": options.model,
        "ollama_url": options.ollama_url,
        "total_samples_loaded": len(samples),
        "total_samples_processed": processed_samples,
        "total_dialog_chunks_processed": processed_chunks,
        "total_errors": errors,
        "memory_path": str(memory_path),
        "chromadb_path": str(chroma_dir),
        "total_vector_documents": vector_count,
    }
