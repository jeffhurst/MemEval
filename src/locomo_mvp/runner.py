from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from locomo_mvp.dataset import (
    format_conversation_for_print,
    format_qa_for_print,
    iter_qa_items,
    load_locomo_dataset,
)
from locomo_mvp.ollama_client import OllamaClient, OllamaError
from locomo_mvp.prompts import build_question_only_prompt
from locomo_mvp.results import ResultRecord, ResultWriter


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


def run_evaluation(options: RunOptions) -> dict:
    started_at = datetime.now(timezone.utc)
    samples = load_locomo_dataset(options.data_path)
    client = OllamaClient(options.ollama_url, options.model)

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
        prompt = build_question_only_prompt(example.question)

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
