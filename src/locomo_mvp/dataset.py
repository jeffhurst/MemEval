from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


@dataclass(slots=True)
class ConversationTurn:
    speaker: str = "Unknown"
    dia_id: str = ""
    text: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class QaItem:
    question: str = ""
    answer: str = ""
    category: int | str | None = None
    evidence: list[str] = field(default_factory=list)


@dataclass(slots=True)
class LocomoSample:
    sample_id: str
    speaker_a: str = "Speaker A"
    speaker_b: str = "Speaker B"
    session_dates: dict[str, str] = field(default_factory=dict)
    sessions: dict[str, list[ConversationTurn]] = field(default_factory=dict)
    qa: list[QaItem] = field(default_factory=list)


@dataclass(slots=True)
class QaExample:
    sample_id: str
    question_index: int
    question: str
    answer: str
    category: int | str | None
    evidence: list[str]
    sample: LocomoSample


def _to_evidence_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(item) for item in raw]
    return [str(raw)]


def load_locomo_dataset(path: Path) -> list[LocomoSample]:
    with path.open("r", encoding="utf-8") as f:
        raw_samples = json.load(f)

    if not isinstance(raw_samples, list):
        raise ValueError("LoCoMo dataset root must be a JSON list.")

    samples: list[LocomoSample] = []
    for idx, raw in enumerate(raw_samples):
        sample_id = str(raw.get("sample_id", f"sample_{idx}"))
        conversation = raw.get("conversation", {}) or {}
        speaker_a = str(conversation.get("speaker_a", "Speaker A"))
        speaker_b = str(conversation.get("speaker_b", "Speaker B"))

        session_dates: dict[str, str] = {}
        sessions: dict[str, list[ConversationTurn]] = {}

        for key, value in conversation.items():
            if key.startswith("session_") and key.endswith("_date_time"):
                session_name = key[: -len("_date_time")]
                session_dates[session_name] = str(value)
            elif key.startswith("session_") and isinstance(value, list):
                turns: list[ConversationTurn] = []
                for turn in value:
                    if not isinstance(turn, dict):
                        continue
                    extra = {
                        k: v
                        for k, v in turn.items()
                        if k not in {"speaker", "dia_id", "text"}
                    }
                    turns.append(
                        ConversationTurn(
                            speaker=str(turn.get("speaker", "Unknown")),
                            dia_id=str(turn.get("dia_id", "")),
                            text=str(turn.get("text", "")),
                            extra=extra,
                        )
                    )
                sessions[key] = turns

        qa_items: list[QaItem] = []
        for qa in raw.get("qa", []) or []:
            if not isinstance(qa, dict):
                continue
            qa_items.append(
                QaItem(
                    question=str(qa.get("question", "")),
                    answer=str(qa.get("answer", "")),
                    category=qa.get("category"),
                    evidence=_to_evidence_list(qa.get("evidence")),
                )
            )

        samples.append(
            LocomoSample(
                sample_id=sample_id,
                speaker_a=speaker_a,
                speaker_b=speaker_b,
                session_dates=session_dates,
                sessions=sessions,
                qa=qa_items,
            )
        )

    return samples


def iter_qa_items(
    samples: list[LocomoSample],
    sample_id: str | None = None,
    max_samples: int | None = None,
    max_questions: int | None = None,
    category: int | str | None = None,
) -> Iterator[QaExample]:
    yielded = 0
    processed_samples = 0

    normalized_category = str(category) if category is not None else None

    for sample in samples:
        if sample_id and sample.sample_id != sample_id:
            continue

        if max_samples is not None and processed_samples >= max_samples:
            break

        processed_samples += 1

        for q_idx, qa in enumerate(sample.qa):
            if normalized_category is not None and str(qa.category) != normalized_category:
                continue

            if max_questions is not None and yielded >= max_questions:
                return

            yield QaExample(
                sample_id=sample.sample_id,
                question_index=q_idx,
                question=qa.question,
                answer=qa.answer,
                category=qa.category,
                evidence=qa.evidence,
                sample=sample,
            )
            yielded += 1


def format_conversation_for_print(sample: LocomoSample) -> str:
    lines = [
        f"SAMPLE: {sample.sample_id}",
        f"SPEAKERS: {sample.speaker_a} / {sample.speaker_b}",
        "",
        "--- CONVERSATION ---",
    ]

    session_names = sorted(sample.sessions.keys(), key=lambda s: (len(s), s))
    for session_name in session_names:
        date_str = sample.session_dates.get(session_name, "unknown date")
        lines.append(f"[{session_name} | {date_str}]")
        for turn in sample.sessions[session_name]:
            prefix = f"{turn.dia_id} " if turn.dia_id else ""
            lines.append(f"{prefix}{turn.speaker}: {turn.text}")
        lines.append("")

    return "\n".join(lines).strip()


def format_qa_for_print(example: QaExample) -> str:
    evidence = ", ".join(example.evidence) if example.evidence else "(none)"
    return (
        f"QA {example.question_index + 1}\n"
        f"Sample ID: {example.sample_id}\n"
        f"Category: {example.category}\n"
        f"Evidence: {evidence}\n\n"
        f"Question:\n{example.question}\n\n"
        f"Ground truth:\n{example.answer}"
    )
