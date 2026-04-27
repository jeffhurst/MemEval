from __future__ import annotations


def build_question_only_prompt(question: str) -> str:
    clean_question = question.strip()
    return f"Question:\n{clean_question}\n\n"
