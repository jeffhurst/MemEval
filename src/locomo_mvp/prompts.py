from __future__ import annotations


def build_question_only_prompt(question: str) -> str:
    clean_question = question.strip()
    return (
        "You are answering a question from a long-term conversation memory benchmark.\n\n"
        "For this MVP, you are NOT given the conversation history.\n"
        "Answer as best you can from the question alone.\n"
        "If the answer cannot be determined from the prompt, say: \"Not enough information in prompt.\"\n\n"
        f"Question:\n{clean_question}\n\n"
        "Answer:\n"
    )
