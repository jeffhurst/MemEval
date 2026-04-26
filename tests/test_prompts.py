from locomo_mvp.prompts import build_question_only_prompt


def test_prompt_question_only() -> None:
    prompt = build_question_only_prompt("When?")
    assert "When?" in prompt
    assert "ground" not in prompt.lower()
    assert "evidence" not in prompt.lower()
    assert "session_1" not in prompt
