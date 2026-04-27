from locomo_mvp import cli


def test_main_search_db_mode(monkeypatch) -> None:
    called = {"remember": None, "memorize": False, "run_eval": False}

    monkeypatch.setattr(
        "sys.argv",
        ["locomo_mvp", "--searchDB"],
    )
    monkeypatch.setattr("builtins.input", lambda prompt: "2" if "how many" in prompt else "question?")
    monkeypatch.setattr("locomo_mvp.cli.remember", lambda n, q: called.__setitem__("remember", (n, q)))
    monkeypatch.setattr("locomo_mvp.cli.memorize", lambda options: called.__setitem__("memorize", True))
    monkeypatch.setattr("locomo_mvp.cli.run_evaluation", lambda options: called.__setitem__("run_eval", True))

    cli.main()

    assert called["remember"] == (2, "question?")
    assert called["memorize"] is False
    assert called["run_eval"] is False
