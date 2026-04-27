from pathlib import Path

import argparse

from locomo_mvp.config import load_settings
from locomo_mvp.runner import RunOptions, memorize, run_evaluation, wipe_memory_artifacts


def build_parser() -> argparse.ArgumentParser:
    settings = load_settings()
    parser = argparse.ArgumentParser(description="LoCoMo MVP evaluator for local Ollama models")
    parser.add_argument("--data", default=str(settings.data_path))
    parser.add_argument("--output-dir", default=str(settings.results_dir))
    parser.add_argument("--ollama-url", default=settings.ollama_base_url)
    parser.add_argument("--model", default=settings.ollama_model)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--category", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--hide-conversation", action="store_true")
    parser.add_argument("--show-prompt", action="store_true")
    parser.add_argument("--memorize", action="store_true")
    parser.add_argument("--wipe", action="store_true", help="Delete memory.txt and ChromaDB before run")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    options = RunOptions(
        data_path=Path(args.data),
        output_dir=Path(args.output_dir),
        ollama_url=args.ollama_url,
        model=args.model,
        max_samples=args.max_samples,
        max_questions=args.max_questions,
        sample_id=args.sample_id,
        category=args.category,
        dry_run=args.dry_run,
        no_save=args.no_save,
        hide_conversation=args.hide_conversation,
        show_prompt=args.show_prompt,
    )

    if args.wipe:
        wipe_result = wipe_memory_artifacts()
        print("Wipe completed:")
        print(f"- memory_deleted: {wipe_result['memory_deleted']}")
        print(f"- chromadb_deleted: {wipe_result['chromadb_deleted']}")

    summary = memorize(options) if args.memorize else run_evaluation(options)
    print("\nRun summary:")
    for key, value in summary.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
