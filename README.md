# LoCoMo MVP Evaluator (Ollama)

A minimal Python CLI harness to run LoCoMo10 QA questions against a **local Ollama model**.

## What this project does

- Loads `data/locomo10.json`
- Iterates LoCoMo samples and QA items
- Builds a **question-only** prompt (no conversation context)
- Calls local Ollama (`/api/generate`)
- Prints sample/QA/model output to terminal
- Saves per-question JSONL results and a run summary JSON

## What this project intentionally does NOT do (yet)

- No memory or retrieval
- No RAG / embeddings / vector DB
- No conversation-history injection
- No automatic correctness scoring
- No LoCoMo F1 evaluation

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -e .[dev]
```

3. Copy env template:

```bash
cp .env.example .env
```

## Run Ollama

Start Ollama locally (default expected URL is `http://localhost:11434`).

Pull the model:

```bash
ollama pull gemma4:e4b
```

## Run the evaluator

```bash
python -m locomo_mvp --max-samples 1 --max-questions 3
```

### Dry run (no Ollama calls)

```bash
python -m locomo_mvp --dry-run --max-samples 1 --max-questions 3
```

## Results location

Outputs are saved under `results/`:

- `locomo_mvp_run_YYYYMMDD_HHMMSS.jsonl`
- `locomo_mvp_run_YYYYMMDD_HHMMSS_summary.json`

## Tests

```bash
pytest
```
