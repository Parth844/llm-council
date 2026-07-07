.PHONY: run test eval install

install:
	uv venv --allow-existing && uv pip install -e ".[dev]"

run:
	uv run uvicorn council.api:app --reload --port 8000

test:
	uv run pytest -q

eval:
	uv pip install -q -e ".[eval]"
	uv run python eval/run_eval.py --n 100 --rounds 2
