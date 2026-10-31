export PYTHONHASHSEED := 0

.PHONY: setup lint format typecheck test cov data

setup:
	uv sync --all-extras

data:
	uv run speechinsight data

lint:
	uv run ruff check src tests

format:
	uv run ruff format src tests

typecheck:
	uv run mypy

test:
	uv run pytest -m "not ml"

cov:
	uv run pytest -m "not ml" --cov --cov-report=term-missing
