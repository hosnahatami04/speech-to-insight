export PYTHONHASHSEED := 0

.PHONY: setup lint format typecheck test cov data evaluate-smoke audit

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
	uv run pytest -m "not ml" --cov --cov-report=term-missing --cov-fail-under=85

evaluate-smoke:
	uv run speechinsight evaluate-smoke

audit:
	uv run pip-audit --strict
