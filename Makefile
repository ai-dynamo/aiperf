
.PHONY: ruff lint ruff-fix lint-fix format fmt check-format check-fmt install-dev setup-venv install-uv

VENV ?= .venv

ruff lint:
	ruff check .

ruff-fix lint-fix:
	ruff check . --fix

format fmt:
	ruff format .

check-format check-fmt:
	ruff format . --check-only

install-uv:
	curl -LsSf https://astral.sh/uv/install.sh | sh

setup-venv: install-uv
	uv venv $(VENV)
	$(MAKE) install-dev

install-dev:
	. $(VENV)/bin/activate && uv pip install -e .
