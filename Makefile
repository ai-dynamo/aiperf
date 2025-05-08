
.PHONY: ruff lint ruff-fix lint-fix format fmt check-format check-fmt

ruff lint:
	ruff check .

ruff-fix lint-fix:
	ruff check . --fix

format fmt:
	ruff format .

check-format check-fmt:
	ruff format . --check
