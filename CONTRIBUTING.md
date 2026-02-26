<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Contributing to AIPerf

For technical architecture, see [`docs/architecture.md`](docs/architecture.md). For AI assistant instructions, see `CLAUDE.md`.

## Development Setup

### Prerequisites

- **Python 3.10+**
- **uv**: Package manager (installed automatically by `make first-time-setup`)
- **pre-commit**: For automated code quality checks

### Initial Setup

```bash
# Clone the repository
git clone <repository-url>
cd aiperf

# One-command setup: creates venv, installs project + mock server + pre-commit hooks
make first-time-setup

# Or manually:
make setup-venv       # Create virtual environment
make install          # Install project + mock server in editable mode
pre-commit install    # Install pre-commit hooks
```

### Available Commands

| Command | Description |
|---------|-------------|
| `make first-time-setup` | Full environment setup (venv + install + hooks) |
| `make install` | Install project and mock server in editable mode |
| `make install-app` | Install project only |
| `make install-mock-server` | Install mock server only |
| `make test` | Unit tests (parallel, excludes integration) |
| `make test-verbose` | Unit tests with DEBUG logging |
| `make test-all` | All tests (unit + component integration + integration) |
| `make test-integration` | Integration tests with mock server |
| `make test-integration-verbose` | Integration tests with real-time output |
| `make test-component-integration` | Component integration tests |
| `make test-ci` | CI mode: unit + component integration with coverage |
| `make test-imports` | Verify all modules can be imported |
| `make test-stress` | Stress tests with mock server |
| `make coverage` | Unit tests with HTML/XML coverage report |
| `make lint` | Run ruff linter |
| `make lint-fix` | Auto-fix linter errors |
| `make fmt` | Format code with ruff |
| `make check-fmt` | Check formatting without changes |
| `make validate-plugin-schemas` | Validate plugins.yaml against schemas |
| `make generate-all-plugin-files` | Regenerate plugin enums, overloads, schemas |
| `make generate-all-docs` | Regenerate CLI + env var documentation |
| `make generate-cli-docs` | Regenerate CLI documentation |
| `make generate-env-vars-docs` | Regenerate environment variable documentation |
| `make docker` | Build Docker image |
| `make docker-run` | Run Docker container |
| `make clean` | Clean caches and build artifacts |
| `make version` | Print project version |

Direct pytest commands:

```bash
uv run pytest tests/unit/ -n auto                          # Unit tests (parallel)
uv run pytest -m integration -n auto                       # Integration tests (multiprocess)
uv run pytest -m component_integration -n auto             # Component integration tests
```

### Pre-Commit Hooks

The repository uses pre-commit hooks defined in `.pre-commit-config.yaml`:

**General hooks:**
- `check-ast` - Verify Python AST validity
- `debug-statements` - Detect leftover debug statements
- `detect-private-key` - Prevent committing private keys
- `check-added-large-files` - Fail if files > 5MB added
- `check-case-conflict` - Detect case-insensitive filename conflicts
- `check-merge-conflict` - Detect merge conflict markers
- `check-json` - Validate JSON syntax
- `check-toml` - Validate TOML syntax
- `check-yaml` - Validate YAML syntax
- `end-of-file-fixer` - Ensure files end with newline
- `trailing-whitespace` - Remove trailing whitespace
- `mixed-line-ending` - Enforce consistent line endings
- `no-commit-to-branch` - Prevent direct commits to main

**Code quality hooks:**
- `codespell` - Spell checking
- `ruff` - Lint with auto-fix
- `ruff-format` - Format with ruff

**Project-specific hooks:**
- `add-license` - Add SPDX copyright headers
- `generate-cli-docs` - Regenerate CLI documentation when Python files change
- `generate-env-vars-docs` - Regenerate env var docs when environment.py changes
- `generate-plugin-artifacts` - Regenerate plugin enums/overloads/schemas
- `validate-plugin-schemas` - Validate plugin YAML against schemas
- `test-imports` - Verify all modules can be imported

Run pre-commit after every code change, even before creating commits. Do not wait until commit time to discover problems.

### Package Management

Always use `uv` (never pip): `uv add package`, `uv run pytest`.

## Contribution Guidelines

Contributions that fix documentation errors or that make small changes to existing code can be contributed directly by following the rules below and submitting a PR.

Contributions intended to add significant new functionality must follow a more collaborative path. Before submitting a large PR, submit a GitHub issue describing the proposed change so the AIPerf team can provide feedback:

- A design for your change will be agreed upon to ensure consistency with AIPerf's architecture.
- The Dynamo project is spread across multiple GitHub repositories. The AIPerf team will provide guidance about how and where your enhancement should be implemented.
- Testing is critical. Plan on spending significant time on creating tests. The team will help design testing compatible with existing infrastructure.
- User-visible features need documentation.

## Contribution Rules

- Code style is enforced by `ruff` (formatting + linting). Follow existing conventions.
- Avoid introducing unnecessary complexity.
- Keep PRs concise and focused on a single concern.
- Build log must be clean: no warnings or errors.
- All tests must pass.
- No license or patent conflicts. You must certify compliance with the [license terms](https://github.com/ai-dynamo/aiperf/blob/main/LICENSE) and sign off on the [Developer Certificate of Origin (DCO)](https://developercertificate.org).

## Git Workflow

Feature branches use `<username>/feature-name` format, forked from `main`.

## Running GitHub Actions Locally

You can use the `act` tool to run GitHub Actions locally. See [act usage](https://nektosact.com/introduction.html).

```bash
act -j run-integration-tests
```

You can also use the VSCode extension [GitHub Local Actions](https://marketplace.visualstudio.com/items?itemName=SanjulaGanepola.github-local-actions).

## Developer Certificate of Origin

AIPerf is open source under the Apache 2.0 license (see [the Apache site](https://www.apache.org/licenses/LICENSE-2.0) or [LICENSE](./LICENSE)).

We respect intellectual property rights and want to ensure all contributions are correctly attributed and licensed. A Developer Certificate of Origin (DCO) is a lightweight mechanism to do that.

The DCO is a declaration attached to every contribution. In the commit message, the developer adds a `Signed-off-by` statement and thereby agrees to the DCO, which you can find at [DeveloperCertificate.org](http://developercertificate.org/).

We require that every contribution is signed with a DCO, verified by a required CI check. Please use your real name. We do not accept anonymous contributors or pseudonyms.

Each commit must include:

```text
Signed-off-by: Jane Smith <jane.smith@email.com>
```

You can use `-s` or `--signoff` to add the `Signed-off-by` line automatically.

If your pull request fails the DCO check, see the [DCO Troubleshooting Guide](DCO.md).
