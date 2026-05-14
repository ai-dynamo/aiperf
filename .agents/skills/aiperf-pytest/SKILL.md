---
name: aiperf-pytest
description: Use BEFORE invoking pytest in the aiperf repo for any reason — "run the tests", "run unit tests", "verify with pytest", "kick off the integration suite", "re-run the failing tests", "pytest tests/unit/...". Codifies the project's pytest invocation rules (always -n auto, never combine subfolders, slow-marker deselect, MALLOC_ARENA_MAX for integration, anti-collision pre-flight). Wraps the single canonical command per test tier.
---

# AIPerf Pytest Invocation

A guardrail wrapper around `uv run pytest`. The aiperf test suite has enough operational gotchas — xdist worker OOM, slow-marker stalls at 98%, MALLOC arena SIGABRT under load, dead-letter on subfolder-combine — that "just run pytest" is a 30-minute mistake.

## The iron rule

**One subfolder per pytest command.** Never two. The project's test tiers (unit, component_integration, integration) have different fixture sets, conftest behaviors, and marker semantics. Combining them in one invocation produces confused collection and slow runs.

```bash
# WRONG — combines subfolders
uv run pytest -n auto tests/unit/ tests/component_integration/

# WRONG — runs by marker across the tree (collects everything, triggers slow-marker stall)
uv run pytest -n auto -m component_integration

# RIGHT — one tier per invocation, path-scoped
uv run pytest -n auto tests/unit/
uv run pytest -n auto tests/component_integration/
uv run pytest -n auto tests/integration/
```

## Pre-flight (mandatory)

Before launching, check for concurrent pytest processes — `slow`-marked corpus tests can OOM xdist workers when two `-n auto` shells overlap, producing a silent `cannot send (already closed?)` hang at 98%.

```bash
pgrep -af pytest | wc -l   # expect 0 (or only your own shell's grep)
```

If non-zero and they're not yours: wait, or kill them with the user's approval. Do NOT spawn another `-n auto` shell concurrent with an active one.

## Canonical commands by tier

| Tier | Command | Notes |
|---|---|---|
| Unit | `uv run pytest -n auto tests/unit/` | Fast, isolated. Default. |
| Component integration | `uv run pytest -n auto tests/component_integration/` | Single-process, real services. |
| Integration | `MALLOC_ARENA_MAX=2 uv run pytest -n auto tests/integration/` | Multiprocess. **You must set `MALLOC_ARENA_MAX=2` explicitly** — it is NOT pre-set in `tests/integration/conftest.py` on this branch. Without it, xdist workers SIGABRT under heavy load. |
| Single test file | `uv run pytest -n auto tests/<tier>/<path>/test_<name>.py` | Same `-n auto` rule applies. |
| Single test ID | `uv run pytest tests/<tier>/.../test_x.py::test_func` | `-n auto` not required when targeting a single test ID; omit to keep output readable. |

The `-n auto` rule is non-negotiable — even single-file invocations get it (xdist handles small N gracefully).

## Slow-marker handling

`slow`-marked tests can OOM xdist workers under heavy concurrency. On this branch the `slow` marker is **run by default** (not auto-deselected), and there is no global pytest `--timeout=300`. If two concurrent `-n auto` shells overlap, the slow corpus can OOM workers and produce the silent `cannot send (already closed?)` stall at 98%.

If you hit the stall, deselect explicitly:

```bash
uv run pytest -n auto tests/unit/ -m 'not slow' --timeout=300
```

If the user explicitly asks to run slow tests:

```bash
uv run pytest -n 4 tests/unit/ -m slow   # NOT -n auto for slow corpus
```

`-n 4` (or lower) instead of `-n auto` — slow tests are memory-heavy and need worker headroom.

## Re-running selectively

If a test fails and you need to investigate, re-run the SPECIFIC failing test ID (no `-n auto` needed for a single test) rather than the entire tier. Re-running the whole tier without a code change wastes compute and rarely surfaces new information.

## Red flags — STOP, you're rationalizing

| Thought | Reality |
|---|---|
| "I'll just combine the unit and integration folders for speed" | One subfolder per command. Splitting is faster in practice (no cross-tier fixture pollution, faster fail-fast). |
| "I'll run `-m integration` across the tree, it's clean" | Collects whole tree → triggers slow-marker fixture flakiness. Use the path-scoped form. |
| "Another pytest is running but mine is a different tier so it's fine" | xdist workers from both `-n auto` shells fight over CPU; one of them OOMs at ~98%. Wait for the other to finish. |
| "I'll bump `-n auto` to `-n 16` to go faster" | OOM territory. The defaults are tuned. Leave them. |
| "I'll re-run the whole tier 'just to be sure'" | One canonical run = evidence. Don't re-run unless you changed something. |
| "I'll skip the `pgrep` pre-flight" | The stall is silent — you'll lose 20 min wondering why xdist 'hangs at 98%'. Pre-flight takes 1 second. |

## Common mistakes

- **`pytest` without `uv run`** — uses system Python, not the project venv. Path issues, missing deps, confusing errors.
- **Running tests from a worktree without `make first-time-setup`** — uses a stale venv. Always re-setup a fresh worktree (see `aiperf-worktree`).
- **`ModuleNotFoundError` from `test-imports` or any pytest run** — the venv is behind `pyproject.toml`. Re-sync with `make first-time-setup`, or for a quick fix install the missing package directly with `uv pip install <pkg>` (never `pip install`). See `aiperf-debug` "Tooling drift" for the canonical recovery.

## Output

Don't pipe to `tee` without `PYTHONUNBUFFERED=1` + `set -o pipefail`. For long runs, prefer logging to a file:

```bash
PYTHONUNBUFFERED=1 uv run pytest -n auto tests/unit/ 2>&1 | tee /tmp/pytest-unit-$(date +%s).log
```

`set -o pipefail` lets the failure exit code propagate so you don't false-success.

Also remember: forgetting `MALLOC_ARENA_MAX=2` when invoking integration tests is the most common cause of SIGABRT under load — it is NOT pre-set in conftest on this branch, so set it explicitly every time.
