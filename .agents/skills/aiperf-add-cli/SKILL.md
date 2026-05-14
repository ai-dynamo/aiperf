---
name: aiperf-add-cli
description: Use BEFORE adding a new `aiperf <command>` subcommand or new CLI flag/option — "add a new aiperf command", "add a --flag to profile", "expose this as a CLI option", "register a new cyclopts command", "make this configurable from the CLI". Commands live in src/aiperf/cli_commands/ (one file each), exported as `app: cyclopts.App`, lazily registered via import strings from src/aiperf/cli.py. Heavy imports at module-top break `aiperf --help` perf. Pre-commit's generate-cli-docs hook will rewrite docs and trigger the heredoc-reflow trap unless you regen first.
---

# AIPerf Add CLI Command / Option

The aiperf CLI uses cyclopts with **lazy registration** from `src/aiperf/cli.py`. Each command lives in its own file under `src/aiperf/cli_commands/`. Two project-specific rules drive the layout:

1. **No heavy imports at module top** — `aiperf --help` must stay sub-second. Import expensive modules INSIDE the function body.
2. **Lazy registration via import strings** — `cli.py` registers commands by string path (`aiperf.cli_commands.profile:app`), not by importing them, so command modules only load when invoked.

## Existing commands

```
src/aiperf/cli_commands/
  profile.py             # aiperf profile (the main benchmark command)
  plot.py                # aiperf plot
  synthesize.py          # aiperf synthesize agentic-code
  validate.py            # aiperf validate
  speed_bench_report.py  # aiperf speed-bench-report
  plugins.py             # aiperf plugins
  analyze_trace.py       # aiperf analyze-trace
  service.py             # aiperf service (single-service launcher)
```

Read any of these for the canonical shape before adding a new one.

## Steps for a new command

### 1. Create the command file

```python
# src/aiperf/cli_commands/your_command.py
import cyclopts
from typing import Annotated
from pydantic import Field

app = cyclopts.App(name="your-command", help="One-line summary of what this does.")

@app.default
def run(
    arg: Annotated[str, cyclopts.Parameter(name=("--arg", "-a"))] = "default",
    flag: Annotated[bool, cyclopts.Parameter(name="--flag")] = False,
) -> None:
    """Long-form help that aiperf your-command --help will show."""
    # IMPORT EXPENSIVE THINGS HERE, not at module top
    from aiperf.dataset.synthesis import some_expensive_thing
    ...
```

### 2. Register lazily in `cli.py`

```python
# src/aiperf/cli.py — register with lazy import strings (positional, not obj=)
app.command("aiperf.cli_commands.your_command:app", name="your-command")
```

The first positional argument IS the import string. This avoids triggering the import of `your_command.py` (and its heavy deps) until the user actually runs `aiperf your-command`.

### 3. Add a flag to an existing command

For `aiperf profile` (and similar commands that accept Pydantic config models), the canonical pattern is to add a `Field` to the relevant Pydantic config model under `src/aiperf/common/config/` rather than to the CLI command file directly. cyclopts derives the flag from the Pydantic field metadata.

For a self-contained new command that doesn't use a Pydantic config:

```python
@app.default
def run(
    arg: Annotated[str, cyclopts.Parameter(name=("--arg", "-a"))] = "default",
    flag: Annotated[bool, cyclopts.Parameter(name="--flag")] = False,
) -> None:
    ...
```

The `Field(description=...)` requirement applies to Pydantic-driven flags; `make generate-cli-docs` reads it to populate `docs/cli-options.md`.

### 4. Regenerate docs

```bash
make generate-cli-docs
```

This regenerates `docs/cli-options.md` from cyclopts introspection. Pre-commit's `generate-cli-docs` hook re-runs on commit; running manually first avoids the heredoc-reflow trap.

### 5. Validate

```bash
# Did help stay fast?
time aiperf --help >/dev/null   # should stay snappy (~1s on dev hardware); if it doubled, you imported too eagerly. No CI gate — quality bar, not enforced.

# Does the new command/flag work?
aiperf your-command --arg foo --flag
# or
aiperf profile --your-new-flag 200 ...
```

### 6. Docs

Per CLAUDE.md's Documentation table: CLI changes update `docs/cli-options.md` (auto via `make generate-cli-docs`). For a new command, also add a tutorial section to `docs/tutorials/` if the command is user-facing, and add the tutorial to `README.md`'s tutorial index and `docs/index.yml`.

## When the flag selects a plugin variant

If your new flag picks among plugin implementations (e.g., `--timing-mode burst` selects a specific `TimingStrategy` plugin), the flag's value should map to the plugin's enum (`TimingMode`). Don't hardcode the registry lookup inside the CLI command — use `plugins.get_class(PluginType.X, name)`.

## Red flags — STOP, you're rationalizing

| Thought | Reality |
|---|---|
| "I'll just import everything at the top of my command file" | `aiperf --help` perf will regress. Import inside the function. |
| "I'll register it directly with `app.command(your_command)` not the string form" | Direct registration imports the module at startup. Use the positional string form (`app.command("aiperf.cli_commands.X:app", name="X")`) to keep lazy. NOTE: it's positional, NOT `obj=` — `app.command(name=..., obj=...)` raises TypeError. |
| "Skip `Field(description=...)`, it's just a flag" | Required. `make generate-cli-docs` uses it to populate `docs/cli-options.md`. |
| "Skip `make generate-cli-docs`, pre-commit will run it" | When pre-commit rewrites the docs mid-commit, the heredoc message is lost. Regenerate manually first. |
| "I'll add a `--mode <plugin-name>` string flag without the enum" | Strings drift; the enum is canonical. Use `Literal[...]` or the actual enum type so cyclopts validates and `--help` shows the choices. |
| "The new command is a script not a command" | Scripts go in `tools/`. CLI commands go in `cli_commands/` and integrate with the help system. Pick one path; don't mix. |

## Common mistakes

- **Heavy imports at module top.** `import torch`, `import transformers`, `from aiperf.dataset.synthesis import ...` at top will balloon `--help` time. Move them inside the function.
- **Using `argparse` instead of `cyclopts`.** All existing commands use cyclopts; sticking to it keeps the surface uniform.
- **Forgetting to add the command to `README.md`'s tutorial index** when it's a user-facing command — fern docs may build but won't be discoverable.
- **Default values that depend on `Environment.X`** — Pydantic resolves defaults at definition time, not invocation. Read `Environment.X` inside the function body or use `Field(default_factory=lambda: Environment.X)`.

## Composition

- `aiperf-commit` for the staging step (the cli-docs regen is a known reflow trigger).
- `aiperf-add-env-var` if the flag's default should come from a tunable env var.
- `aiperf-add-plugin` if the command wires a new plugin category.
