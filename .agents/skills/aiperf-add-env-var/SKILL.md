---
name: aiperf-add-env-var
description: Use BEFORE adding any tunable, timeout, threshold, port, or constant to aiperf code — "add a timeout", "make X configurable", "add an env var", "add a setting", "expose Y as an AIPERF_* env var", code review noticing module-level constants. Project rule: every tunable lives as a Field on a _XxxSettings class in src/aiperf/common/environment.py, accessible via Environment.<SECTION>.<NAME>, configurable via AIPERF_<SECTION>_<NAME> env var. Module/class constants in service code get rejected on review.
---

# AIPerf Add Environment Variable / Tunable

The aiperf project rule: **every tunable lives as a Pydantic `Field` on a `_XxxSettings` class in `src/aiperf/common/environment.py`** — never as a module-level or class-level constant in service code. Failure to follow this lands a tunable that can't be tuned without a code change.

## Why

- All tunables get an `AIPERF_*` env var automatically.
- All tunables get documented in `docs/environment-variables.md` automatically via `make generate-env-vars-docs`.
- Section-based grouping keeps the surface scannable.
- Single source of truth: `Environment.HTTP.SO_RCVBUF`, not three constants scattered across services.

## Settings-class taxonomy

```
src/aiperf/common/environment.py:

class _APIServerSettings    → Environment.API_SERVER.*
class _CompressionSettings  → Environment.COMPRESSION.*
class _ConfigSettings       → Environment.CONFIG.*
class _DatasetSettings      → Environment.DATASET.*
class _DeveloperSettings    → Environment.DEV.*
class _GPUSettings          → Environment.GPU.*
class _HTTPSettings         → Environment.HTTP.*
class _LoggingSettings      → Environment.LOGGING.*
class _MetricsSettings      → Environment.METRICS.*
class _MLflowSettings       → Environment.MLFLOW.*
class _OTelSettings         → Environment.OTEL.*
class _RecordSettings       → Environment.RECORD.*
class _ServerMetricsSettings → Environment.SERVER_METRICS.*
class _ServiceSettings      → Environment.SERVICE.*
class _TimingSettings       → Environment.TIMING.*
class _TokenizerSettings    → Environment.TOKENIZER.*
class _UISettings           → Environment.UI.*
class _WorkerSettings       → Environment.WORKER.*
class _ZMQSettings          → Environment.ZMQ.*
```

Pick the closest existing section. Add a new `_XxxSettings` class only if your tunable doesn't belong anywhere existing AND the section makes sense as its own grouping. New telemetry tunables go under `_OTelSettings` / `_MLflowSettings`; tokenizer tunables under `_TokenizerSettings`.

## Steps

### 1. Add the Field

```python
# src/aiperf/common/environment.py inside the right _XxxSettings class

your_tunable: Annotated[
    int,
    Field(
        default=1000,
        description="One-line semantic description. What it controls, units, when to tune.",
        ge=1,           # constrain if appropriate
        le=10000,
    ),
] = 1000
```

The `Field(description=...)` is **required** per the project's coding standards (it's how `make generate-env-vars-docs` populates the docs entry). No exceptions.

### 2. Pick up the value at the call site

```python
# src/aiperf/your_service.py
from aiperf.common.environment import Environment

class YourService(BaseComponentService):
    @on_start
    async def start(self) -> None:
        self._timeout = Environment.HTTP.YOUR_TUNABLE
        ...
```

Do NOT cache the value at module import time — `Environment.*` resolves lazily and respects env-var overrides set before the first access.

### 3. Regenerate docs

```bash
make generate-env-vars-docs
```

This rewrites `docs/environment-variables.md` from the source of truth (the Pydantic Field metadata). Pre-commit's `generate-env-vars-docs` hook re-runs it on commit; running manually avoids the heredoc-reflow trap in `aiperf-commit`.

### 4. Verify the env var works

```bash
AIPERF_HTTP_YOUR_TUNABLE=5000 python -c "from aiperf.common.environment import Environment; print(Environment.HTTP.YOUR_TUNABLE)"
# Should print: 5000
```

The env var naming is `AIPERF_<SECTION>_<FIELD_NAME>` (uppercase, underscores). Where `<SECTION>` matches the `_XxxSettings` class (HTTP, WORKER, etc.).

### 5. Update docs/cross-reference

`make generate-env-vars-docs` handles `docs/environment-variables.md` automatically. If your tunable is referenced in tutorials or in `docs/architecture.md`, update those by hand.

## Constraints / validation

The Pydantic `Field` is the right place to enforce constraints:
- Numeric ranges: `ge=`, `le=`, `gt=`, `lt=`.
- Enum: `Literal["a", "b", "c"]` or a real Enum.
- Required-non-default: omit `default=`; the env var becomes mandatory.

Constraint failures surface at startup, not at use, with a clear Pydantic error — much better than `AssertionError` mid-run.

## Red flags — STOP, you're rationalizing

| Thought | Reality |
|---|---|
| "I'll just add `TIMEOUT_S = 30` at the top of my service file" | Project rule: every tunable goes in `environment.py`. Module/class constants in service code get rejected on review. |
| "Skip `Field(description=...)`, it's a stub" | `Field(description=...)` is required on every field per coding standards. `make generate-env-vars-docs` uses it. |
| "I'll skip `make generate-env-vars-docs`, pre-commit will run it" | When pre-commit rewrites the docs mid-commit, the heredoc message is lost. Regenerate manually first. |
| "I'll create a new `_FooSettings` class for this one tunable" | Adds taxonomy noise. Use the closest existing section unless the section is genuinely new. |
| "I'll cache `Environment.HTTP.X` at module import time" | Lazy resolution is the point — caching defeats env-var overrides set after import. Read it at the call site every time (it's a Pydantic property, near-zero cost). |
| "The env var name doesn't need to match the Field name exactly" | It does. Pydantic's `BaseSettings` derives the env var name from the field. Renaming one without the other silently breaks tuning. |

## Common mistakes

- **Missing `Field(description=...)`** — `make generate-env-vars-docs` falls back to a blank docs entry; pre-commit will still pass but the docs land bare.
- **Putting the Field in the wrong section** — works mechanically, but breaks the section-grouping convention and makes the env var harder to find.
- **Using `Optional[int]` or `Union[int, None]`** — project convention is `int | None`. No `Optional`/`Union` shorthand.
- **Defaulting to a sentinel value (`-1`, `""`)** instead of `None` — Pydantic handles `None` cleanly; sentinels require everyone to know the convention.

## Composition

- `aiperf-commit` for the staging step (the env-vars-docs regen is a known reflow trigger).
- `aiperf-add-metric` if the tunable controls a metric's behavior — both skills compose.
