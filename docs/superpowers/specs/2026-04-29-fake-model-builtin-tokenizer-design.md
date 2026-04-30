# Fake-Model-Name → Builtin-Tokenizer Fallback

**Date:** 2026-04-29
**Status:** Design
**Author:** acasagrande@nvidia.com

## Motivation

LLMs frequently hallucinate placeholder values for `--model` when generating
AIPerf invocations from natural-language prompts: `mock-model`, `test-model`,
`fake-model`, `my-model`, `your-model`, etc. With the current behavior, the
tokenizer name defaults to `--model` when `--tokenizer` is unset, so AIPerf
attempts a HuggingFace Hub alias lookup on the placeholder string. The lookup
fails (or worse, ambiguously matches an unrelated repo), and the user sees a
confusing tokenizer error rather than a benchmark run.

These are not "real" failures — the user is iterating against a mock or test
server and just wants token counts. The right behavior is to detect the
hallucinated name, substitute the `builtin` (tiktoken) tokenizer, and emit
a single warning so the user knows what happened.

## Trigger

Detection runs inside `validate_tokenizer_early()`
(`src/aiperf/common/tokenizer_validator.py`) — the existing single chokepoint
where model names are turned into tokenizer names.

It runs **only when `tokenizer_cfg.name is None`** (i.e. the user did not pass
`--tokenizer`). An explicit `--tokenizer <X>` is always authoritative, even
if `<X>` itself looks like a placeholder.

It runs **after** the existing `BUILTIN_TOKENIZER_NAME` / tiktoken short-circuit
and **before** `_resolve_aliases()`.

## Detection rule

New helper in a small dedicated module
``src/aiperf/common/tokenizer_fake_names.py`` (own file to keep
``tokenizer.py`` under the 500-line ergonomics ceiling):

```python
def is_fake_model_name(name: str) -> bool:
    """Return True if *name* looks like an LLM-hallucinated placeholder."""
```

Algorithm:

1. **Reject path-like input.** If `name` contains `/` or `\`, or starts with
   `.` or `~`, return `False`. (Catches HF `org/repo`, absolute paths, and
   relative paths without a filesystem syscall.)
2. **Normalize.** Lowercase and replace `_` with `-`.
3. **Exact-match set.** Return `True` if the normalized name is in:
   `{"test", "mock", "fake", "dummy", "example", "sample", "placeholder"}`.
4. **Substring set.** Return `True` if the normalized name contains any of:
   `"mock-"`, `"-mock"`, `"fake-"`, `"-fake"`, `"test-model"`, `"-test-model"`,
   `"your-model"`, `"my-model"`, `"model-name"`, `"model-id"`.
5. Otherwise `False`.

Constants live in module scope alongside `BUILTIN_TOKENIZER_NAME` so the rule
is greppable and one-stop. `is_fake_model_name` is added to `__all__`.

### Examples

| Name | Result | Reason |
|---|---|---|
| `mock-model` | `True` | substring `mock-` |
| `test-model` | `True` | exact substring |
| `fake-llama` | `True` | substring `fake-` |
| `my-model` | `True` | substring `my-model` |
| `MOCK_MODEL` | `True` | normalize → `mock-model` |
| `Test-Model-v2` | `True` | substring `test-model` |
| `placeholder` | `True` | exact match |
| `meta-llama/Llama-3-test-finetune` | `False` | contains `/` |
| `./mock-model` | `False` | starts with `.` |
| `gpt2` | `False` | no match |
| `Qwen/Qwen3-0.6B` | `False` | contains `/` |
| `Llama-3-test-finetune` | `True` | substring `test-` … see Risks below |

## Substitution behavior

In `validate_tokenizer_early()`, after the early returns and the
existing `BUILTIN_TOKENIZER_NAME` / tiktoken short-circuit:

1. Partition `model_names` into `fake_models` and `real_models` via
   `is_fake_model_name`.
2. For each name in `fake_models`, log one `WARNING`-level line:
   ```
   Model name 'mock-llama' looks like a placeholder; defaulting tokenizer to
   'builtin' (tiktoken o200k_base). Pass --tokenizer <name> to override.
   ```
3. If `real_models` is empty: return
   `{model: BUILTIN_TOKENIZER_NAME for model in model_names}` and skip alias
   resolution and prefetch entirely (no Hub calls, no subprocess pool).
4. If `real_models` is non-empty: pass only `real_models` into the existing
   `_resolve_aliases()` + `_prefetch_tokenizers()` path. Merge the result:
   fake names map to `BUILTIN_TOKENIZER_NAME`, real names to their resolved
   canonical IDs. Return the merged dict.

The `tokenizer_cfg.name is not None` branch (user passed `--tokenizer X`) is
untouched — that path already returns `{model: resolved[X]}` for every model.

## Risks and trade-offs

- **False positives on real model names containing `test`/`mock`/etc.** The
  substring rules use compound markers (`mock-`, `-mock`, `test-model`, …)
  rather than bare `test`/`mock` to keep the false-positive surface small.
  A name like `Llama-3-test-finetune` will still match (`test-` substring).
  Documented escape hatch: pass `--tokenizer <real-tokenizer>` explicitly,
  which bypasses the check entirely.
- **`/` is the strongest signal of a real HF repo.** Anyone running a real
  custom HF model uses `org/repo` form, so the path-like check by itself
  prevents most legitimate cases from being miscategorized.
- **No env-var tuning.** The pattern set is hardcoded. If real-world traffic
  shows we need to grow it, edit the constants. We explicitly do not add a
  CLI flag or env var for this — the override is `--tokenizer`.

## Files touched

| File | Change |
|---|---|
| `src/aiperf/common/tokenizer_fake_names.py` | New module: `is_fake_model_name()` + private constant sets. |
| `src/aiperf/common/tokenizer_validator.py` | New `_partition_fake_models()` helper; `validate_tokenizer_early` calls it when `--tokenizer` is unset, logs warning per fake, skips prefetch when all fake, mixed-merge otherwise. |
| `src/aiperf/config/v1/_tokenizer.py` | Append one sentence to `--tokenizer` description: "If `--tokenizer` is not set and the model name looks like an obvious placeholder (e.g. `mock-model`, `test-model`), AIPerf substitutes `builtin` automatically." |
| `docs/reference/tokenizer-auto-detection.md` | New section "Placeholder Model Name Detection" between "Built-in Tokenizer" and "Automatic Cache Detection": describes trigger conditions, full pattern list, sample warning output, and the explicit-`--tokenizer` opt-out. |
| `tests/unit/common/test_tokenizer.py` | `TestIsFakeModelName` parametrized positives + negatives. |
| `tests/unit/common/test_tokenizer_validator.py` | `TestValidatorFakeModelFallback` covering all-fake, mixed, and explicit-tokenizer-overrides. |

No changes to `docs/index.yml` (modifying an existing tracked file).
No changes to auto-generated `docs/cli-options.md` (description regenerates).

## Out of scope

- **Endpoint-side handling of fake model names.** The fake name is still sent
  to the inference server as the user typed it — mock servers happily echo
  `mock-llama` in responses. This change touches only tokenizer resolution.
- **Did-you-mean suggestions.** The existing ambiguous-name panel already
  handles real typos against the HF index. Placeholder strings won't have
  meaningful suggestions, so the warning is intentionally terse.
- **Integration with `--tokenizer builtin` warning suppression.** No
  warning is emitted when the user explicitly passes `--tokenizer builtin`
  — that path is unchanged.
- **Telemetry / metrics on how often this fires.** Could be added later if
  we want to study LLM-generated invocations; not part of this change.

## Test plan

- Unit: `is_fake_model_name` parametrized on the table in the Detection
  Rule section (positives + negatives).
- Unit: `validate_tokenizer_early` with `model_names=["mock-llama"]` and no
  `--tokenizer` returns `{"mock-llama": "builtin"}` and never calls
  `_resolve_aliases`.
- Unit: `validate_tokenizer_early` with `model_names=["mock-llama", "Qwen/Qwen3-0.6B"]`
  returns `{"mock-llama": "builtin", "Qwen/Qwen3-0.6B": <resolved>}` and
  passes only `{"Qwen/Qwen3-0.6B"}` into `_resolve_aliases`.
- Unit: `validate_tokenizer_early` with explicit `tokenizer_cfg.name="mock-model"`
  does **not** trigger the fallback (explicit user choice wins).
- Manual smoke: `aiperf profile --model mock-llama --endpoint-type chat …`
  against a mock server completes with token counts and emits exactly one
  warning line for the placeholder.
