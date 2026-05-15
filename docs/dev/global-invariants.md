<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Global Property-Test Invariants

The suite under [`tests/unit/property/`](https://github.com/ai-dynamo/aiperf/tree/main/tests/unit/property)
holds AIPerf's **mechanical CI gates**: tests that walk the source tree
or fuzz Pydantic models to enforce cross-cutting contracts a future PR
cannot accidentally regress. They are the canonical home for "no new
debt allowed" rules — pair every cross-cutting fix (NaN/inf, missing
bounds, validator crashes) with an invariant test here so the next
contributor cannot reintroduce the bug.

This page documents the invariants in force, the baseline-ratchet
pattern, and how to extend the suite when a new cross-cutting rule
needs enforcement.

## Why mechanical, not example-based

Example-based unit tests catch a specific bug. Mechanical invariants
catch the **class** of bug. Three concrete cases motivated this suite:

1. **NaN/inf leakage** — a single `float` field that forgot
   `FiniteFloat` silently let NaN flow through every exporter, and
   `orjson.dumps` coerced it to JSON `null` indistinguishable from
   "absent". Auditing every field by hand was untenable.
2. **Missing numeric bounds** — config fields without `ge`/`le`
   accepted negative concurrencies, zero-length distributions, and
   `-inf` percentiles, all of which crashed deep inside numpy. Adding
   bounds one-at-a-time worked, but new fields kept shipping without
   them.
3. **Validator crashes on adversarial input** — Pydantic validators
   were leaking `AttributeError`/`KeyError`/`TypeError` instead of
   clean `ValidationError`s when fed malformed-but-typed input,
   breaking error-message UX and obscuring real bugs.

Each invariant test below converts one of these classes into an
AST-walk or hypothesis-fuzz that fails CI on the first regression.

## The invariants in force

All three currently-enforced invariants live in
[`tests/unit/property/test_finite_invariants.py`](https://github.com/ai-dynamo/aiperf/tree/main/tests/unit/property/test_finite_invariants.py)
plus the fuzzer in
[`test_pydantic_field_fuzz.py`](https://github.com/ai-dynamo/aiperf/tree/main/tests/unit/property/test_pydantic_field_fuzz.py)
and the round-trip in
[`test_dump_config_roundtrip.py`](https://github.com/ai-dynamo/aiperf/tree/main/tests/unit/property/test_dump_config_roundtrip.py).

### `test_every_json_exporter_calls_scrub_non_finite`

Walks every `.py` file under `src/aiperf/exporters/` and
`src/aiperf/server_metrics/`. If a file imports `orjson` and calls
`orjson.dumps`, it must also import `scrub_non_finite` from
`aiperf.common.finite` — or be listed in `ORJSON_SCRUB_WHITELIST`
with a documented reason (e.g. "metadata-only — no metric values").

**To add a new exporter**: import `scrub_non_finite` and apply it to
the payload immediately before `orjson.dumps`:

```python
from aiperf.common.finite import scrub_non_finite

out_path.write_bytes(orjson.dumps(scrub_non_finite(payload)))
```

If the exporter genuinely does not handle metric values (e.g. dumps
only configuration metadata), add an entry to `ORJSON_SCRUB_WHITELIST`
in the test module with a one-line reason. Anonymous whitelisting is
rejected at review.

### `test_every_metric_field_is_finite_or_optional`

Imports every Pydantic model under `src/aiperf/`, inspects each
`float`/`float | None` field, and checks whether the field name
matches a metric-suggestive pattern (`*_p99`, `*_mean`, `latency_*`,
`ttft_*`, `itl_*`, `throughput_*`, ...). Metric-named fields must be
annotated `FiniteFloat` or `FiniteFloat | None`.

**Existing debt** is captured in
[`_metric_field_baseline.txt`](https://github.com/ai-dynamo/aiperf/tree/main/tests/unit/property/_metric_field_baseline.txt)
as `Module.ClassName.field_name` lines. The baseline is a one-way
ratchet — fields can leave (when fixed) but new fields cannot enter
without an explicit code-review carve-out.

### `test_every_numeric_field_has_bounds`

For every numeric Pydantic field on every model under
`src/aiperf/`, the test requires at least one of: `ge`, `gt`, `le`,
`lt`, the `FiniteFloat` type, or a custom `AfterValidator`. Raw `int`
or `float` fields with no constraint are rejected.

**Existing debt** lives in
[`_numeric_bounds_baseline.txt`](https://github.com/ai-dynamo/aiperf/tree/main/tests/unit/property/_numeric_bounds_baseline.txt)
(currently ~390 entries). Same ratchet rules apply.

### `test_dump_config_roundtrip` (parametrized over all bundled templates)

For each YAML template under `src/aiperf/config/templates/`, calls
`load_config_from_string(dump_config(load_config(path)))` and asserts
the re-loaded config is structurally equal to the original. Catches:

- Field aliases that don't survive round-trip.
- `BeforeValidator`/`AfterValidator` chains that mutate on load but
  not on dump (or vice versa).
- `mode="json"` serialization that drops or coerces fields.
- Sweep-envelope keys that `model_dump` flattens incorrectly.

Adding a new template under `src/aiperf/config/templates/`
automatically extends coverage — no test change needed.

### Hypothesis fuzz: `test_pydantic_field_fuzz.py`

Property: every targeted Pydantic model either validates cleanly OR
raises a clean `pydantic.ValidationError` /
`aiperf.common.exceptions.ConfigurationError` / `ValueError`. **Any
other** exception type (`AttributeError`, `TypeError`, `KeyError`,
`RecursionError`, `IndexError`) means a validator crashed on
adversarial input rather than rejecting it cleanly.

Adversarial input strategies live in
[`_strategies.py`](https://github.com/ai-dynamo/aiperf/tree/main/tests/unit/property/_strategies.py)
and intentionally include NaN, +/-inf, very-large/small floats, empty
and very-long strings, control characters, negative ints, dotted-path
nonsense, and unhashable choices.

Currently fuzzes 19 models including `SamplingDimension`,
`SearchSpaceDimension`, `SLAFilter`, `AdaptiveObjective`, the currently targeted sweep
envelope variants (`GridSweep`, `ScenarioSweep`, `SobolSweep`,
`LatinHypercubeSweep`, `AdaptiveSearchSweep`; `ZipSweep` is not yet
fuzzed), all distribution types
(`FixedDistribution`, `NormalDistribution`, `LogNormalDistribution`,
`MultimodalDistribution`, `EmpiricalDistribution`), and
`CLIConfig`.

## The baseline-ratchet pattern

Two text files act as one-way debt counters:

- `tests/unit/property/_metric_field_baseline.txt`
- `tests/unit/property/_numeric_bounds_baseline.txt`

Both are auto-loaded by their respective tests and treated as a
"grandfathered" allowlist. The ratchet rule: **entries can leave the
file (when the field is fixed), but new entries cannot be added
without explicit reviewer sign-off.** A regular PR that touches the
baseline to add a field will be flagged in review as taking on new
debt, not fixing existing debt.

When you fix a field, just delete its line from the baseline; the
test starts enforcing the constraint on that field on the next CI
run.

## Extending the suite

### Adding a new mechanical invariant

1. Decide the contract (e.g. "every service handler decorated with
   `@on_message` must declare a `MessageType`").
2. Add a test in `test_finite_invariants.py` (for AST/import-walk
   invariants) or a new `test_<area>_invariants.py` file (for
   higher-level contracts).
3. If the codebase has existing violations, create
   `_<area>_baseline.txt` and load it in the test as a one-way
   ratchet. Document the ratchet rule in this page.
4. Update the project rule files (`AGENTS.md`, `CLAUDE.md`,
   `.github/copilot-instructions.md`, `.cursor/rules/python.mdc`) to
   reference the new invariant under the relevant Coding Standards
   subsection.

### Fuzzing a new Pydantic model

1. Add a `<model>_inputs() -> st.SearchStrategy[dict]` strategy in
   `_strategies.py`. Compose primitive `adversarial_*` strategies
   from the same file — do not write a "happy-path-only" strategy.
2. Add a `test_<model>_never_unhandled` test in
   `test_pydantic_field_fuzz.py` calling `_check_no_unhandled(Model,
   data)`.
3. Run `uv run pytest tests/unit/property/ -n auto`. If the test
   surfaces a real bug (validator crash on adversarial input), fix
   the validator — don't relax the test.

### Adding a new YAML template

The `test_dump_config_roundtrip` parametrization auto-discovers
`src/aiperf/config/templates/**/*.yaml`. If your template intentionally cannot
round-trip (e.g. depends on runtime context), skip it via the
existing skip-decorator with a documented reason.

## Running locally

```bash
uv run pytest tests/unit/property/ -n auto
```

Runs in seconds; no external services required. The fuzz tests use
bounded `max_examples` so CI flakes are zero.

## Related docs

- [`patterns.md`](patterns.md) — the NaN/Inf Discipline Pattern
  explains the runtime primitives (`FiniteFloat`, `scrub_non_finite`,
  `is_finite_value`, `nan_safe_mean`/`nan_safe_std`) that the
  invariants enforce.
- `src/aiperf/common/finite.py` — module-level docstring with the
  three failure modes that motivate the discipline.
