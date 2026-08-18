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

## CLI flag routing under `--config`

`resolve_config` merges a YAML config file with explicitly-set CLI
flags. Historically it only knew how to route a subset of `CLIConfig`
into the merged dict, and **every other flag was discarded without a
word** — `aiperf profile -f base.yaml --random-seed 42` ran with a
different seed than the user asked for and said nothing. For a
benchmarking tool that silently corrupts published numbers.

The invariant now in force:

> A CLI flag passed alongside `--config` either changes the resolved
> config, or raises an error naming the flag. It is never silently
> ignored.

### `CLIConfig` is the source of truth

The guarantee is scoped to what `resolve_config` can see, and its
signature is the boundary:

```python
def resolve_config(cli_config: CLIConfig, config_file: Path | None = None) -> AIPerfConfig
```

Both entry points — `aiperf profile` and `aiperf service` — call it with
nothing but their `CLIConfig` and a path. So a CLI option can influence
the resolved `AIPerfConfig` **only** by being a field on `CLIConfig`,
and that is a structural property of the call, not a convention someone
has to remember. It is what lets the classification test enumerate
`CLIConfig.model_fields` and claim completeness: there is no second
source of flags for it to miss.

Two consequences worth stating plainly:

- **Command-level parameters are out of scope, by construction.**
  `aiperf service` declares `--service-id`, `--health-host`,
  `--health-port` and its `service_type` argument directly on the
  command function rather than on `CLIConfig`. They are service-instance
  plumbing, never passed to `resolve_config`, and so cannot be dropped
  by this mechanism — they never enter it. Anything that *is* benchmark
  configuration belongs on `CLIConfig`, precisely so this guarantee
  covers it.
- **The assumption is falsifiable, and this is how it would break.**
  Giving `resolve_config` another parameter that carries user intent, or
  having it read flags from module state or the environment, puts values
  into the resolved config that no test enumerates. If you need to do
  that, extend the classification to cover the new source in the same
  breath — otherwise the suite will keep reporting a completeness it no
  longer has.

Every field on `CLIConfig` must be classified in
[`_config_flag_routing.py`](https://github.com/ai-dynamo/aiperf/tree/main/src/aiperf/config/flags/_config_flag_routing.py):

| Set | Meaning |
| --- | --- |
| `ROUTED_UNDER_CONFIG` | Reaches `AIPerfConfig`. Derived from the resolver's own routing tables where possible, so it cannot drift from them. |
| `UNROUTED_UNDER_CONFIG` | Known not to route. Raises `ConfigurationError` naming the flag. |
| `EXEMPT_FROM_CONFIG_ROUTING` | Not benchmark config at all (`--config` itself). Each entry needs a stated reason. |
| `COMPANION_ROUTED` | Routes only alongside another flag (`--model-selection-strategy` needs `--model-names`). Rejected when the companion is absent. |
| `MAGIC_LIST_ONLY_UNDER_CONFIG` | Routes in list form only (`--isl 128 256` becomes a sweep parameter; scalar `--isl 128` goes to the dataset). Decided per value. |

### Why the guarantee holds

Four layers, each covering the previous one's blind spot:

1. **The universe is derived, not maintained.** Every set is
   subtracted from `frozenset(CLIConfig.model_fields)`. You cannot add
   a CLI option without adding a field, so nothing is checked against
   a list someone has to remember to update.
2. **Runtime is default-deny.** `reject_unrouted_cli_flags` computes
   `model_fields_set - ROUTED - EXEMPT` and raises on the remainder —
   "is this known-good?", not "is this known-bad?". It gates on
   `model_fields_set`, never truthiness, so `--prompt-batch-size 0`
   counts as set.
3. **Classification is mandatory.**
   `test_every_cli_config_field_is_classified` fails when a field
   belongs to none of the sets, so a new flag fails CI at authoring
   time.
4. **The classification is verified, not trusted.** Layer 3 only
   proves someone applied a label.
   `test_routed_field_never_silently_no_ops` drives every routed field
   against both a synthetic and a file dataset and asserts it changes
   the config or raises. This is what catches whole-section
   classifications (`OUTPUT`/`TOKENIZER`/`ACCURACY`/`SWEEPING` are
   marked routed wholesale, so a new member is auto-classified and
   layer 3 passes vacuously).

Two further invariants guard the merge itself:
`test_override_emits_no_key_the_user_did_not_set` runs `build_dataset`
twice with different values for one field — any emitted key identical
across both runs is a materialized default that would overwrite a YAML
value the user never mentioned. `test_routed_dataset_field_actually_changes_the_resolved_config`
covers the dataset block specifically.

### Adding a new CLI flag

1. Add the field to `CLIConfig` as usual. CI now fails with
   `CLI fields are unclassified for the --config path: [...]`.
2. Decide what the flag should do under `--config`:
   - **Route it** (preferred). Dataset-shaping fields flow through
     `build_dataset` automatically. Otherwise wire it into
     `build_cli_overrides` — check first whether a builder already
     exists, as `build_mlflow`/`build_otel`/`build_network_latency`
     did while going uncalled.
   - **Or list it in `UNROUTED_UNDER_CONFIG`**, so users get an error
     naming the flag rather than a wrong benchmark.
3. If the invariant suite cannot generate a value from the
   annotation (`typing.Any`, free-form strings), add one to
   `FIELD_PROBE_VALUES` or list the field in `UNDRIVABLE_FIELDS` with
   a reason. Do not let it report as skipped — a skip is invisible in
   CI, which is exactly how a coverage hole hides.

### Known gaps

`--sweep-type` and `--disable-auto-fixed-schedule` are unrouted and
loud: the first needs a sweep block to attach to, the second is
consumed during phase construction, which this path does not rebuild.

The flags in `SWEEP_FIELDS_NOT_ROUTED` (`--concurrency-min/max/steps`,
`--isl-*`/`--osl-*`, the `*-sla-ms` filters, `--parameter-sweep-*`)
resolve cleanly and change nothing, so they are rejected. Verified
individually — notably `--ttft-sla-ms` does **not** take effect even
alongside `--search-recipe` and `--streaming`. Routing them is
follow-up work; until then the failure is loud.

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
