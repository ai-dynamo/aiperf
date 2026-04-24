# `aiperf kube show` — Render AIPerfJob CR Design

**Date:** 2026-04-24
**Status:** Approved (brainstorm)

## Purpose

Add `aiperf kube show --path <file>` — a read-only command that reads an
AIPerfJob CR YAML, renders Jinja2 templates and `${ENV_VAR}` substitutions in
`spec.benchmark`, validates the result against `AIPerfConfig`, and prints the
full CR (`apiVersion`, `kind`, `metadata`, `spec.*`) back as YAML with all
templates resolved.

## Problem

Two existing commands each cover half of what a recipe author wants to see:

- **`aiperf config show --path <file>`** expands Jinja2, env vars, and
  defaults — but operates on a plain benchmark YAML, not an AIPerfJob CR.
- **`aiperf kube profile --config <cr> --image <img> --dry-run`** emits the CR
  as JSON, but leaves `{{ ... }}` strings unrendered (the operator renders
  them at apply time).

Recipe authors currently have to:
```bash
yq '.spec.benchmark' recipe.yaml > /tmp/bench.yaml
aiperf config show --path /tmp/bench.yaml
```
— which drops `metadata`, `spec.image`, `spec.podTemplate`, etc. from the
output.

## Design

### Command surface

```
aiperf kube show --path <file>
```

- Single required flag: `--path`.
- Output: YAML only, to stdout. No `--format` flag (JSON is not useful for a
  CR that will be `kubectl apply`'d).
- One path per invocation (matches `aiperf config show`).
- v1 does **not** add `--strict`, `--no-interpolate`, multi-file, or stdin
  support. These can be added later if a real need appears.

### What renders

- **Rendered:** everything under `spec.benchmark`. Jinja2 `{{ ... }}`
  expressions, `${ENV_VAR}` / `${ENV_VAR:default}` substitutions, and the
  `variables:` block (stripped after use).
- **Passed through untouched:** `apiVersion`, `kind`, `metadata`, and every
  other key under `spec` (`image`, `imagePullPolicy`, `resourceMode`,
  `connectionsPerWorker`, `timeoutSeconds`, `ttlSecondsAfterFinished`,
  `cancel`, `podTemplate`, `scheduling`, etc.).
- **Not injected:** operator-side Kubernetes runtime config (ZMQ hosts,
  `dataset_api_base_url`, `service_run_type: kubernetes`). The output reflects
  what's in the CR, not what the operator adds at apply time.

### Rendering pipeline

1. `yaml.safe_load` the input file.
2. Validate structure via the existing
   `aiperf.kubernetes.validate.validate_yaml_structure` helper:
   `kind == "AIPerfJob"` and `spec.benchmark` is a dict.
3. Call `aiperf.operator.spec_converter.extract_benchmark_config(spec)`. Per
   its docstring, this runs `expand_config_dict` (env vars + Jinja2) and
   `AIPerfConfig.model_validate`, and deliberately **does not** apply K8s
   runtime injection — exactly what we need.
4. Serialize the validated config via
   `aiperf.config.dump_config(config, exclude_defaults=True, exclude_none=True)`.
   Same parameters `aiperf config show` uses, so the output looks like the
   user's input with templates resolved, not bloated with every default.
5. Re-parse step (4)'s YAML, assign it to `doc["spec"]["benchmark"]`, and
   `yaml.safe_dump(doc, sort_keys=False)` the whole document.
6. Write to stdout.

### Error handling

All errors exit 1 with a clear single-line message to stderr:

- Missing file / not a file / YAML parse error — matches `validate.py`'s
  existing messages.
- `kind != "AIPerfJob"` — `"Not an AIPerfJob manifest: kind='{kind}'"`.
- Missing `spec.benchmark` —
  `"spec.benchmark: required — nothing to render"`.
- Pydantic validation failure — re-raise the pydantic error; it's already
  human-readable and points at the offending field.

### Files touched

- **New:** `src/aiperf/cli_commands/kube/show.py` (~60 lines including
  docstring and imports). Modeled on
  `src/aiperf/cli_commands/kube/validate.py` and the `show_config` function
  in `src/aiperf/cli_commands/config_cli.py`.
- **Modified:** `src/aiperf/cli_commands/kube/_app.py` — register the new
  `show` subcommand next to `validate`.
- **New:** `tests/unit/cli_commands/kube/test_show.py` — see Testing.
- **Modified:** `docs/cli-options.md` — regenerated via
  `make generate-cli-docs` (pre-commit hook will also do this).

No docs file needs to be hand-written; the command is discoverable via
`aiperf kube --help` and self-documents via its own `--help`.

## Testing

Single test file `tests/unit/cli_commands/kube/test_show.py` with the
following cases:

1. **Renders Jinja2 templates.** Input CR has
   `variables: {concurrency_per_gpu: 2, deployment_gpu_count: 16}` and a phase
   with `concurrency: "{{ concurrency_per_gpu * deployment_gpu_count }}"`.
   Assert output YAML parses as a dict, `spec.benchmark.phases.<name>.concurrency`
   equals `32` as an int, and `variables:` key is absent from output.

2. **Resolves env var with default.** Input uses `${FOO:ninety-nine}` on a
   string field; `FOO` is unset. Assert that field in the output equals
   `"ninety-nine"`.

3. **Passes through non-benchmark spec fields.** Input has
   `spec.image`, `spec.podTemplate`, `spec.connectionsPerWorker`, and
   `metadata.name`. Assert all of these appear verbatim in the output.

4. **Missing file exits non-zero.** Call the command with a path that doesn't
   exist; assert non-zero exit and a message referencing the path.

5. **Wrong kind exits non-zero.** Input is a valid YAML doc with
   `kind: Pod`. Assert non-zero exit and a message naming the kind.

6. **Invalid benchmark exits non-zero.** Input has `profiling` as the first
   phase with `seamless: true` (the same invariant that broke our earlier
   `config show` attempt). Assert non-zero exit; assert the pydantic message
   appears in output.

All tests use `click.testing.CliRunner` or the project's existing typer test
helper (whichever `tests/unit/cli_commands/kube/` already uses — to be
confirmed during implementation).

## Out of scope for v1

Deliberately excluded; can be added later if the use case arrives:

- JSON output (`--format json`).
- Multi-file / glob support.
- Stdin input (`--path -`).
- `--strict` mode (error on unknown `spec.benchmark` fields pre-expansion).
- `--no-interpolate` (skip env var substitution, matching `config show`).
- Hooking K8s runtime injection behind an opt-in flag.
