# `artifacts.user_files` — User-Defined Templated Output Files

**Status:** approved
**Date:** 2026-04-24
**Branch target:** `ajc/k8s` (or successor)
**Driver:** parity with the upstream dynamo recipe's `input_config.json` sidecar, generalized into a reusable, schema-driven mechanism for any user-declared metadata or summary file.

---

## 1. Problem

The upstream dynamo recipe (`recipes/deepseek-r1/trtllm/disagg/wide_ep/gb200/perf.yaml` in
`aiperf-dynamo/dynamo`) is a `batch/v1 Job` that pip-installs `aiperf==0.6.0` and shells
out to `aiperf profile`. Among other things, it writes an `input_config.json` sidecar
to the run directory holding deployment-shape metadata (`gpu_count`, `concurrency_per_gpu`,
`isl`, `osl`, `endpoint`, `model`) which downstream tooling consumes.

The translated `AIPerfJob` form on this branch has no equivalent. A previous round
proposed hardcoding a `compatibility.writeDynamoInputConfig: true` flag, but that
bakes the dynamo schema into the operator. A user-defined templated-files mechanism
is more general, more honest about what's happening, and unlocks future use cases
(custom run notes, deployment manifests, downstream-tool manifests) at zero marginal cost.

## 2. Goals

- A user can declare arbitrary output files in their AIPerf config that materialize
  into the run directory before the benchmark begins.
- File contents are jinja-templated using `variables:` plus a stable, documented set of
  system-injected names.
- Works identically for `aiperf profile` (local) and the Kube controller pod, because
  both load the same `ArtifactsConfig`.
- Failure modes are loud and early: bad path, missing variable, or write error aborts
  the run before any benchmark service starts.

## 3. Non-goals (v1)

- Post-run files that see results-of-the-run (`results.*` namespace).
- Per-phase / per-concurrency files.
- Per-file `required: false` opt-out.
- Per-file `overwrite: false` safety net (writes are unconditional clobbers).
- A `aiperf config validate` flag that pre-renders user_files with a stub context.
- A "recipe registry" / `--recipe` shortcut on the kube CLI (separate concern).

All five are non-breaking future extensions.

## 4. Decisions captured during brainstorm

| # | Decision | Choice |
|---|---|---|
| 1 | Where does the field live | **In `ArtifactsConfig`** — both local CLI and Kube get it via the shared config block |
| 2 | `content` shape | **Format-driven**: structured (dict/list/scalar) for `format: json|yaml`, string for `format: text`. Format inferred from content type when omitted. |
| 3 | Templating context | **`variables:` + small documented injected set** (`epoch`, `job_name`, `namespace`, `model`, `endpoint_url`, `artifact_dir`) |
| 4 | Strictness for undefined variables | **Tighten globally** — switch the existing renderer to `jinja2.StrictUndefined`. Audit existing templates as part of this spec. |
| 5 | Path semantics | **Run dir + subdirs**, relative only, `..`/absolute rejected, validated at config load AND at write time |
| 6 | Existing-file behavior | **Overwrite by default**, no opt-in flag |
| 7 | Lifecycle | **Pre-run only** (before warmup) |
| 8 | Field name | **`artifacts.user_files`** |
| 9 | Failure mode | **Fail fast** — abort the run on first error, no per-file `required` |

## 5. Schema

### 5.1 Config YAML

```yaml
artifacts:
  user_files:
    - path: input_config.json
      format: json                 # optional; inferred from content type when omitted
      content:
        gpu_count: "{{ deployment_gpu_count }}"
        concurrency_per_gpu: "{{ concurrency_per_gpu }}"
        total_concurrency: "{{ concurrency_per_gpu * deployment_gpu_count }}"
        mode: "{{ deployment_mode | default('disagg') }}"
        endpoint: "{{ endpoint_url }}"
        model: "{{ model }}"

    - path: meta/notes.md
      content: |                   # string content → format defaults to "text"
        Run {{ job_name }} started at {{ epoch }}.
        Targeting {{ model }} @ {{ endpoint_url }}.
```

### 5.2 Pydantic model

New module: `src/aiperf/config/user_files.py`

```python
class UserFile(BaseConfig):
    """One user-declared output file rendered into the run directory before benchmark start."""

    path: Annotated[
        str,
        Field(
            description=(
                "Output path relative to the run directory. Subdirectories allowed. "
                "Absolute paths and any segment equal to '..' are rejected."
            ),
        ),
    ]

    format: Annotated[
        Literal["json", "yaml", "text"] | None,
        Field(
            default=None,
            description=(
                "Serialization format. If omitted: 'text' when content is a string, "
                "'json' otherwise."
            ),
        ),
    ] = None

    content: Annotated[
        Any,
        Field(
            description=(
                "Templated value. Structured (dict/list/scalar) for json/yaml; string for text. "
                "Jinja2 expressions in any string leaf are rendered with the user_files context "
                "(see docs/kubernetes/user-files.md for available names)."
            ),
        ),
    ]

    @model_validator(mode="after")
    def _validate_path_and_format(self) -> "UserFile": ...
```

Path validation at load time:
- Reject absolute paths (`Path(self.path).is_absolute()`).
- Reject any path segment equal to `".."`.
- Reject null bytes and ASCII control chars (defensive against weird YAML input).
- Reject empty path.

Format inference:
- `format is None` and `isinstance(content, str)` → `format = "text"`.
- `format is None` and not str → `format = "json"`.
- `format in {"json", "yaml"}` and `isinstance(content, str)` → validation error
  ("structured content required for {format}; wrap in a dict or set format: text").
- `format == "text"` and not isinstance(content, str) → validation error
  ("text content must be a string; got {type}").

Added to `ArtifactsConfig` (in `src/aiperf/config/artifacts.py`):

```python
user_files: Annotated[
    list[UserFile],
    Field(
        default_factory=list,
        description=(
            "User-defined templated files materialized into the run directory before "
            "the benchmark begins. See docs/kubernetes/user-files.md."
        ),
    ),
] = []
```

### 5.3 Preserving the `variables:` block

The current loader pops `variables` from the config dict at `loader/core.py:187` before
pydantic validation, so it does not survive on the resolved `AIPerfConfig` object.
For run-time templating we need the user's variables to be available at run start.

**Change:** stop popping. Add a `variables: dict[str, Any]` field to `AIPerfConfig`
(default `{}`, `Field(description="User-defined variables for jinja2 templating; ...")`)
and remove the `data.pop("variables", None)` line. The load-time render pass already
consumed `variables` for its own templating; the field then persists for any run-time
renderer (initially just `user_files`, but the surface is general).

This is a one-line removal plus one new field. Backwards-compatible: existing
configs without a `variables:` key get the default `{}`. Configs that already use
`variables:` get the same load-time behavior plus the new run-time availability.

### 5.4 Templating context

Context dict assembled by `build_user_file_context(config, run_meta) -> dict[str, Any]`:

```python
{
    # Tier 1: user-declared variables block (already resolved at config load)
    **config.variables,
    # Tier 2: system-injected names (run-time, stable API)
    "epoch":         run_meta.epoch,            # str, e.g. "1714000000"
    "job_name":      run_meta.job_name,         # str, AIPerfJob name in k8s; --artifact-dir basename locally
    "namespace":     run_meta.namespace,        # str, k8s namespace; "" locally
    "model":         config.benchmark.models[0],
    "endpoint_url":  config.benchmark.endpoint.urls[0],
    "artifact_dir":  str(run_dir),              # absolute path
}
```

Collisions: if a user `variables:` key shadows an injected name, the injected name wins
and a `WARNING` is logged at startup ("variable '{name}' shadowed by system-injected name;
rename to avoid").

Stable-API guarantee: adding new injected names is non-breaking. Renaming or removing
requires a deprecation cycle (deprecation log warning for one minor release, then removal).
The injected set MUST be documented in `docs/kubernetes/user-files.md` and any addition
MUST update that doc in the same PR.

## 6. Render lifecycle

User_files render at **run start**, not at config load. Two requirements drove this:

1. `epoch`, `job_name`, `namespace` are not known at config load time on the operator side.
2. `artifact_dir` is computed by `results_layout.run_dir(...)` which only runs when a
   benchmark actually begins.

Mechanism:

1. Add `"artifacts.user_files"` to `SKIP_TEMPLATE_FIELDS` in
   `src/aiperf/config/loader/jinja.py` so the existing load-time pass leaves
   `user_files.*.content` untouched.

2. New module `src/aiperf/config/user_files.py` exports two functions:

   ```python
   def build_user_file_context(
       config: AIPerfConfig,
       run_meta: RunMeta,
   ) -> dict[str, Any]: ...

   def materialize_user_files(
       files: list[UserFile],
       run_dir: Path,
       context: dict[str, Any],
   ) -> None:
       """Render and write all user_files. Raises UserFileError on first failure.

       Single pass: render → validate resolved path stays within run_dir → write.
       """
   ```

   `RunMeta` is a small dataclass living in the same module (`epoch: str`,
   `job_name: str`, `namespace: str`).

3. Call sites:
   - **Local CLI**: `cli_commands/profile.py`, immediately after the run dir is
     created and before any AIPerf service starts.
   - **Kube controller pod**: in the controller's entrypoint, after
     `results_layout.run_dir(...)` materializes the directory, before
     benchmark services boot.

   Both paths construct the same `RunMeta`; the local path uses
   `--artifact-dir` basename for `job_name` and an empty string for `namespace`.

4. Renderer uses a single `jinja2.Environment(undefined=jinja2.StrictUndefined,
   autoescape=False)` — autoescape is off because output formats include JSON/text
   where HTML escaping is incorrect.

## 7. Strictness change (in-scope)

The existing renderer at `src/aiperf/config/loader/jinja.py:_render_template_string`
calls bare `jinja2.Template(...)` which uses `Undefined` (lenient: missing variable
renders as empty string). Switching to `StrictUndefined` is in scope for this spec.

### 7.1 Implementation

Replace `jinja2.Template(data)` with a module-level
`jinja2.Environment(undefined=jinja2.StrictUndefined, autoescape=False)` whose
`from_string(data).render(**context)` is called instead.

### 7.2 Audit

Before merging, audit and fix any callers that relied on lenient empty-string rendering:

- `src/aiperf/config/templates/embeddings.yaml`
- `src/aiperf/config/templates/latency_test.yaml`
- `src/aiperf/config/templates/request_cancellation.yaml`
- `src/aiperf/config/templates/jinja2_variables.yaml`
- Any YAML in `tests/` or `tests/fixtures/` using `{{ ... }}`.
- Any in-tree configs in `dev/deploy/`, `dev/aiperf-runs/`, `recipes/`.

Audit method: grep for `{{` under those paths; for every match, confirm the variable
is resolvable from the config's `variables:` block (or the documented injected set
for `recipes/`).

### 7.3 Regression test

Add `tests/unit/config/loader/test_jinja_strict_undefined.py`:

- `{{ undefined_var }}` raises `ConfigurationError` whose message names
  `undefined_var` and the YAML path of the offending field.
- `{{ defined }}` with `variables: {defined: 42}` still renders correctly to `42`.

## 8. Errors & failure modes

Single error type: `class UserFileError(AIPerfError)`. All three failure cases
abort the run before any benchmark service boots.

| Failure | When | Message includes |
|---|---|---|
| Path validation | Config load (pydantic) | path, reason (`absolute path rejected` / `'..' rejected` / `non-printable chars` / `empty path`) |
| Format/content mismatch | Config load (pydantic) | path, format, type of content received |
| Template render | Run-start (renderer) | path, missing variable name (or jinja error verbatim), list of available context keys |
| Resolved-path escape | Run-start (write helper) | path, resolved abs path, run_dir |
| Write failure | Run-start (write helper) | path, resolved abs path, OS errno text |

In Kubernetes, the controller pod exits non-zero. The operator's `monitor` handler
reflects this on `status.phase: Failed` with the error message in `status.conditions`
(reusing existing controller-failure surfacing — no new status logic).

Locally, `aiperf profile` exits non-zero with the error printed to stderr.

## 9. Testing strategy

### 9.1 Unit tests (`tests/unit/config/test_user_files.py`)

- `UserFile` model:
  - Valid plain filename, valid subdir, valid deep nested subdir.
  - Invalid: absolute path, contains `..`, empty path, null byte, control char.
  - Format inference: str content → text; dict content → json; explicit `format: yaml`.
  - Format/content mismatch: `format: json` with str content raises;
    `format: text` with dict content raises.
- `build_user_file_context`:
  - All injected names present; values match RunMeta fields.
  - Collision warning logged when a user variable shadows an injected name.
- `materialize_user_files`:
  - Renders structured JSON correctly (int values stay int, not str).
  - Renders YAML correctly (round-trips via `yaml.safe_load`).
  - Renders text correctly with literal newlines preserved.
  - Strict undefined raises `UserFileError` naming the variable and file path.
  - Writes to subdirs (creates intermediate dirs).
  - Path-escape via symlink in `run_dir` rejected.
  - Existing file overwritten without warning (default behavior).
  - Write failure (read-only run_dir) raises `UserFileError`.

### 9.2 Strictness regression (`tests/unit/config/loader/test_jinja_strict_undefined.py`)

- `{{ undefined_var }}` raises `ConfigurationError` mentioning `undefined_var`.
- Existing template fixtures still render successfully (covered by full unit-test pass).

### 9.3 Component-integration

Add `tests/component_integration/test_user_files_e2e.py`:

- `aiperf profile -c config.yaml` with two `user_files` entries (one json, one text).
  Assert files exist in run dir, JSON parses, text contents match expected.
- Failure paths: bad path, missing variable, write failure (chmod read-only run_dir).
  Each aborts the run before benchmark services start.

### 9.4 Kube e2e

Extend the existing operator e2e suite (`tests/e2e/operator_ui/`) with one test:
submit an `AIPerfJob` with a single `user_files` entry, assert the file lands in
the run dir on the controller pod's PVC and downloads correctly via `aiperf kube
results`. (Goal: prove the controller-pod call site fires; not exhaustive.)

## 10. Documentation

- New: `docs/kubernetes/user-files.md` — the user-facing reference. Describes the
  schema, the injected context names (canonical list), examples for the dynamo
  `input_config.json` use case and a free-form `notes.md` use case.
- Update: `docs/kubernetes/configuration.md` — add a one-line link to the new doc.
- Update: `docs/dev/patterns.md` — short pattern: "User-defined templated outputs".
- Update: `docs/architecture.md` if the artifact-dir lifecycle section calls out
  what gets written there.
- Update: `CLAUDE.md` (and the two sync files) — add a one-line note that
  user-defined output files are declared via `artifacts.user_files`.
- Update: `llms.txt` — link the new `user-files.md` page.

## 11. Worked example: dynamo input_config.json

Translates the upstream dynamo `input_config.json` exactly:

```yaml
artifacts:
  user_files:
    - path: input_config.json
      format: json
      content:
        gpu_count: "{{ deployment_gpu_count }}"
        concurrency_per_gpu: "{{ concurrency_per_gpu }}"
        total_concurrency: "{{ concurrency_per_gpu * deployment_gpu_count }}"
        mode: "{{ deployment_mode | default('disagg') }}"
        isl: "{{ isl }}"
        osl: "{{ osl }}"
        endpoint: "{{ endpoint_url }}"
        model: "{{ model }}"
```

With the dynamo recipe's `variables:` block providing `concurrency_per_gpu`,
`deployment_gpu_count`, `isl`, `osl`, `deployment_mode` — all of which already
exist in the translated CR — this produces a JSON file byte-equivalent to the
shell `cat > … <<EOF` block in the upstream Job.

## 12. Open follow-ups (out of scope)

These are explicitly deferred to follow-up specs and are listed here so they
don't get dropped on the floor:

1. **Post-run files** with a `results.*` context namespace.
2. **Per-phase scope** (`scope: run | phase`) for sweeps that want one file per concurrency.
3. **`aiperf config validate --render-user-files`** that pre-renders with a stub context.
4. **Wait-for-model-id readiness gate** — separate spec; the operator currently does
   only one-shot reachability via `health.py`. Out of scope here, listed for tracking.
5. **`workers-max` as a CRD field** — exposed today only on the kube CLI
   (`KubeManageOptions.workers`); making it a CR field is a separate concern.
