<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# User-Defined Output Files (`artifacts.user_files`)

`artifacts.user_files` lets you declare arbitrary templated output files that are
materialized into the run directory before the benchmark begins. Files are rendered
with jinja2 against the user `variables:` block plus a small set of system-injected
names. The same mechanism works for `aiperf profile` (local) and `AIPerfJob`
(Kubernetes) — both load the same config block.

## Quickstart

```yaml
variables:
  isl: 1024
  osl: 512

benchmark:
  artifacts:
    user_files:
      - path: input_config.json
        format: json                 # optional; inferred from content type
        content:
          isl: "{{ isl }}"
          osl: "{{ osl }}"
          endpoint: "{{ endpoint_url }}"
          model: "{{ model }}"

      - path: meta/notes.md          # subdirectories allowed
        content: |
          Run {{ job_name }} started at {{ epoch }}.
          Targeting {{ model }} @ {{ endpoint_url }}.

  models:
    - my-org/my-model
  endpoint:
    type: chat
    urls: ["http://my-frontend:8000"]
  datasets:
    - name: main
      type: synthetic
  phases:
    - name: profiling
      type: concurrency
      concurrency: 10
      requests: 100
```

Result in the run directory:

```
{artifact_dir}/{epoch}_{job_name}/    # operator-managed (AIPerfJob)
├── input_config.json
├── meta/
│   └── notes.md
└── ... (standard AIPerf artifacts)
```

For local `aiperf profile` runs the layout is whatever `artifacts.dir` points at —
typically a single per-run directory with no `{epoch}_{job_name}` wrapper. The
`{{ epoch }}` system-injected name still resolves: locally it's wall-clock seconds
captured at run start, and `{{ job_name }}` is the `--artifact-dir` basename.

## Schema

Each entry is:

| Field | Type | Required | Description |
|---|---|---|---|
| `path` | string | yes | Output path **relative** to the run directory. Subdirectories OK. Absolute paths and any segment equal to `..` are rejected. |
| `format` | `json` \| `yaml` \| `text` | no | Serialization format. If omitted: `text` when `content` is a string, `json` otherwise. |
| `content` | structured or string | yes | Templated value. Dict/list/scalar for `json`/`yaml`; string for `text`. Jinja2 expressions in any string leaf are rendered. |

Format/content compatibility:
- `format: json` or `format: yaml` requires structured `content` (dict/list/scalar).
- `format: text` requires string `content`.

**Dict keys are not rendered.** Jinja2 expressions only resolve in string *values*,
not in dict keys. `content: {"{{ model }}": "x"}` writes a file with the literal key
`"{{ model }}"`, not the resolved model name. Put templated values where they belong
— in values — or pre-flatten the dict before passing it to AIPerf.

## Templating context

Inside `content`, you can reference:

**1. User-declared variables** — anything you put in the top-level `variables:` block of your config. Variables may reference each other (in any YAML order); cross-references are resolved in dependency order at config-load time, so a derived variable like `total_concurrency: "{{ concurrency_per_gpu * deployment_gpu_count }}"` already holds its computed value (`120`) by the time `user_files` rendering runs. Cycles raise `ConfigurationError`.

**2. System-injected names** (stable API):

| Name | Type | Meaning |
|---|---|---|
| `epoch` | str | Run epoch identifier (e.g. `"1714000000"`). |
| `job_name` | str | AIPerfJob name in Kubernetes; `--artifact-dir` basename locally. |
| `namespace` | str | Kubernetes namespace; empty string locally. |
| `model` | str | First entry of `benchmark.models`. |
| `endpoint_url` | str | First entry of `benchmark.endpoint.urls`. |
| `artifact_dir` | str | Absolute path to the run directory. |

**Collision rule:** if a user `variables:` key shadows an injected name, the injected name wins and a `WARNING` is logged at startup. Rename your variable.

## Errors

These are all fatal — the benchmark does not start.

| Failure | Cause | Where you see it |
|---|---|---|
| Path validation | Absolute path, `..` segment, empty path, control chars | Config load (pydantic `ValidationError`) |
| Format/content mismatch | e.g. `format: json` with `content: "string"` | Config load (pydantic `ValidationError`) |
| Undefined variable | Template references a name not in context | Run start (`UserFileError`); message names the file path and the variable |
| Path escape | Resolved path is not inside the run directory | Run start (`UserFileError`) |
| Write failure | Disk full, permission denied, etc. | Run start (`UserFileError`); message includes resolved path and OS error |

In Kubernetes, controller-pod failures surface on `status.phase: Failed` with the
error in `status.conditions`.

## Use cases

- **Sidecar metadata** — produce an `input_config.json` for downstream tooling that expects
  the dynamo-style deployment-shape file.
- **Run notes** — write a `notes.md` summarizing what this run is for, who triggered it.
- **Manifests** — emit a manifest a downstream pipeline will read.

## Limitations (v1)

- **Pre-run only.** Files render before the benchmark starts. Post-run files that include
  results are tracked as a future extension.
- **Files always overwrite.** No `overwrite: false` safety net.
- **No `required: false`.** Every declared file must materialize successfully or the run aborts.
- **Strict undefined.** A typo in `{{ varaibles_name }}` is a hard error, not a silent empty string.
