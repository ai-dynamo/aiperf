<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Harbor sample task packages

Hand-authored Harbor-compatible packages covering the package shapes
`aiperf eval` accepts. They exercise the importer and the eval execution path
by hand, against a real Docker daemon, in the cases the automated suites do not
reach.

Resolve them from a test with `CARGO_MANIFEST_DIR`, the same idiom
`fixtures/harbor_p0` uses:

```rust
let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
    .join("fixtures/harbor_samples/swe-style/task.json");
```

Or run one directly:

```bash
aiperf eval --task rust/e2e-tests/fixtures/harbor_samples/swe-style/task.json \
    --image sha256:<64-hex-digest> --agent-command '<shell command>'
```

## Contents

Standalone JSON packages (schema-less `task.json`, separate verifier, so each
requires Docker):

| Directory | Task |
| --- | --- |
| `humaneval-style/` | `humaneval/HE-26-remove-duplicates` — implement a function to spec |
| `intercode-style/` | `intercode/bash-file-analysis` — shell-driven log analysis |
| `swe-style/` | `swe-bench/django-fix-off-by-one` — repair a seeded off-by-one bug |

Schema `1.1` `native_graph` task directories. Each pairs `task.toml` with a
`graph.json` program, `model-bindings.toml`, and `adapters.toml`, and needs
`--model-runtime model-runtime.toml` plus the accompanying
`lifecycle-request.json`:

| Directory | Task |
| --- | --- |
| `native-graph-task/` | `example/ng-code-review` — single-stage review graph |
| `ng-multi-node/` | `ng-bench/multi-stage-analysis` — multi-stage graph |

## Relationship to `fixtures/harbor_p0`

`harbor_p0/` holds the packages that the automated tests
(`test_harbor_p0.rs`, `test_harbor_pinned_git.rs`,
`test_harbor_verifier_isolation.rs`) assert against; treat those as pinned
contract inputs. The packages here are exploratory samples, not currently
referenced by any test, and are free to grow as new package shapes appear.
