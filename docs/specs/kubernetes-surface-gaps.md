# Kubernetes surface gaps

SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Companion to [kubernetes-control-plane-isolation.md](kubernetes-control-plane-isolation.md).
That record states the `native-k8s/v1` boundary as designed. This one states what the shipped
surface does **not** do, verified against code at `cf09af5034`. Every entry carries a
`file:line`. An entry leaves this document when the gap closes, not when it is explained.

## Deployment blockers

**No shipped code mints cross-host role material.** Every cellular process outside the
same-host path must load an `AIPRFSEC` blob (magic, version, 9-byte role, 32-byte run nonce,
Ed25519 seed, controller verifier, roster) through `read_private_deployment_file`
(`rust/runtime/src/engine/cellular_bootstrap.rs:296-327`, exact `0600` + `O_NOFOLLOW` +
regular-file). The only shipped writer is
`rust/cli/src/kube/bootstrap.rs:16 create_bundle`, which has **no production caller** —
`git grep create_bundle -- rust/` returns the definition and its own two tests — and which
takes opaque `contents: &[u8]` rather than producing `AIPRFSEC` bytes. `encode_material`
(`cellular_bootstrap.rs:343`) is private and reachable only from `mint_local_security`, the
same-host path. `aiperf kube profile` consumes pre-existing files via
`--bootstrap-material <role>=<path>` (`rust/cli/src/kube/submission.rs:71-149`);
`docs/kubernetes/source-checkout-deploy.md:157` calls them *"opaque cellular-execution
material produced with the envelope"* and names no producer. Cross-host Kubernetes and SLURM
cannot be run end to end without hand-authoring the binary format.

**`aiperf slurm generate` emits a script that cannot run.** `build_sbatch_script`
(`rust/cli/src/slurm/generate.rs:134-165`) exports only `AIPERF_CELL_LAUNCHER` and
`AIPERF_CONTROLLER_PORT`. `AIPERF_CONTROLLER_BOOTSTRAP_FILE` and `AIPERF_ROLE_BOOTSTRAP_FILE`
are required at runtime (`rust/cli/src/slurm/mod.rs:104,107`) and never generated, so every
task fails at `require_bootstrap_mount`. Structurally, `srun` exports one environment to all
tasks while cell *N* needs material encoding `Cell(N)` (`cellular_bootstrap.rs:387`); no
per-rank wrapper is generated or documented.

**`AIPERF_CONTROLLER_PORT` is never projected.** The operator emits every other
`AIPERF_CELL_*` variable (`aiperf-k8s-operator/src/aiperf_k8s_operator/reconciliation.py:34-54`)
but not this one. The controller defaults to 9500
(`rust/runtime/src/engine/cellular_controller.rs:1745`) while cells dial
`envelope.controller_address` verbatim; the shipped fixture
`contracts/native-k8s/v1/fixtures/valid-one-cell-envelope.json` uses `controller:443`.
Nothing in the operator, the CRD schema, or `rust/cli/src/kube/contract.rs` cross-validates
the two.

## Trust boundaries the shipped code does not enforce

**Controller-to-cell pushes are unauthenticated.** The cell-to-controller `subscribe`
direction authenticates through `registration_authority.open_payload`
(`rust/runtime/src/cellular/transport/phaser_velo.rs:88-93`). The controller-to-cell `event`
push does not (`phaser_velo.rs:169-178`): no role, purpose, nonce, sequence, peer digest,
payload digest, or replay ledger, behind an unbounded `mpsc` (`:165`). Any party that can
reach a cell's Velo instance and address `aiperf.phaser.event` can inject phase transitions,
including a `Finalized` that ends the cell's measurement early. Disclosed at
`phaser_velo.rs:26-32` and in the four agent files. `cellular/transport/dataset_velo.rs` has
the same property and carries **no** equivalent module-level disclosure.

**The operator holds cluster-scoped Secret `get`.** `deploy/aiperf-k8s-operator/rbac/
operator-clusterrole.yaml` grants `secrets: ["delete","get"]`, and
`aiperf-k8s-operator/src/aiperf_k8s_operator/main.py:307-349` performs a real
`read_namespaced_secret`. Kubernetes returns `.data` on an exact-name `get`, so
reference-only handling is enforced by code discipline and tests
(`contract.py:189-203`, `tests/contract/test_contract.py:505`), not by an API-level boundary.
`docs/kubernetes/rbac-security.md:36-40` states this; the verb table at `:25` does not — it
lists jobsets as `create, delete, get` while both manifests grant `create, delete, get,
list, watch`, under a heading claiming the table holds *"only the permissions exercised by
current reconciliation."*

**The cellular dependency is a personal fork.** `rust/Cargo.lock` pins `velo` to
`git+https://github.com/ajcasagrande/velo.git?branch=feat%2Fconnect-by-endpoint`, not an
upstream tag.

## Silent degradation

**Sidecar telemetry is dropped in any multi-cell run.** `server_metrics`, `gpu_telemetry`,
and `network_latency` are discarded whenever `cells > 1`
(`rust/runtime/src/engine/cellular_controller.rs`, `warn_dropped_sidecar_telemetry`). This is
a warning, not a gate: a cluster run with GPU telemetry enabled produces a report with no
telemetry and exit 0.

**Per-cell worker counts assume co-location.** `let per_cell = (workers /
u64::from(cell_count)).max(1);` carries the comment *"so N cell processes on one host target
~`workers` total threads."* When cells are pods on separate nodes this under-threads each pod
by a factor of `cell_count`.

**Cross-host transport is feature-gated twice.** `rust/runtime/src/cellular/transport/` is
gated on `all(feature = "cellular", feature = "engine")` since `225bf6eebc`. An image built
without `engine` silently has no cross-host cellular. This is the second occurrence of the
pattern; the first was the pre-`1623a6bb7a` `velo` gate, where a default `cargo build`
produced a binary that failed closed on `cells > 1`.

## Refused commands

`rust/cli/src/kube/command.rs:24-40` lists fifteen names; five refuse before any cluster
access, two are pure aliases, eight are distinct live commands.

| Command | Site | Recorded justification |
|---|---|---|
| `sweep`, `index` | `command.rs:59-63` | `kubernetes-control-plane-isolation.md:80-83` — v1 ships no corresponding custom resources |
| `dashboard` | `command.rs:64-68` | assertion only; no record explains why the working loopback forwarder was deleted |
| `init`, `generate` | `command.rs:54-58` | **none** — no commit body, spec text, code comment, or test |

`debug` is a pure alias of `show` and `attach` of `watch` (`command.rs:79-85`); no record
addresses whether they were intended to differ. `preflight` is `GET /version`
(`command.rs:77`) and ignores the `--namespace` it parses.

Hierarchical aggregation is refused at four independent layers: `NativeK8sRole` has no
`Aggregator` variant (`rust/cli/src/kube/contract.rs:74-81`); the envelope schema pins
`roles` to `minItems: 3, maxItems: 3`; `contracts/native-k8s/v1/image-capabilities.schema.json`
requires `"hierarchicalAggregation": {"const": false}`; and the runtime bails in both
`rust/runtime/src/engine/cellular_aggregator.rs:22-25` and `cellular_controller.rs:110-113`.

## Untested and unvalidated

`contracts/native-k8s/v1/progress-status.schema.json` and its byte-identical copy under
`aiperf-k8s-operator/src/aiperf_k8s_operator/contracts/v1/` are validated by nothing — zero
code references, while the other three schemas each have a validator. `rust/cli/src/k8s.rs`
`progress_body` output is never schema-checked.

The four `native-k8s/v1` schemas exist as byte-identical duplicates in two trees with no sync
check; the CRD is likewise duplicated between `deploy/aiperf-k8s-operator/crds/` and the
chart's `crds/`.

The kind CI job covers `preflight` and `list` only.
`aiperf-k8s-operator/tests/contract/test_ci_workflow.py:79` asserts `'"profile"' not in
live_contract` — the live surface is contractually barred from exercising submission. No
operator-side test asserts that a fourth-role envelope is refused; that coverage is Rust-only.

## Documentation that contradicts the shipped surface

`docs/kubernetes/kueue.md` (291 lines) documents an integration with **zero backing code** —
`kueue|localqueue|clusterqueue|suspend` returns no matches across `rust/`,
`aiperf-k8s-operator/src/`, `deploy/`, `contracts/`, `src/aiperf/`. It documents
`--queue-name`/`--priority-class` flags that `command.rs:460-486` does not accept (and
unknown flags are silently ignored, so the documented invocation submits unqueued),
`spec.scheduling.queueName` and `spec.benchmark.*` fields against a CRD whose `spec` requires
only `envelope`, plus `dev/versions.py`, `_verify_kueue_local_queue`, and
`templates/benchmark-namespace.yaml` — none of which exist.

`src/aiperf/cli_runner/_multi_run.py:232` is a live runtime error string instructing users to
*"Use the AIPerfSweep CRD"*, and `:233` links `docs/kubernetes/sweeps.md`, deleted by
`8ef87d7c51`. Ten further files still reference `AIPerfSweep`, including
`tests/unit/config/test_benchmark_plan.py:611` (*"Mirrors the CEL admission rule on
`AIPerfSweepSpec`"* — no such CRD or rule exists).

Results carry a 24-hour staging and 7-day published TTL
(`aiperf-k8s-operator/src/aiperf_k8s_operator/results.py:90,93`, expiry `:380-397`). No
document mentions either; `llms.txt`, the four agent files, and three `docs/kubernetes/`
pages all promise retention after producer deletion and operator restart with no expiry.

`CLAUDE.md:93`, `AGENTS.md:93`, and `llms.txt:71-73` list the refusals as sweep/index/
dashboard, omitting `init` and `generate`.
`kubernetes-control-plane-isolation.md:27` says the aggregator role is refused *"before
argument parsing"* — true of the internal `--aggregator` flag (`rust/cli/src/main.rs:43`
precedes `:47`) but not of the public subcommand, which requires `--config`, projects an
envelope, and re-execs before the child refuses
(`rust/cli/src/cellular_role.rs:96-121` → `execute_mode.rs:217-235`).
`kubernetes-control-plane-isolation.md:67-68` attributes automount disabling to cell pods
only; `reconciliation.py:385` and `:402` apply it to both.
`rust/cli/src/kube/results.rs:6` still describes the manifest as *"authenticated"* after
`81a91acb8a` removed all read authentication.

`tools/check_docs_current.py` and `tools/check_agent_files_sync.py` both exit 0 and catch
none of the above.

## Repository hygiene affecting this surface

`uv.lock` still declares `kopf`, `kubernetes-asyncio`, and `aiosqlite` as project dependencies
after `ae6de0c22a` removed them from `pyproject.toml`; the two files are desynced.
`pyproject.toml:103-107` claims *"121 packaged non-`.py` files across 17 leaf directories
(operator UI `.js`/`.css`/`.html`, …)"* — the actual count is 53 files across 12 leaf
directories with no operator UI. A duplicate `specs/` tree (40 files) survives at the
repository root, diverged from `docs/specs/` for `architecture.md`, `slurm-native.md`, and
`cellular.md`, indexed by neither `docs/specs/README.md` nor `llms.txt`. Four
`typed-factory-runner.md.*.contest.md` files (~227 KB, two recording failed contests) are
tracked inside `docs/specs/` and referenced by nothing.
