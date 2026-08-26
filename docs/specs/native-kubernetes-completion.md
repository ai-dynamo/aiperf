# Native Kubernetes completion

SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Forward-looking design record. Closes the blocking and refused entries in
[kubernetes-surface-gaps.md](kubernetes-surface-gaps.md) while keeping the reconciliation
distribution in Python.

**Division of labor.** `aiperf-k8s-operator` stays Python and stays a reconciler: it watches
custom resources, materializes JobSets, and serves durable results. It gains one CRD and one
route; it gains no config parsing, no argv synthesis, and no plan expansion. Everything that
computes — envelope authoring, role-material minting, sweep expansion, child submission,
aggregation, rendering — is the native `aiperf` binary. The operator's only new
responsibility is running a pod that executes native `aiperf`.

**Contract.** `native-k8s/v1` is extended, not superseded. The workload envelope keeps its
exactly-three-role shape (`controller`, `cell`, `results-sidecar`). Sweeps are a *separate*
resource with a *separate* schema and a fourth role that never appears in a workload
envelope. No change to `image-capabilities.schema.json` semantics;
`hierarchicalAggregation: false` stays pinned.

## Phase 1 — Make cross-host runnable

Today nothing mints `AIPRFSEC` material for a deployment, so cross-host Kubernetes and SLURM
cannot run end to end. This phase is a prerequisite for every other phase.

### 1.1 Export a deployment minting seam

`rust/runtime/src/engine/cellular_bootstrap.rs` already contains the complete generator.
`mint_local_security(roles: &[CellularRole])` produces a run nonce, a controller signing key,
one signing key per role, the roster, and `encode_material` bytes per role — but keeps the
controller context in-process and returns role bundles only through the `pub(crate)`
`LocalRoleProvisioner`.

Add one public seam beside it:

```rust
/// One run's complete deployment material: the controller bundle plus one bundle per role.
pub struct DeploymentMaterial {
    pub controller: Vec<u8>,
    pub roles: BTreeMap<CellularRole, Vec<u8>>,
}

/// Mint material for a deployment that provisions every process out of band.
pub fn mint_deployment_material(roles: &[CellularRole]) -> Result<DeploymentMaterial>;
```

It shares `mint_local_security`'s body: one nonce, one controller key, one key per role, one
roster. The only addition is emitting the controller's own bundle,
`encode_material(None, run_nonce, &controller_signer, controller_verifier, &roster)`.
`CellularRole` and `DeploymentMaterial` become `pub`; nothing else in the module changes
visibility, and no private-key type crosses the boundary — only opaque bytes.

Invariants the seam must hold, each with a unit test: one nonce shared by every bundle; the
roster byte-identical in every bundle and ordered exactly `Cell(0..N)` ascending; the
controller bundle's signing key equal to the roster's controller verifier; every role
bundle's own key equal to its roster entry. These are the exact checks `decode_material`
(`:399-443`) already enforces on read, so the tests assert round-trip through `decode_material`
rather than re-implementing the format.

### 1.2 Mint during `aiperf kube profile`

`rust/cli/src/kube/submission.rs::submit_profile_transactionally` currently requires
`--bootstrap-material <role>=<path>` for every role. Change to: mint by default, accept the
flag as an override for operators who provision material themselves.

Ordering inside the existing compensating transaction (`submission.rs:328-401`), which must
not change shape:

1. Validate the envelope and the image-capability document. No material yet.
2. `mint_deployment_material(&[Cell(0), …, Cell(cells-1)])`.
3. Write each bundle through the existing `kube::bootstrap::create_bundle` — which already
   does `O_CREAT|O_EXCL|O_NOFOLLOW`, mode `0600`, `sync_all`, and returns the
   `BootstrapReference`. This gives `create_bundle` the production caller it has never had.
4. Create the immutable Secrets, unchanged.
5. Submit the CR.

Rollback already deletes created Secrets; extend it to unlink the minted files. Material
lives under a per-run directory beneath the OS temp root, `0700`, removed on success and on
rollback. **The bytes never enter the envelope, the CR, argv, logs, or an error message** —
only `{secretName, role, mountPath, sha256}`, which the existing schema pins with
`additionalProperties: false`.

Test: submission mints exactly `cells + 1` bundles; the envelope after submission contains no
byte sequence from any bundle; a forced failure at step 5 leaves no Secret and no file.

### 1.3 Project `AIPERF_CONTROLLER_PORT`

`reconciliation.py` emits every other `AIPERF_CELL_*` variable but not this one, so the
controller silently defaults to 9500 while cells dial `envelope.controller_address` verbatim.
Add the variable to the controller container from the envelope's coordinate, and reject at
`validate_envelope` any `controllerAddress` whose port disagrees with the projected value.
One operator contract test; one Rust envelope-validation test.

### 1.4 Per-rank material for SLURM

`srun` exports one environment to every task, but cell *N* needs material encoding `Cell(N)`.
Resolve it inside the binary rather than in the script.

- `aiperf slurm generate` mints once into `<run-dir>/bootstrap/`: `controller.bin` and
  `cell-<n>.bin`, each `0600`, directory `0700`.
- The generated sbatch exports `AIPERF_CONTROLLER_BOOTSTRAP_FILE=<run-dir>/bootstrap/controller.bin`
  and a new `AIPERF_ROLE_BOOTSTRAP_DIR=<run-dir>/bootstrap`.
- `rust/cli/src/slurm/mod.rs` derives its own path: rank 0 uses the controller file; rank *N*
  uses `<dir>/cell-<N-1>.bin`, where the ordinal comes from the same `SLURM_PROCID` the rank
  dispatch already reads. `AIPERF_ROLE_BOOTSTRAP_FILE` remains accepted and takes precedence,
  so an operator-provisioned mount still works.
- `require_bootstrap_mount` keeps its `0600` and regular-file preconditions; only path
  resolution changes.

`generate` gains `--run-dir` (default: a timestamped directory beside the config) and must
refuse to overwrite existing material — `create_new` semantics, not truncate. A generated
script must run: the acceptance test is the existing `slurm_sim.sh` loopback simulation
driven end to end from `aiperf slurm generate` output rather than hand-set `SLURM_*`.

## Phase 2 — `init` and `generate`

Both currently refuse with no recorded reason. Both are local-only and need no cluster.

`aiperf kube init` writes a starter pair: a Config-v2 YAML and a matching
`image-capabilities.json` skeleton with the digest field left as a placeholder that
`validate` rejects until filled. It reuses the existing config templates rather than
introducing a second template system.

`aiperf kube generate` renders, from a Config-v2 file plus `--image <sha256:…>` and
`--cells N`, the complete `native-k8s/v1` controller envelope to stdout or `--output`. It
performs no cluster contact and mints no material — material belongs to submission, because
minting outside a transaction would leak files with no owner. The rendered envelope is
exactly what `profile` would submit, so `generate | validate` must pass.

This makes the authoring path inspectable, which is the practical reason both commands were
worth having.

## Phase 3 — Native `AIPerfSweep`

The reason this is cheap: the native sweep implementation already exists and is the live one.
`rust/cli/src/sweep/plan.rs::build_benchmark_plan` expands axes into `Vec<BenchmarkRun>`,
`sweep/mod.rs::expand` handles grid and zip, and `sweep/aggregate.rs` (1,561 lines) and
`sweep/confidence.rs` already produce the cross-run aggregate and multi-run confidence
intervals. None of it is Kubernetes-aware. This phase gives it a cluster driver.

### 3.1 Resource and schema

New CRD `aiperfsweeps.aiperf.nvidia.com`, `v1alpha1`, `AIPerfSweep`, shortName `aps`. Its
`spec` mirrors AIPerfJob's discipline exactly: a single required, immutable `sweepEnvelope`
(`self == oldSelf`), `x-kubernetes-preserve-unknown-fields`, with strict validation living in
`contracts/native-k8s/v1/sweep-envelope.schema.json` on both sides rather than in the CRD.

The sweep envelope carries: `contractVersion`, `runId`, `namespace`, `sweepId`,
`imageReference`, the base Config-v2 document, the axis set, `trials`, and the
sweep-controller role (`name`, `command`, `argv`, `environment`).
`additionalProperties: false` throughout. It does **not** carry child envelopes — expansion is
the controller's job, and pre-expanding would put a plan-sized document in etcd.

`status` mirrors AIPerfJob's shape: `phase` from the same enum, plus `childRuns` (name, ref,
phase), `completedRuns`, `failedRuns`. CEL rules keep `sweepId` immutable and forbid
`completedRuns` regressing.

### 3.2 Operator side — reconcile only

`aiperf-k8s-operator` gains a `@kopf.on.create` handler for the new plural that:

1. Validates the sweep envelope against the schema (metadata-only, same posture as today —
   it does not read the base config).
2. Provisions a per-run ServiceAccount/Role/RoleBinding scoped to `create`, `get`, `list`,
   `watch`, `delete` on `aiperfjobs` **in its own namespace only**, plus `patch` on its own
   `aiperfsweeps/status` by `resourceNames`.
3. Materializes one JobSet with one replica running the sweep-controller role verbatim from
   the envelope — same `_container` projection as today, no argv synthesis.
4. Watches child AIPerfJobs by owner reference and rolls their phases into
   `.status.childRuns`.

That is the whole Python delta: one handler, one RBAC template, one status roll-up. No sweep
logic enters the operator.

### 3.3 Native sweep-controller role

New in-cluster role `aiperf sweep-controller`, alongside `controller`, `cell`, and
`results-sidecar` in `rust/cli/src/cellular_role.rs`'s neighborhood but registered in
`dispatch.rs` as its own command. It:

1. Reads its sweep envelope from the mounted ConfigMap.
2. Expands with the existing `build_benchmark_plan` — the same code path a local
   `aiperf profile --concurrency 1,2,4` takes, so local and cluster sweeps expand identically
   by construction.
3. For each child run: mints child material with `mint_deployment_material`, creates the
   child Secrets and one child AIPerfJob CR through the existing `kube::client::KubeClient`,
   with an owner reference to the AIPerfSweep so deletion cascades.
4. Applies a bounded concurrency window (`maxConcurrentRuns`, default 1) so a sweep does not
   flood the cluster; serial by default because benchmark results from concurrent runs on
   shared hardware are not comparable.
5. Watches children to terminal state through the existing bounded-watch client, patching
   `.status` as each completes.
6. On completion, fetches each child's results through the existing results service, feeds
   them to `sweep/aggregate.rs`, and uploads the aggregate as its own run through the
   existing sidecar upload path — so `aiperf kube results <sweep>` retrieves the aggregate
   with no new retrieval surface.

Failure policy: a child that fails is recorded and the sweep continues by default;
`--fail-fast` stops and cancels outstanding children. The sweep is `Failed` only if it
produced no successful child.

### 3.4 `aiperf kube sweep`

Replaces the refusal. Mirrors `profile`: `--envelope <sweep-envelope> --image-capabilities
<doc>`, validates both, mints the sweep-controller bundle, creates the Secret, submits the
AIPerfSweep — inside the same compensating transaction. `--watch` follows `.status.childRuns`
using the same renderer as `kube watch`.

## Phase 4 — `index` and `dashboard`

`aiperf kube index` needs no new custom resource; the original `AIPerfIndex` idea is dropped.
The operator already maintains a results index and serves `GET /index/stats`. Add
`GET /api/results/{namespace}` returning the retained `(job, run, created, ready, artifact
count)` tuples, and make `kube index` a native call that renders it through the existing
`render.rs` text/JSON formatter. Small Python addition, no reconciliation change.

`aiperf kube dashboard` restores the deleted `rust/cli/src/kube/dashboard.rs` — the
in-process loopback-only forwarder that spawns no kubectl — and points the **existing** local
SPA under `rust/cli/src/server/ui/` at the operator's results API instead of a local artifact
directory. That SPA already renders runs, run detail, comparison, and sweeps; the only change
is a results source behind a trait. The operator's 14-line health stub stays a health stub;
no dashboard is served from the cluster.

## Non-goals

**Hierarchical aggregation stays refused.** Re-enabling it needs controller-owned aggregator
provisioning, authenticated admission on every tree edge, attested controller replies, and
bounded lifecycle ownership — a separate contract, as recorded in
[kubernetes-surface-gaps.md](kubernetes-surface-gaps.md). Nothing here weakens the four
refusal layers, and the sweep-controller role is deliberately *not* a cellular peer: it talks
to the Kubernetes API, never to cells.

**Per-push authentication on the controller-to-cell dataset and phaser routes** is a separate
follow-up and is not addressed here.

**The results plane keeps its current posture** — digest and length metadata, no signatures,
no capability tokens, Kubernetes RBAC on the Service proxy. This is an accepted decision, not
an omission.

## Verification

Per-phase, proportional, and product-level where behavior is new.

Phase 1 requires an ignored kind test that submits a real multi-cell `profile` with minted
material and asserts the run reaches `Completed` with per-cell records — the first end-to-end
cross-host proof the repository will have. Phase 1.4 requires the `slurm_sim.sh` acceptance
described above.

Phase 3 requires the product-level test the project mandates for new behavior: a sweep over a
deterministic `aiperf-mock-server` with pinned tokenizer, fixed TTFT and ITL, and jitter at
zero, asserting per-child raw records — ISL, OSL, model, streaming mode, status — and that the
uploaded aggregate matches a local `aiperf profile` sweep over the same axes within the
documented transport-overhead tolerance. Byte-exact local-versus-cluster expansion is
assertable directly because both call `build_benchmark_plan`.

Every phase adds hermetic coverage first; kind-dependent tests stay `#[ignore]` and run only
in the provisioned CI job.

## Sequencing

Phase 1 is a prerequisite for everything and is independently valuable — it makes the shipped
cross-host feature actually runnable. Phase 2 is small and unblocks authoring ergonomics.
Phase 3 is the substantial one and depends on Phase 1 for child material. Phase 4 is
independent of all three and can land whenever.
