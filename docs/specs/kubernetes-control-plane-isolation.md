<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Native Kubernetes control-plane isolation

## Purpose

Define the native Kubernetes control-plane boundary: a Rust-owned `aiperf kube`
interface and execution plane plus an independently packaged Python
reconciliation service. The boundary is versioned data, not shared source code.

## Contract and distribution versions

- Contract: `native-k8s/v1`, schemas in `contracts/native-k8s/v1/`.
- CRD: `aiperfjobs.aiperf.nvidia.com`, group `aiperf.nvidia.com`, version
  `v1alpha1`.
- Reconciliation distribution: `aiperf-k8s-operator` 0.1.0, Python-only, with
  its own `pyproject.toml` and dependency set.

## Role topology

`native-k8s/v1` has exactly three workload roles: `controller`, `cell`, and
`results-sidecar`. Hierarchical aggregation is refused; a hierarchy-capable
release requires a separately approved `native-k8s/v2` contract. The native
`aggregator` role is refused before argument parsing.

## Ownership split

`aiperf-cli` owns user authentication, configuration projection, image
capability validation, submitted envelopes, bootstrap-secret creation,
AIPerfJob submission, native workload argv, and user-facing results rendering.
It creates immutable named bootstrap Secrets and emits only their name, role,
mount path, and SHA-256 digest into the envelope. The public `kube` dispatch
routes to `kube::command::run`; no command delegates to Python.

`aiperf-k8s-operator` is an independent Python distribution. It owns JobSet
reconciliation, retry and recovery, result indexing, a cluster-local API, and
its dashboard. It does not import `aiperf.*`, parse benchmark configuration,
construct native argv, mint bootstrap material, or read, list, hash, or log
Secret data. It validates only declared Secret reference metadata: name, role
label, immutability, and digest annotation.

The cross-boundary schema directory is `contracts/native-k8s/v1/`. Consumers
reject unknown contract versions, unknown required capabilities, malformed
envelopes, and image capability mismatches before a JobSet is created. A
producer may add optional fields. Controller status reporting is best-effort
after a valid controller envelope starts.

A v1 controller envelope carries the run and CR identity, image digest, cell
count, config and artifact references, controller address, and complete
per-role command, argv, environment, and bootstrap references. The operator
materializes those values; it never derives a container image, replica count,
or argv.

## Native command surface

`aiperf kube` exposes fifteen native commands: `init`, `validate`, `profile`,
`sweep`, `generate`, `attach`, `list`, `logs`, `results`, `show`, `debug`,
`watch`, `preflight`, `dashboard`, and `index`. Kubernetes access flows through
one `KubeClient`/`KubeTransport` seam with finite request and watch deadlines,
a bounded response body, and bounded newline-delimited watch records. Watch
streams reconnect a bounded number of times. Log streaming preserves bytes
without reframing. `dashboard` forwards in process on loopback only and rejects
non-loopback peers before any payload; no command spawns `kubectl`.

## Results contract

The results contract is `results-manifest.json`. It is atomically written and
fsynced after artifacts commit, before the private
`.aiperf_results_ready.json` compatibility marker and the controller completion
report. The native results sidecar gates, lists, and serves the manifest and
only manifest-declared artifacts; the compatibility marker is not a network
API. `aiperf kube results` accepts only a ready manifest, refuses traversing or
duplicate paths and malformed digests, and verifies each artifact's SHA-256
before writing it, so a substituted transfer never lands on disk.

## Verification

- `cargo fmt --check` and `cargo clippy --all-targets` are clean for this
  surface.
- `cargo test -p aiperf-cli --lib`: 193 passed, 0 failed, including 30
  `kube::` tests.
- `cargo test -p aiperf-e2e-tests --test kube_cli_contract`: 16 passed, 0
  failed, 4 ignored. The ignored tests require kind, Helm, and `KUBECONFIG`,
  and run in the `native-cli-kind` CI job with `-- --ignored`.
- `pytest -n auto aiperf-k8s-operator/tests/unit aiperf-k8s-operator/tests/contract`:
  9 passed, covering exact two-JobSet projection, metadata-only Secret
  validation, absence of `aiperf.*` imports, and manifest-gated artifact
  serving.
- `tools/check_agent_files_sync.py` and `tools/check_docs_current.py` exit
  zero.

## Source anchors

- `rust/cli/src/kube/` — native command surface, client seam, contract DTOs,
  submission, rendering, results, and loopback forwarding.
- `rust/cli/src/k8s.rs` — native in-cluster status and completion reporting.
- `rust/cli/src/cellular_role.rs` — native role selection and refusal behavior.
- `rust/cli/src/results_sidecar.rs` — native artifact serving seam.
- `rust/runtime/src/engine/cellular_bootstrap.rs` — role bootstrap handling.
- `contracts/native-k8s/v1/` — versioned data contracts and fixtures.
- `aiperf-k8s-operator/` — independent reconciliation distribution.
- `deploy/aiperf-k8s-operator/` — CRD, RBAC, and Helm chart.
- `rust/e2e-tests/tests/kube_cli_contract.rs` — public command contract tests.
