<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Native Kubernetes control-plane isolation

## Purpose

Define the native Kubernetes control-plane boundary. The target is a Rust-owned
`aiperf kube` interface and execution plane plus an independently packaged
Python reconciliation service. The boundary is versioned data, not shared
source code.

## Built

The native cellular controller already owns controller/cell execution and
best-effort in-cluster AIPerfJob progress reporting. The current Python root
CLI does not mount a `kube` command; it is not a compatible public fallback.
The current native `aggregator` role is refused.

## Future requirements

`native-k8s/v1` has exactly three workload roles: `controller`, `cell`, and
`results-sidecar`. Hierarchical aggregation is refused; a hierarchy-capable
release requires a separately approved `native-k8s/v2` contract.

`aiperf-cli` owns user authentication, configuration projection, image
capability validation, submitted envelopes, bootstrap-secret creation,
AIPerfJob submission, native workload argv, and user-facing results rendering.
It creates immutable named bootstrap objects and emits only their names, role,
mount path, and SHA-256 digest into the envelope.

`aiperf-k8s-operator` is an independent Python distribution. It owns JobSet
reconciliation, retry/recovery, result indexing, a cluster-local API, and its
dashboard. It must not import `aiperf.*`, parse benchmark configuration,
construct native argv, mint bootstrap material, or read/list/hash/log Secret
data. It validates only declared Secret reference metadata: name, role label,
immutability, and digest annotation.

The cross-boundary schema directory is `contracts/native-k8s/v1/`. Consumers
reject unknown contract versions, unknown required capabilities, malformed
envelopes, and image capability mismatches before a JobSet is created. A
producer may add optional fields. Controller status reporting remains
best-effort after a valid controller envelope starts.

A v1 controller envelope carries the run and CR identity, image digest, cell
count, config/artifact references, controller address, and complete per-role
command, argv, environment, and bootstrap references. The operator materializes
those values; it never derives a container image, replica count, or argv.

The results contract is `results-manifest.json`. It is atomically written and
fsynced after artifacts commit, before the private legacy
`.aiperf_results_ready.json` compatibility marker and controller completion
report. The native results sidecar gates, lists, and serves the manifest and
only manifest-declared artifacts; the compatibility marker is not a network
API.

## Source anchors

- `rust/cli/src/k8s.rs` — native in-cluster status and completion reporting.
- `rust/cli/src/cellular_role.rs` — native role selection and refusal behavior.
- `rust/cli/src/results_sidecar.rs` — native artifact serving seam.
- `rust/runtime/src/engine/cellular_bootstrap.rs` — role bootstrap handling.
- `contracts/native-k8s/v1/` — planned versioned data contracts.
- `aiperf-k8s-operator/` — planned independent reconciliation distribution.
