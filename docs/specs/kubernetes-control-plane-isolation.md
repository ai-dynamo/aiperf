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

`aiperf-cli` owns user authentication, configuration projection, selected image
capability-document validation, submitted envelopes, bootstrap-secret creation,
AIPerfJob submission, native workload argv, and user-facing results rendering.
It creates immutable named bootstrap Secrets and emits only their name, role,
mount path, and SHA-256 digest into the envelope. Submission creates the CR,
binds every Secret to the returned CR UID, and compensates by removing newly
created resources when admission or owner binding fails. The public `kube` dispatch
routes to `kube::command::run`; no command delegates to Python.

`aiperf-k8s-operator` is an independent Python distribution. It owns JobSet
reconciliation, retry and recovery, durable result indexing, and a
cluster-local API. It does not import `aiperf.*`, parse benchmark configuration,
construct native argv, or mint bootstrap material. It validates every declared
controller/cell Secret reference's name, role label, immutability, digest
annotation, and exact non-blocking CR owner reference. The results sidecar has
no bootstrap material; controller and cell bootstraps remain cellular-execution
material rather than results-service credentials.

The cross-boundary schema directory is `contracts/native-k8s/v1/`. Consumers
reject unknown contract versions, unknown required capabilities, malformed
envelopes, and image capability mismatches before a JobSet is created. A
producer may add optional fields. Controller status reporting is best-effort
after a valid controller envelope starts.

A v1 controller envelope carries the run and CR identity, a pullable immutable
image reference and its matching identity digest, cell count, config and
artifact references, controller address, and complete per-role command, argv,
environment, and bootstrap references. The operator projects the exact image
reference and never derives a container image, replica count, or argv.
Identity fields are bounded DNS labels, the artifact root is a canonical
`/results` descendant, and the configuration reference includes a content
digest. The operator verifies the complete source ConfigMap maps, creates an
immutable per-incarnation snapshot, and mounts only that snapshot.

The AIPerfJob spec is immutable after creation. The operator provisions one
deterministically named ServiceAccount, Role, and RoleBinding per accepted run.
The controller can patch only its exact AIPerfJob status subresource; cell pods
disable service-account token automounting. The chart installs the pinned
JobSet dependency on a fresh cluster.

## Native command surface

`aiperf kube` recognizes fifteen native command names: `init`, `validate`, `profile`,
`sweep`, `generate`, `attach`, `list`, `logs`, `results`, `show`, `debug`,
`watch`, `preflight`, `dashboard`, and `index`. Kubernetes access flows through
one `KubeClient`/`KubeTransport` seam with finite request and watch deadlines,
a bounded response body, and bounded newline-delimited watch records. Watch
streams reconnect a bounded number of times. Log streaming preserves bytes
without reframing. `validate` and `profile` require an explicit
`--image-capabilities` document bound to the envelope's image digest. `sweep` submits an `AIPerfSweep` CR; `index` lists retained result runs; `dashboard` serves the local SPA backed by the operator's results API. No command
spawns `kubectl`.

## Results contract

The results contract is `results-manifest.json`. It is atomically written and
fsynced after artifacts commit, before the private
`.aiperf_results_ready.json` compatibility marker and the controller completion
report. The native results sidecar exposes health only; neither the manifest nor
artifacts are published from the workload pod. The compatibility marker is not
a network API. The controller and regular sidecar share one writable results
`emptyDir`. Through retained no-follow descriptors, the sidecar validates and
streams only the exact manifest-declared artifact set, carrying SHA-256 and
length metadata. The operator stages artifacts on its PVC and atomically
publishes only the matching manifest-declared set; incomplete state stays
unreadable. A durable manifest acknowledgement ends the sidecar; missing
manifests or an exhausted retry budget fail it, allowing the Job to become
terminal without hanging. Published results are retained on the PVC across
producer deletion and operator restart; they are identified by namespace, job,
and run, not a UID incarnation.

`aiperf kube results` accepts only a ready manifest, refuses traversing or
duplicate paths and malformed digests, and verifies each artifact's SHA-256
before writing it through retained no-follow destination descriptors. Retrieval
requires a trusted `--run-id` and selects only the local
`--operator-service`/`--operator-namespace` identity before using the
Kubernetes Service proxy; workload annotations cannot redirect it. The CLI
addresses the durable triple directly: it does not retrieve an AIPerfJob or a
Secret and sends no application credential. Kubernetes authentication and RBAC
protect the Service-proxy hop. The chart requires an explicit operator image
repository and tag, exposes the default `aiperf-k8s-operator` Service on port
8080 as ClusterIP, and mounts a single-owner PVC. Default uploads use
cluster-local HTTP, which assumes trusted namespace-level in-cluster access and
does not provide transport confidentiality. No external ingress is shipped.

## Verification

The hermetic Rust contract is exercised with
`cargo test -p aiperf-cli --lib kube` and
`cargo test -p aiperf-e2e-tests --test kube_cli_contract`. Operator unit and
contract behavior is exercised with
`pytest aiperf-k8s-operator/tests/unit aiperf-k8s-operator/tests/contract`.

The operator integration test is environment-gated: with neither
`KUBECONFIG` nor `AIPERF_K8S_INTEGRATION` it reports a skip. The kind CI job
installs the chart with Helm `--wait`, then runs the test against the live API.
The test requires an available operator Deployment, an established AIPerfJob
CRD, and no invented AIPerf custom-resource kinds.

## Source anchors

- `rust/cli/src/kube/` — native command surface, client seam, contract DTOs,
  submission, rendering, and results.
- `rust/cli/src/k8s.rs` — native in-cluster status and completion reporting.
- `rust/cli/src/cellular_role.rs` — native role selection and refusal behavior.
- `rust/cli/src/results_sidecar.rs` — native artifact serving seam.
- `rust/runtime/src/engine/cellular_bootstrap.rs` — role bootstrap handling.
- `contracts/native-k8s/v1/` — versioned data contracts and fixtures.
- `aiperf-k8s-operator/` — independent reconciliation distribution.
- `deploy/aiperf-k8s-operator/` — CRD, RBAC, and Helm chart.
- `rust/e2e-tests/tests/kube_cli_contract.rs` — public command contract tests.
