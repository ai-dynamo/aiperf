---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Preflight Checks
---

# Preflight Checks

`native-k8s/v1` validation is a native `aiperf kube` responsibility. The
`validate` and `profile` commands require a strict workload envelope and an
explicit `--image-capabilities` document whose image digest matches that
envelope. The envelope also requires a pullable immutable
`registry/repository@sha256:<64hex>` image reference whose suffix exactly
matches its separate image digest. They validate all three bindings before
creating bootstrap Secrets or submitting an AIPerfJob. The separate
`preflight` command performs an authenticated, bounded Kubernetes `/version`
reachability check.

The independently packaged `aiperf-k8s-operator` only validates the versioned
envelope after submission. It may reject an unsupported version, malformed
role/reference metadata, or a workload that cannot be materialized, but it does
not parse benchmark configuration, construct native arguments, or perform
configuration preflight on the CLI's behalf.

## Native validation boundary

The submitted controller envelope must describe exactly the `controller`,
`cell`, and `results-sidecar` roles. It contains a fixed digest-qualified image
reference, its identity digest, command, argv, environment, cell count,
controller address, artifact/config references, and bootstrap references. The
CLI rejects unknown contract versions, bare digests or tags as image
references, image-reference/digest mismatches, missing image capabilities,
capability/image-digest mismatches, roles, and fields before a JobSet can exist.
Namespace, job, and run identities are bounded DNS labels. The artifact root is
restricted to canonical `/results` descendants, and `configRef` binds both the
ConfigMap name and the SHA-256 digest of its complete `data`/`binaryData` maps.
The non-envelope `--namespace`, AIPerfJob-name, and trusted `--run-id` command
arguments use the same DNS-label syntax before they can enter an API path.

Rust creates immutable bootstrap Secret material for the controller and cells
and records each reference's name, role, mount path, and digest in the envelope.
The results sidecar has no bootstrap reference. For each source-bound bootstrap
reference, the operator performs an exact-name Secret `GET` and validates its
immutable identity metadata and exact AIPerfJob UID owner reference. Kubernetes
returns the complete Secret because RBAC cannot authorize metadata-only reads.
These controller and cell bootstraps remain cellular-execution material; they
are not credentials for results upload or retrieval.

Before creating the JobSet, the operator reads the named source ConfigMap,
hashes the canonical complete content, and refuses a digest mismatch. It copies
that content into an immutable, per-incarnation ConfigMap; workload volumes
mount only the snapshot, never the mutable source name.

## Cluster prerequisites

A native submission requires a reachable Kubernetes API and credentials selected
by explicit `--kubeconfig`, then `KUBECONFIG`, then `$HOME/.kube/config`.
Submission requires the AIPerfJob CRD, JobSet support, and a permitted workload
namespace. The Helm chart installs its pinned JobSet dependency by default.
Kubernetes and the operator remain authoritative for CRD discovery, namespace
authorization, quota, and admission after the bounded client reaches the API.

The chart installs operator RBAC for reconciliation. The operator provisions a
per-run ServiceAccount, Role, and RoleBinding for controller status reporting,
with exact `resourceNames` patch authority only. Automatic token mounting is
disabled for every workload pod; only the controller container mounts a
projected API token. The operator ClusterRole grants cluster-wide Secret
`delete` and `get` to validate and clean up controller/cell bootstrap material.
It has no Secret list/watch/create/update/patch permission. See
[RBAC and Security](rbac-security.md) for the complete authority split.

## Results preflight

The final result contract is `results-manifest.json`. The native controller
writes and fsyncs the manifest after committed artifacts, before its private
`.aiperf_results_ready.json` compatibility marker and completion status update.
The manifest, not the compatibility marker, is the durable readiness gate. The
native results sidecar exposes only health from the workload pod. It validates
all local inputs through a retained no-follow results-root descriptor, streams
bounded uploads of the exact declared set to the fixed cluster-local results API,
publishes the manifest last, and exits only after its durable acknowledgement so
the Job can terminate. Missing manifests and upload retries have finite terminal
budgets. Partial uploads remain unreadable. Completed results are retained on
the operator PVC and `aiperf kube results` retrieves them through the Kubernetes
Service proxy using a trusted `--run-id` and locally selected operator Service
and namespace; no application credential is involved.

## Failure handling

A validation failure before cluster access creates no bootstrap Secret or
AIPerfJob. Profile submission is a compensating transaction: newly created
bootstrap Secrets are removed if CR admission fails, and a created CR is removed
together with newly created Secrets if UID owner binding fails. The operator
requires every controller/cell bootstrap Secret to carry that exact non-blocking
owner reference
before it can create a JobSet. A reconciliation failure after submission is
recorded by Kopf without making the operator reinterpret configuration or
synthesize a replacement workload. The accepted lifecycle status is structured
and monotonic; controller progress reporting is best effort once a valid run
starts. All per-incarnation resources carry exact AIPerfJob owner references.
The references do not request `blockOwnerDeletion`; the delete handler
explicitly removes the JobSet, controller/cell bootstrap Secrets, immutable
snapshot, and per-run RBAC before releasing its finalizer. It does not remove
PVC-retained published results.

## Further reading

- [Native Kubernetes control-plane isolation](../specs/kubernetes-control-plane-isolation.md)
- [RBAC and Security](rbac-security.md)
- [Kueue Integration](kueue.md)
- [Production Deployments](production.md)
