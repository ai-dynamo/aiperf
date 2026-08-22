---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Preflight Checks
---

# Preflight Checks

`native-k8s/v1` preflight is a native `aiperf kube` responsibility. Before it
creates bootstrap material or submits an AIPerfJob, the CLI validates the local
configuration projection, Kubernetes authentication, target namespace, required
APIs, image capability document, and the submitted workload envelope.

The independently packaged `aiperf-k8s-operator` only validates the versioned
envelope after submission. It may reject an unsupported version, malformed
role/reference metadata, or a workload that cannot be materialized, but it does
not parse benchmark configuration, construct native arguments, or perform
configuration preflight on the CLI's behalf.

## Native validation boundary

The submitted controller envelope must describe exactly the `controller`,
`cell`, and `results-sidecar` roles. It contains fixed image, command, argv,
environment, cell-count, controller-address, artifact/config references, and
bootstrap references. The CLI rejects unknown contract versions, capabilities,
roles, and fields before a JobSet can exist.

Bootstrap validation is reference-only across the Rust/operator boundary. Rust
creates immutable role-specific Secret material and records its name, role,
mount path, and digest in the envelope. The operator validates declared name,
role label, immutable flag, and digest annotation only. It must not read, list,
hash, log, or otherwise access Secret data.

## Cluster prerequisites

A native submission requires a reachable Kubernetes API and credentials selected
by explicit `--kubeconfig`, then `KUBECONFIG`, then `$HOME/.kube/config`.
It requires the AIPerfJob CRD, JobSet support, a permitted workload namespace,
and any requested Kueue queue. The CLI reports authorization, TLS, quota,
namespace, and admission prerequisites before it submits a run.

The chart installs operator RBAC for reconciliation and workload RBAC for
controller completion/progress reporting. It deliberately does not grant the
operator Secret-data read permissions. See [RBAC and Security](rbac-security.md)
for the complete authority split.

## Results preflight

The final result contract is `results-manifest.json`. The native controller
writes and fsyncs the manifest after committed artifacts, before its private
`.aiperf_results_ready.json` compatibility marker and completion status update.
The manifest, not the compatibility marker, is the public readiness gate. The
native results sidecar exposes only a valid manifest and the artifacts it
declares.

## Failure handling

A validation failure before submission creates no bootstrap Secret, envelope, or
AIPerfJob. A reconciliation failure after submission is recorded in AIPerfJob
status without making the operator reinterpret configuration or synthesize a
replacement workload. Controller progress reporting is best effort once a valid
run starts.

## Further reading

- [Native Kubernetes control-plane isolation](../specs/kubernetes-control-plane-isolation.md)
- [RBAC and Security](rbac-security.md)
- [Kueue Integration](kueue.md)
- [Production Deployments](production.md)
