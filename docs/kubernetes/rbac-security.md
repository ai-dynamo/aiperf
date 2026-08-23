---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: RBAC and Security
---

# RBAC and Security

Native Kubernetes v1 separates the cluster-scoped operator identity from one
namespace-scoped identity per benchmark run. The chart creates the operator
ServiceAccount, ClusterRole, and ClusterRoleBinding by default. Set
`rbac.create=false` and `serviceAccount.create=false` only when an administrator
has pre-provisioned equivalent resources and set `serviceAccount.name` to that
identity.

## Operator authority

The operator ClusterRole contains only the permissions exercised by current
reconciliation:

| API group | Resources | Verbs | Purpose |
|---|---|---|---|
| `aiperf.nvidia.com` | `aiperfjobs` | `get, list, watch, patch` | Reconcile jobs and maintain the Kopf finalizer |
| `aiperf.nvidia.com` | `aiperfjobs/status` | `patch` | Publish bounded lifecycle status |
| `jobset.x-k8s.io` | `jobsets` | `create, delete, get` | Materialize, validate, and remove the exact workload |
| core | `secrets` | `delete, get` | Read and clean up exact controller/cell bootstrap references |
| core | `configmaps` | `create, delete, get` | Read source configuration and own immutable config snapshots |
| core | `serviceaccounts` | `create, delete, get` | Own the exact run identity |
| `rbac.authorization.k8s.io` | `roles`, `rolebindings` | `create, delete, get` | Bind and remove the exact run identity |

The embedded Kopf runtime is explicitly cluster-wide, standalone, and has
Kubernetes Event posting disabled. It therefore does not require Event or
peering-object permissions beyond the table above.

There is no shipped AIPerfSweep resource or sweep-handler identity. The
operator cannot list, watch, create, update, or patch Secrets. Kubernetes
returns the complete Secret for an exact-name `get`; the operator validates
each controller/cell bootstrap reference's immutable metadata, digest
annotation, and exact CR-UID owner reference. The results sidecar has no
bootstrap material. Controller and cell bootstraps remain cellular-execution
material and are never written into the AIPerfJob, JobSet, logs, or results
store.

## Per-run workload authority

For each accepted `(namespace, job ID, run ID)`, the operator creates a
deterministically named ServiceAccount, Role, and RoleBinding. Existing objects
are accepted only when they contain the exact expected identity and rules. The
Role grants only `patch` on the submitted
`aiperfjobs/status` subresource, constrained by its exact `resourceNames`
entry. It grants no main-resource authority. The CRD admits only the monotonic
`Pending -> PublishingResults -> Completed` lifecycle (with terminal failure),
keeps run and JobSet identity immutable, and requires `Completed` and
`resultsReady` to be published together.

Every workload pod disables automatic ServiceAccount token mounting. The
controller container alone mounts a short-lived projected Kubernetes API token
and cluster CA for best-effort status reporting. The results sidecar in the
same pod does not mount that projection, and cell containers receive no API
credential. Workload pods receive no Secret API permissions; bootstrap bytes
arrive through exact read-only Secret volumes.

The AIPerfJob's exact UID is a non-blocking owner reference on its controller
and cell bootstrap Secrets, JobSet, ServiceAccount, Role, RoleBinding, and
configuration snapshot. These references do not require delete-on-owner or
owner-finalizer admission authority. A delete handler explicitly removes that
workload set; Kubernetes garbage collection is only a backstop. Results use the
durable namespace/job/run identity, not a UID incarnation.

The source ConfigMap is user-owned. `configRef.sha256` covers canonical JSON of
its complete `data` and `binaryData` maps. Reconciliation verifies that digest
and creates an immutable, UID-owned snapshot before JobSet creation. The JobSet
mounts the snapshot name, so deleting or replacing the source cannot change an
accepted workload.

## Durable results boundary

The controller and regular results-sidecar container share a writable
`emptyDir` only within their Job pod. The sidecar validates the committed
manifest, streams each exact declared artifact to the operator API, and then
publishes the manifest. Each upload carries SHA-256 and length metadata, which
the service validates against the bytes. The results service has no application
token, signature, public-key verifier, results-read Secret, or authority
ConfigMap; its durable identity is exactly namespace, job ID, and run ID.

The chart exposes the API as a ClusterIP Service and stores published results
on a mandatory single-replica PVC. Partial uploads remain under a private
staging directory and are not readable. Only an exact declared set can be
atomically published; the sidecar exits after the durable manifest response so
the Job can become terminal. AIPerfJob deletion cleans workload resources but
does not release published PVC results; the index rebuilds them on operator
restart. `aiperf kube results` reaches persisted results through the Kubernetes
Service proxy using trusted local `--operator-service`/`--operator-namespace`
configuration and an explicit `--run-id`. Workload-controlled annotations
never select that proxy target.

The chart's default upload transport is cluster-local HTTP and assumes trusted
namespace-level in-cluster access. It does not provide confidentiality or
traffic-analysis protection. Clusters that require confidentiality must enforce
it with a service mesh or equivalent network layer and should restrict access to
the operator Service with NetworkPolicy. No external Ingress is shipped.
