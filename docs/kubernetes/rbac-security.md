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
| core | `secrets` | `create, delete, get` | Read exact bootstrap references and own the dedicated results-read capability |
| core | `configmaps` | `create, delete, get` | Read source configuration and own immutable config/result-authority snapshots |
| core | `serviceaccounts` | `create, delete, get` | Own the exact run identity |
| `rbac.authorization.k8s.io` | `roles`, `rolebindings` | `create, delete, get` | Bind and remove the exact run identity |

The embedded Kopf runtime is explicitly cluster-wide, standalone, and has
Kubernetes Event posting disabled. It therefore does not require Event or
peering-object permissions beyond the table above.

There is no shipped AIPerfSweep resource or sweep-handler identity. The
operator cannot list, watch, update, or patch Secrets. Kubernetes returns the
complete Secret for an exact-name `get`; the operator validates every
reference's immutable metadata, digest annotation, and exact CR-UID owner
reference. It reads only the
results-sidecar's `data.bootstrap`, verifies those bytes against the envelope
digest, derives an object-incarnation-bound Ed25519 public verifier, and
discards the private bytes after reconciliation. It also creates one distinct
immutable 32-byte results-read capability Secret. That capability is random;
it is not role bootstrap material and cannot be used for cellular admission.
Private bootstrap material is never written into the AIPerfJob, JobSet, logs,
result-authority record, or results store.

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

The AIPerfJob's exact UID is a non-blocking owner reference on its bootstrap
Secrets, JobSet, ServiceAccount, Role, RoleBinding, configuration snapshot,
read-capability Secret, and authority ConfigMap. These references do not require
delete-on-owner or owner-finalizer admission authority. A delete handler
explicitly removes the complete set; Kubernetes garbage collection is only a
backstop. A name-reused AIPerfJob cannot inherit authority from an earlier
object incarnation.

The source ConfigMap is user-owned. `configRef.sha256` covers canonical JSON of
its complete `data` and `binaryData` maps. Reconciliation verifies that digest
and creates an immutable, UID-owned snapshot before JobSet creation. The JobSet
mounts the snapshot name, so deleting or replacing the source cannot change an
accepted workload.

## Durable results boundary

The controller and regular results-sidecar container share a writable
`emptyDir` only within their Job pod. The sidecar validates the committed
manifest, uploads each exact declared artifact, then publishes the manifest to
the operator API. Each request is signed by a key derived from the private
sidecar bootstrap and bound to namespace, job, run, object kind, path, SHA-256,
and byte length, including the AIPerfJob UID.

The operator accepts the JobSet first with a required authority-ConfigMap
startup gate, then re-reads the AIPerfJob UID, immutable envelope, and bootstrap
references. Only after that revalidation does it create the immutable authority
ConfigMap containing the upload public key and the read-capability hash. The
ConfigMap contains no private capability. Upload authorization recomputes the
expected verifier from the exact immutable bootstrap Secret and current CR UID
and validates the complete ConfigMap identity before accepting a request.

Completed-result reads require the dedicated bearer capability. A trusted
caller obtains only the dedicated results-read Secret through Kubernetes
authorization; it never reads a controller, cell, or results-sidecar bootstrap
Secret. The API validates the bearer against the ConfigMap hash and the exact
namespace, job, run, and object UID before serving a manifest or artifact.

The chart exposes the API as a ClusterIP Service and stores published results
on a mandatory single-replica PVC. Partial uploads remain under a private
staging directory and are not readable. Only an exact declared set can be
atomically published; the sidecar exits after the durable manifest response so
the Job can become terminal. `aiperf kube results` reaches persisted results
through the Kubernetes Service proxy using trusted local
`--operator-service`/`--operator-namespace` configuration. Workload-controlled
annotations never select that privileged proxy target.

The chart's default upload transport is cluster-local HTTP. Signatures provide
request authentication and content integrity, not confidentiality or traffic
analysis protection. Clusters that require confidentiality must enforce it
with a service mesh or equivalent network layer and should restrict access to
the operator Service with NetworkPolicy. No external Ingress or dashboard is
shipped.
