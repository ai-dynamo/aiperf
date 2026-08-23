---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Results Sidecar
---

# Native Kubernetes Results Sidecar

Every `native-k8s/v1` controller Job pod contains the controller and one regular
`aiperf results-sidecar` container. Both mount the same writable results
`emptyDir` at the envelope's `artifactRoot`; it is not shared with the operator
pod. Cell pods do not mount results and do not receive Kubernetes API tokens.

The controller atomically writes and fsyncs `results-manifest.json` only after
its declared artifacts are complete. The sidecar's port 9091 exposes health
only; result bytes are never published directly from the workload pod:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/healthz` | Liveness |

The sidecar retains a descriptor for the results root and opens every manifest
and artifact with no-follow semantics. Non-POSIX
platforms fail closed. Malformed manifests, noncanonical or duplicate paths,
control characters in response metadata, ancestor or leaf symlinks, special
files, missing files, and digest or length mismatches fail before upload.

## Durable publication and Job completion

The sidecar validates the complete local declared set and streams each artifact
to the fixed operator API before uploading the manifest. Upload metadata carries
the SHA-256 digest and byte length; it is validated with the received bytes and
is not an application credential. The sidecar has no bootstrap, token,
signature, public key, or AIPerfJob-UID-incarnation result identity.

Artifact uploads remain private staging state. The operator publishes a run
only when the manifest names exactly the staged set and every byte still
matches; publication is atomic and fsynced onto the chart's results PVC.
Byte-identical artifact and manifest replays return success, while conflicting
replays fail, and replay after publication never recreates staging state.
Staging is bounded by run, byte, and artifact quotas; abandoned state, including
partial temporary bytes left by a restart, is counted and expires. Wrong-path,
oversized, or incomplete transactions never become readable. Durable storage and
reads are keyed by namespace, job ID, and run ID rather than run ID alone.

Network failures, HTTP 429, and server errors are retried within a finite
ten-minute budget; each attempt has a 30-second timeout. Contract and content
conflicts fail the sidecar. Waiting for a manifest is
also bounded (30 minutes by default): expiry or an exhausted upload budget makes
the sidecar exit nonzero, so a missing manifest cannot leave the Job running
forever. After a 200/201 response for the durable manifest, the regular sidecar
stops its health server and exits successfully. This lets the controller Job
reach a terminal state without discarding post-exit result retrieval.

The default upload URL is the fixed chart Service
`http://aiperf-k8s-operator.<operator-namespace>.svc:8080`. This is cluster-local
HTTP and assumes trusted namespace-level in-cluster access; it does not encrypt
result content. Use a service mesh or equivalent network layer where transport
confidentiality is required.

## Retrieving completed results

`aiperf kube results <job> --run-id <run>` uses only the Kubernetes Service
proxy for the trusted local operator identity. Defaults are
`--operator-service aiperf-k8s-operator` and `--operator-namespace
aiperf-system`; set them explicitly for a nonstandard chart service or
namespace. The CLI ignores workload-controlled endpoint annotations. It goes
directly to the fixed Service-proxy prefix for the supplied namespace, job, and
trusted run ID; it does not fetch an AIPerfJob or Secret before download.
Kubernetes authentication and RBAC protect the API-server Service-proxy hop.
The operator does not ship an external ingress or application-level read token.
