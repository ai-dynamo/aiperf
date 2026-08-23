# AIPerf Kubernetes Operator

`aiperf-k8s-operator` is the independent reconciliation and durable-results
distribution for the native AIPerf Kubernetes contract. It does not import the
`aiperf` Python package. Its only compatibility boundary is
`contracts/native-k8s/v1`.

The process supervises Kopf reconciliation and the FastAPI service on port
8080. The Helm chart requires an explicit operator image repository and tag,
installs its pinned JobSet dependency, a fixed `aiperf-k8s-operator` ClusterIP
Service, and a single-owner PVC for atomically published results.

The controller and regular results sidecar share a pod-local `emptyDir`. The
sidecar streams bounded artifacts with SHA-256 and length metadata to the
cluster-local Service, then uploads the manifest last. The service stages the
bytes, atomically publishes the exact manifest-declared set on the PVC, and
acknowledges the sidecar only after that durable commit. Results are retained
under the namespace/job/run triple and are retrievable through `aiperf kube
results <job> --namespace <ns> --run-id <run>` via the fixed Kubernetes Service
proxy after producer deletion or operator restart.

This is a trusted namespace-level in-cluster service, not an application
authentication protocol: it has no upload signatures, tokens, public keys,
results-read Secrets, authority ConfigMaps, or external ingress. Kubernetes
authentication and RBAC protect the off-cluster Service-proxy hop. Controller
and cell bootstraps remain separate cellular-execution material.
