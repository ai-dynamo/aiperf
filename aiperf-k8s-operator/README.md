# AIPerf Kubernetes Operator

`aiperf-k8s-operator` is the independent reconciliation and durable-results
distribution for the native AIPerf Kubernetes contract. It does not import the
`aiperf` Python package. Its only compatibility boundary is
`contracts/native-k8s/v1`.

The process supervises Kopf reconciliation and the FastAPI service on port
8080. The Helm chart installs its pinned JobSet dependency, a fixed
`aiperf-k8s-operator` ClusterIP Service, and a PVC for atomically published
manifest-authorized results.
