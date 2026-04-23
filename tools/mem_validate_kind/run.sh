#!/usr/bin/env bash
# Validate AIPerf worker-layout memory savings against a real kind cluster.
#
# Pipeline:
#   1. Ensure a kind cluster exists (reuse or create).
#   2. Build the probe image (aiperf-slim + transformers/tokenizers + probe.py).
#   3. Load the image into kind.
#   4. Apply two pods: N-container mode, and 1-container-with-N-children mode.
#   5. Wait for every worker-shaped process to signal readiness via /shared.
#   6. Write /shared/GO into both pods to trigger synchronized PSS snapshots.
#   7. Collect snapshots, plus each container's cgroup memory.current.
#   8. Print a side-by-side comparison.
#
# Defaults to N=10 children; override with N=16 ./run.sh etc.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
CLUSTER_NAME="${CLUSTER_NAME:-aiperf-mem-validate}"
N_CHILDREN="${N:-10}"
IMAGE="localhost/aiperf-mem-probe:latest"
BASE_IMAGE="aiperf-slim:amd64-final"
OUT="${HERE}/results"
mkdir -p "${OUT}"

log() { printf '\033[1;36m[%s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*"; }
err() { printf '\033[1;31m[%s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*" >&2; }

step_cluster() {
  if kind get clusters 2>/dev/null | grep -qx "${CLUSTER_NAME}"; then
    log "kind cluster '${CLUSTER_NAME}' already exists"
  else
    log "creating kind cluster '${CLUSTER_NAME}'"
    kind create cluster --name "${CLUSTER_NAME}"
  fi
  kubectl --context "kind-${CLUSTER_NAME}" cluster-info >/dev/null
  KUBE_CTX="kind-${CLUSTER_NAME}"
}

step_image() {
  if ! docker image inspect "${BASE_IMAGE}" >/dev/null 2>&1; then
    err "base image ${BASE_IMAGE} missing; build it first from deploy/Dockerfile.aiperf-slim"
    exit 2
  fi
  log "building probe image ${IMAGE}"
  docker build -t "${IMAGE}" -f "${HERE}/Dockerfile" "${HERE}" 2>&1 | tail -40
  log "loading ${IMAGE} into kind cluster"
  kind load docker-image "${IMAGE}" --name "${CLUSTER_NAME}"
}

step_apply() {
  KIND_NODE="$(kubectl --context "${KUBE_CTX}" get nodes -o jsonpath='{.items[0].metadata.name}')"
  log "target node: ${KIND_NODE}"

  # Build the N-container block for the containers-mode manifest.
  WORKER_CONTAINERS=""
  for i in $(seq -f "%02g" 1 "${N_CHILDREN}"); do
    WORKER_CONTAINERS+="$(cat <<EOF
    - name: w${i}
      image: ${IMAGE}
      imagePullPolicy: Never
      args:
        - worker
        - --label
        - c-w${i}
        - --tokenizers
        - Qwen/Qwen3-0.6B
        - openai/gpt-oss-120b
      env:
        - name: HF_HOME
          value: /hf_cache
        - name: TRANSFORMERS_CACHE
          value: /hf_cache
        - name: PYTHONUNBUFFERED
          value: "1"
      resources:
        requests:
          cpu: "100m"
          memory: "256Mi"
      volumeMounts:
        - name: shared
          mountPath: /shared
        - name: hf-cache
          mountPath: /hf_cache
EOF
)"
    WORKER_CONTAINERS+=$'\n'
  done

  export KIND_NODE N_CHILDREN WORKER_CONTAINERS

  # Render manifests via plain envsubst-style bash substitution.
  for tpl in "${HERE}/manifests/pod-containers.yaml.tpl" \
             "${HERE}/manifests/pod-forkserver.yaml.tpl" \
             "${HERE}/manifests/pod-mp-forkserver.yaml.tpl"; do
    out="${tpl%.tpl}"
    python3 -c "
import os, sys
s = open(sys.argv[1]).read()
for k in ('KIND_NODE', 'N_CHILDREN', 'WORKER_CONTAINERS'):
    s = s.replace('\${' + k + '}', os.environ.get(k, ''))
open(sys.argv[2], 'w').write(s)
" "${tpl}" "${out}"
  done

  log "deleting any previous pods"
  kubectl --context "${KUBE_CTX}" delete pod mem-containers mem-forkserver mem-mp-forkserver --ignore-not-found --wait=true

  log "applying containers-mode pod"
  kubectl --context "${KUBE_CTX}" apply -f "${HERE}/manifests/pod-containers.yaml"
  log "applying forkserver-mode pod (os.fork)"
  kubectl --context "${KUBE_CTX}" apply -f "${HERE}/manifests/pod-forkserver.yaml"
  log "applying mp-forkserver-mode pod (multiprocessing.set_forkserver_preload)"
  kubectl --context "${KUBE_CTX}" apply -f "${HERE}/manifests/pod-mp-forkserver.yaml"
}

step_wait_ready() {
  log "waiting for pods to be Running"
  kubectl --context "${KUBE_CTX}" wait --for=condition=Ready pod/mem-containers --timeout=600s || true
  kubectl --context "${KUBE_CTX}" wait --for=condition=Ready pod/mem-forkserver --timeout=600s || true
  kubectl --context "${KUBE_CTX}" wait --for=condition=Ready pod/mem-mp-forkserver --timeout=600s || true

  # Now wait for the per-process ready markers so we know every worker has
  # finished loading tokenizers and hit its pre-GO snapshot.
  log "waiting for ${N_CHILDREN} worker-ready markers in mem-containers"
  for i in $(seq -f "%02g" 1 "${N_CHILDREN}"); do
    local_tries=0
    until kubectl --context "${KUBE_CTX}" exec -c "w${i}" mem-containers -- \
      test -f "/shared/c-w${i}.ready" 2>/dev/null; do
      local_tries=$((local_tries + 1))
      if [[ ${local_tries} -gt 300 ]]; then
        err "w${i} never became ready"
        kubectl --context "${KUBE_CTX}" logs -c "w${i}" mem-containers | tail -30 >&2
        exit 3
      fi
      sleep 2
    done
  done

  log "waiting for forkserver (os.fork) parent + ${N_CHILDREN} children ready markers"
  local_tries=0
  until kubectl --context "${KUBE_CTX}" exec mem-forkserver -- \
    test -f "/shared/fs-parent.ready" 2>/dev/null; do
    local_tries=$((local_tries + 1))
    if [[ ${local_tries} -gt 300 ]]; then
      err "forkserver parent never became ready"
      kubectl --context "${KUBE_CTX}" logs mem-forkserver | tail -50 >&2
      exit 3
    fi
    sleep 2
  done
  for i in $(seq -f "%02g" 0 $(($N_CHILDREN - 1))); do
    until kubectl --context "${KUBE_CTX}" exec mem-forkserver -- \
      test -f "/shared/fs-child-${i}.ready" 2>/dev/null; do
      sleep 2
    done
  done

  log "waiting for mp-forkserver parent + ${N_CHILDREN} children ready markers"
  local_tries=0
  until kubectl --context "${KUBE_CTX}" exec mem-mp-forkserver -- \
    test -f "/shared/mp-mp-parent.ready" 2>/dev/null; do
    local_tries=$((local_tries + 1))
    if [[ ${local_tries} -gt 300 ]]; then
      err "mp-forkserver parent never became ready"
      kubectl --context "${KUBE_CTX}" logs mem-mp-forkserver | tail -80 >&2
      exit 3
    fi
    sleep 2
  done
  for i in $(seq -f "%02g" 0 $(($N_CHILDREN - 1))); do
    until kubectl --context "${KUBE_CTX}" exec mem-mp-forkserver -- \
      test -f "/shared/mp-mp-child-${i}.ready" 2>/dev/null; do
      sleep 2
    done
  done

  log "all workers and fork children ready; settling 3s before synchronized snapshot"
  sleep 3
}

step_go() {
  log "firing synchronized GO into all three pods"
  kubectl --context "${KUBE_CTX}" exec mem-containers -c w01 -- touch /shared/GO
  kubectl --context "${KUBE_CTX}" exec mem-forkserver -- touch /shared/GO
  kubectl --context "${KUBE_CTX}" exec mem-mp-forkserver -- touch /shared/GO
  # Let every process write its post-GO snapshot.
  sleep 5
}

step_collect() {
  rm -rf "${OUT}/containers" "${OUT}/forkserver" "${OUT}/mp-forkserver"
  mkdir -p "${OUT}/containers" "${OUT}/forkserver" "${OUT}/mp-forkserver"

  log "collecting snapshots from mem-containers"
  kubectl --context "${KUBE_CTX}" cp -c w01 \
    "mem-containers:/shared/" "${OUT}/containers/" >/dev/null
  log "collecting snapshots from mem-forkserver (os.fork)"
  kubectl --context "${KUBE_CTX}" cp \
    "mem-forkserver:/shared/" "${OUT}/forkserver/" >/dev/null
  log "collecting snapshots from mem-mp-forkserver (multiprocessing.set_forkserver_preload)"
  kubectl --context "${KUBE_CTX}" cp \
    "mem-mp-forkserver:/shared/" "${OUT}/mp-forkserver/" >/dev/null

  log "collecting cgroup memory.current from each container"
  for i in $(seq -f "%02g" 1 "${N_CHILDREN}"); do
    kubectl --context "${KUBE_CTX}" exec mem-containers -c "w${i}" -- \
      cat /sys/fs/cgroup/memory.current > "${OUT}/containers/cgroup-w${i}.txt" 2>/dev/null || true
  done
  kubectl --context "${KUBE_CTX}" exec mem-forkserver -- \
    cat /sys/fs/cgroup/memory.current > "${OUT}/forkserver/cgroup-forkserver.txt" 2>/dev/null || true
  kubectl --context "${KUBE_CTX}" exec mem-mp-forkserver -- \
    cat /sys/fs/cgroup/memory.current > "${OUT}/mp-forkserver/cgroup-forkserver.txt" 2>/dev/null || true
}

step_report() {
  log "results → ${OUT}"
  python3 "${HERE}/report.py" \
    --containers-dir "${OUT}/containers" \
    --forkserver-dir "${OUT}/forkserver" \
    --mp-forkserver-dir "${OUT}/mp-forkserver" \
    --n "${N_CHILDREN}"
}

main() {
  log "=== AIPerf memory-layout validation on kind ==="
  log "cluster=${CLUSTER_NAME} N=${N_CHILDREN} image=${IMAGE}"
  step_cluster
  step_image
  step_apply
  step_wait_ready
  step_go
  step_collect
  step_report
}

main "$@"
