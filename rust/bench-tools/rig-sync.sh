#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Sync this worktree's `rust/` crates to the c4-standard-144 benchmark pod and
# (optionally) build them there. rsync is not installed in the pod, so the
# transfer is a tar stream over `kubectl exec`.
#
#   rig-sync.sh            sync only
#   rig-sync.sh build      sync, then release-build aiperf + aiperf-mock-server
set -euo pipefail

CTX=${RIG_CONTEXT:-nv-prd-dgxc.teleport.sh-dynamo-gcp-dev-02}
NS=${RIG_NAMESPACE:-acasagrande-aiperf-bench}
POD=${RIG_POD:-paper-rig}
CONTAINER=${RIG_CONTAINER:-bench}
REMOTE=${RIG_REMOTE:-/work/src/repo}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

K=(kubectl --context "$CTX" -n "$NS")

echo "==> syncing ${REPO_ROOT}/rust -> ${POD}:${REMOTE}/rust"
tar -C "$REPO_ROOT" \
    --exclude='rust/target' \
    --exclude='rust/artifacts' \
    --exclude='*.log' \
    -czf - rust \
  | "${K[@]}" exec -i "$POD" -c "$CONTAINER" -- \
      bash -c "mkdir -p '$REMOTE' && tar -C '$REMOTE' -xzf -"

if [[ ${1:-} == build ]]; then
  echo "==> building on the rig (144 cores)"
  "${K[@]}" exec "$POD" -c "$CONTAINER" -- bash -lc "
    set -euo pipefail
    export CARGO_HOME=/work/cargo RUSTUP_HOME=/work/rustup PATH=/work/cargo/bin:\$PATH
    export CARGO_TARGET_DIR=/work/target
    cd '$REMOTE/rust'
    time cargo build --release -p aiperf-cli -p aiperf-mock-server 2>&1 | tail -25
  "
fi
