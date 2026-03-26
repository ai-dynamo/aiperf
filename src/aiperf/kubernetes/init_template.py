# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Config template generator for Kubernetes deployments.

Provides an AIPerfJob CR template that users can customize
for their benchmark setup.
"""

KUBE_INIT_TEMPLATE = """\
# AIPerf Kubernetes Benchmark - AIPerfJob Custom Resource
#
# Usage (CLI):
#   aiperf kube profile --config {filename} --image <your-image>
#
# Usage (GitOps / operator):
#   kubectl apply -f {filename}
#
# This file defines an AIPerfJob CR. When using the CLI, --image and other
# Kubernetes flags are still required; benchmark config comes from this file.

apiVersion: aiperf.nvidia.com/v1alpha1
kind: AIPerfJob
metadata:
  name: my-benchmark
spec:
  # === Benchmark Configuration ===
  benchmark:
    # Model name(s) served by the endpoint
    models:
      - "your-model-name"

    # Endpoint to benchmark (list of URLs)
    endpoint:
      urls:
        - "http://your-server:8000"
      streaming: true

    # Dataset configuration
    datasets:
      main:
        type: synthetic
        entries: 1000
        prompts:
          isl:
            mean: 512
            stddev: 0
          osl:
            mean: 128
            stddev: 0

    # Load phases
    phases:
      warmup:
        type: concurrency
        concurrency: 10
        requests: 10
        exclude_from_results: true
      profiling:
        type: concurrency
        concurrency: 50
        requests: 500

  # === Deployment Options ===
  # ttlSecondsAfterFinished: 300
  # timeoutSeconds: 0
  # resourceMode: guaranteed  # guaranteed (requests==limits), burstable (requests only), none (omit all)

  # === Pod Customization ===
  # podTemplate:
  #   nodeSelector:
  #     nvidia.com/gpu.product: "A100"
  #   tolerations:
  #     - key: nvidia.com/gpu
  #       operator: Exists
  #       effect: NoSchedule
  #   imagePullSecrets:
  #     - my-registry-secret
  #   env:
  #     - name: AIPERF_HTTP_CONNECTION_LIMIT
  #       value: "200"
  #   volumes:
  #     - name: model-cache
  #       persistentVolumeClaim:
  #         claimName: model-cache
  #   volumeMounts:
  #     - name: model-cache
  #       mountPath: /root/.cache/huggingface

  # === Kueue Scheduling ===
  # scheduling:
  #   queueName: my-queue
  #   priorityClass: high-priority
"""


def generate_init_template(filename: str) -> str:
    """Generate a config template with the given filename in usage instructions.

    Args:
        filename: The filename to substitute into usage instructions.

    Returns:
        The template string with filename substituted.
    """
    return KUBE_INIT_TEMPLATE.format(filename=filename)
