# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Config template generator for Kubernetes deployments.

Wraps any AIPerf benchmark-config YAML body (e.g. one of the bundled templates
in ``src/aiperf/config/templates/``) in an ``AIPerfJob`` CR shell so users can
``kubectl apply`` or feed it to ``aiperf kube profile --config``.
"""

from __future__ import annotations

import textwrap

_HEADER = """\
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
  name: {job_name}
spec:
  benchmark:
"""

_FOOTER = """\

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


def _strip_leading_meta_headers(content: str) -> str:
    """Drop leading ``# yaml-language-server`` / ``# @template`` metadata blocks.

    Bundled templates carry editor/schema hints and a ``# @template`` metadata
    block at the top that are irrelevant (and misleading) once wrapped under
    ``spec.benchmark``. We strip until the first blank line or first
    non-metadata content line.
    """
    lines = content.splitlines(keepends=True)
    out: list[str] = []
    skipping = True
    for line in lines:
        if skipping:
            stripped = line.strip()
            if stripped.startswith("# yaml-language-server"):
                continue
            if stripped.startswith("# @template"):
                continue
            if stripped.startswith("#") and ": " in stripped[2:]:
                # @template metadata key/value line — skip
                continue
            skipping = False
        out.append(line)
    return "".join(out)


def wrap_as_aiperf_job(
    benchmark_body: str,
    *,
    filename: str = "benchmark.yaml",
    job_name: str = "my-benchmark",
) -> str:
    """Wrap an AIPerf benchmark config body in an AIPerfJob CR.

    Args:
        benchmark_body: YAML content of an AIPerf benchmark config (top-level
            keys like ``model``, ``endpoint``, ``dataset``, ``phases``).
            SPDX headers should already be stripped by the caller; this
            function additionally strips yaml-language-server and
            ``# @template`` metadata blocks.
        filename: Filename used in the usage-instruction comments.
        job_name: Value for ``metadata.name`` on the generated CR.

    Returns:
        A complete AIPerfJob YAML document with the body indented under
        ``spec.benchmark`` and the standard deployment-options / pod /
        scheduling commented blocks appended.
    """
    cleaned = _strip_leading_meta_headers(benchmark_body).rstrip("\n")
    indented = textwrap.indent(cleaned, "    ")
    return (
        _HEADER.format(filename=filename, job_name=job_name) + indented + "\n" + _FOOTER
    )
