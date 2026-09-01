# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Config template generator for Kubernetes deployments.

Wraps any AIPerf benchmark-config YAML body (e.g. one of the bundled templates
in ``src/aiperf/config/templates/``) in an ``AIPerfJob`` CR shell so users can
``kubectl apply`` or feed it to ``aiperf kube profile --config``.

``AIPerfJobSpec`` extends ``AIPerfWorkloadSpec(AIPerfConfig,
DeploymentConfig)`` -- so ``spec.*`` IS the AIPerfConfig envelope; the
``benchmark`` body lives at ``spec.benchmark``. The wrap is shape-aware:

* Envelope-shape input (top-level ``benchmark:`` key) -- the body is
  indented directly under ``spec:``, preserving envelope-level keys
  (``benchmark``, ``model``/``models``, ``dataset``/``datasets``,
  ``variables``, ``random_seed``) at their correct depth.
* Flat-shape input (no top-level ``benchmark:``) -- the entire body is
  indented under ``spec.benchmark:``, the legacy behaviour. Bundled
  templates are all envelope-shape; this branch is retained for ad-hoc
  benchmark snippets and for the test corpus.
"""

from __future__ import annotations

import textwrap

import yaml as _yaml

_HEADER_ENVELOPE = """\
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
"""

_HEADER_FLAT = """\
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
  # resourceMode: burstable  # burstable (requests only, default), guaranteed (requests==limits), none (omit all)

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


def _is_envelope_shape(content: str) -> bool:
    """Return True if ``content`` parses to a dict with a top-level ``benchmark`` key.

    Envelope-shape templates put the swept body under ``benchmark:`` and keep
    cross-variation envelope keys (``variables``, ``random_seed``, ``sweep``,
    ``multi_run``) plus shorthand aliases (``model``, ``dataset``) at the top
    level. Bundled AIPerf templates are all envelope-shape.

    Returns False on parse failure (treated as flat for wrap purposes; the
    apiserver will reject the resulting CR with a more specific error).
    """
    try:
        parsed = _yaml.safe_load(content)
    except _yaml.YAMLError:
        return False
    return isinstance(parsed, dict) and "benchmark" in parsed


def wrap_as_aiperf_job(
    benchmark_body: str,
    *,
    filename: str = "benchmark.yaml",
    job_name: str = "my-benchmark",
) -> str:
    """Wrap an AIPerf benchmark config body in an AIPerfJob CR.

    Args:
        benchmark_body: YAML content of an AIPerf benchmark config. Either
            envelope-shape (top-level ``benchmark:`` key plus optional
            ``model``/``dataset``/``variables``/``random_seed``/``sweep``/
            ``multi_run``) or flat-shape (body fields like ``model``,
            ``endpoint``, ``dataset``, ``phases`` at the top with no
            ``benchmark:`` wrapper). SPDX headers should already be stripped
            by the caller; this function additionally strips
            yaml-language-server and ``# @template`` metadata blocks.
        filename: Filename used in the usage-instruction comments.
        job_name: Value for ``metadata.name`` on the generated CR.

    Returns:
        A complete AIPerfJob YAML document. Envelope-shape input is indented
        directly under ``spec:`` (so ``spec.benchmark`` carries the body and
        envelope-level keys land at ``spec.<key>``). Flat-shape input is
        indented under ``spec.benchmark:`` for back-compat. The standard
        deployment-options / pod / scheduling commented blocks are appended
        to both shapes.
    """
    cleaned = _strip_leading_meta_headers(benchmark_body).rstrip("\n")
    if _is_envelope_shape(cleaned):
        # Envelope-shape: body keys (including the inner ``benchmark:``)
        # live one level below ``spec:``, so two-space indent is correct.
        indented = textwrap.indent(cleaned, "  ")
        header = _HEADER_ENVELOPE
    else:
        # Flat-shape: every body key lives under ``spec.benchmark:``, so
        # we need four-space indent (two for spec, two for benchmark).
        indented = textwrap.indent(cleaned, "    ")
        header = _HEADER_FLAT
    return (
        header.format(filename=filename, job_name=job_name) + indented + "\n" + _FOOTER
    )
