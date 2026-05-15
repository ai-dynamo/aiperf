# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned dependency versions shared by dev/kube.py and test infrastructure.

Single source of truth for version strings that must stay in sync across the
dev CLI and the Kubernetes E2E test fixtures.
"""

JOBSET_VERSION = "v0.8.0"
DEVICE_PLUGIN_VERSION = "v0.17.0"
DYNAMO_VERSION = "0.9.0"
KUEUE_VERSION = "v0.10.1"

JOBSET_CRD_URL_TEMPLATE = (
    "https://github.com/kubernetes-sigs/jobset/releases/download"
    "/{version}/manifests.yaml"
)
KUEUE_MANIFEST_URL_TEMPLATE = (
    "https://github.com/kubernetes-sigs/kueue/releases/download"
    "/{version}/manifests.yaml"
)
