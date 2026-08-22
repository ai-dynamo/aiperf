# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest


@pytest.mark.skipif(not os.environ.get("KUBECONFIG"), reason="requires: kind, helm, and KUBECONFIG")
def test_chart_installation_requires_kind_helm_and_kubeconfig() -> None:
    """requires: kind, helm, and KUBECONFIG."""
    pytest.skip("kind integration is executed by the dedicated CI workflow")
