# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from tests.component_integration.conftest import (
    ComponentIntegrationTestDefaults as defaults,
)
from tests.harness.utils import AIPerfCLI


@pytest.mark.component_integration
class TestDefaultBehavior:
    """Tests for default behavior."""

    def test_default_behavior(self, cli: AIPerfCLI):
        """Test that only providing the model and nothing else still works.

        NOTE: The artifact directory will still be modified by the test-runner
        """
        result = cli.run_sync(
            f"""
            aiperf profile --model {defaults.model}
            """
        )
        assert result.request_count == defaults.request_count
