# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.orchestrator.executor import RunExecutor


def test_run_executor_is_abstract():
    with pytest.raises(TypeError):
        RunExecutor()  # cannot instantiate ABC
