# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mooncake-trace tests for CLIConfig.

The previous suite asserted two private CLIConfig methods:

* ``_count_dataset_entries()`` - line-counted the trace file to derive the
  effective request_count when the user didn't pass one.
* ``_should_use_fixed_schedule_for_trace_dataset()`` - peeked at the trace
  file and auto-enabled fixed_schedule when timestamps were present.

Both methods were removed in the v2 refactor: the v1 CLIConfig is now a
pure CLI-input DTO (no methods, no validators - see
``aiperf.config.flags.cli_config`` module docstring). The behaviors moved
into AIPerfConfig / the v1->v2 converter:

* request-count derivation: handled by AIPerfConfig defaults + the dataset
  manager at runtime.
* fixed-schedule auto-detection on mooncake_trace: handled by the
  converter (`aiperf.config.flags._converter_dataset`) and the trace dataset
  loader.

Those tests have been removed because the methods under test no longer
attach to the class under test. End-to-end coverage of the auto-detection
behavior lives in the dataset / converter test modules.
"""

from aiperf.config.flags.cli_config import CLIConfig


def test_cli_config_with_loadgen_request_count_default_dto_value() -> None:
    """The v1 LoadGeneratorConfig DTO leaves request_count unset (None) by default."""
    config = CLIConfig(
        model_names=["test-model"], **CLIConfig().model_dump(exclude_unset=True)
    )
    assert config.request_count is None


def test_cli_config_with_explicit_loadgen_request_count() -> None:
    """Explicit request_count is preserved on the DTO."""
    config = CLIConfig(
        model_names=["test-model"],
        **CLIConfig(request_count=42).model_dump(exclude_unset=True),
    )
    assert config.request_count == 42
