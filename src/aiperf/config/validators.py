# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Before-validators for the AIPerf CLI configuration.

All functions in this module operate on raw input dicts (before Pydantic model construction).
They handle data normalization and default-setting that must happen before field validation.
"""

import sys
from typing import Any

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.enums import AIPerfLogLevel
from aiperf.config.parsing import coerce_value
from aiperf.plugin.enums import UIType

_logger = AIPerfLogger(__name__)


def _should_quote_arg(x: Any) -> bool:
    """Determine if the value should be quoted in the CLI command."""
    return isinstance(x, str) and not x.startswith("-") and x not in ("profile")


def set_log_level_from_verbose_flags(data: dict[str, Any]) -> None:
    """Set log level based on presence of verbose flags.

    If the user selected verbose or extra verbose flags and the log level is not explicitly set,
    set the log level to TRACE or DEBUG respectively.
    """
    if data.get("extra_verbose"):
        data.setdefault("log_level", AIPerfLogLevel.TRACE)
    elif data.get("verbose"):
        data.setdefault("log_level", AIPerfLogLevel.DEBUG)


def set_ui_type_from_verbose_flags(data: dict[str, Any]) -> None:
    """Set UI type based on presence of verbose flags.

    If the user selected verbose or extra verbose flags and the UI type is not explicitly set,
    set the UI type to SIMPLE. This will allow the user to see the verbose output in the console easier.
    """
    if data.get("ui_type") is None and (
        data.get("verbose") or data.get("extra_verbose")
    ):
        data["ui_type"] = UIType.SIMPLE


def set_cli_command(data: dict[str, Any]) -> None:
    """Set the CLI command from sys.argv if not already provided."""
    if not data.get("cli_command"):
        args = [coerce_value(x) for x in sys.argv[1:]]
        args = [f"'{x}'" if _should_quote_arg(x) else str(x) for x in args]
        data["cli_command"] = " ".join(["aiperf", *args])


def set_benchmark_id(data: dict[str, Any]) -> None:
    """Generate a unique benchmark ID if not already set.

    This ID is shared across all export formats (JSON, CSV, Parquet, etc.)
    to enable correlation of data from the same benchmark run.
    """
    if data.get("benchmark_id"):
        import uuid

        data["benchmark_id"] = str(uuid.uuid4())


def normalize_streaming_for_endpoint(data: dict[str, Any]) -> None:
    """Disable streaming if the endpoint type does not support it.

    Checks plugin metadata for the selected endpoint type and silently
    disables streaming with a warning when unsupported.
    """
    if not data.get("streaming"):
        return

    from aiperf.config.defaults import EndpointDefaults
    from aiperf.plugin import plugins

    endpoint_type = data.get("endpoint_type", EndpointDefaults.TYPE)
    metadata = plugins.get_endpoint_metadata(endpoint_type)
    if not metadata.supports_streaming:
        _logger.warning(
            f"Streaming is not supported for --endpoint-type {endpoint_type}, setting streaming to False"
        )
        data["streaming"] = False
