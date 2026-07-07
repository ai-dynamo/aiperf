# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import contextlib
from dataclasses import dataclass
from typing import ClassVar

import orjson
from rich.console import Console
from rich.panel import Panel

from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models import ErrorDetailsCount
from aiperf.exporters.exporter_config import ExporterConfig


@dataclass
class ErrorInsight:
    """Model describing a detected API error insight."""

    title: str
    problem: str
    causes: list[str]
    investigation: list[str]
    fixes: list[str]


class MaxCompletionTokensDetector:
    @staticmethod
    def detect(error_summary: list[ErrorDetailsCount]) -> ErrorInsight | None:
        if not error_summary or not isinstance(error_summary, list):
            return None

        for item in error_summary:
            err = getattr(item, "error_details", None)
            if err is None:
                continue

            raw_msg = err.message or ""
            parsed = None
            with contextlib.suppress(Exception):
                parsed = orjson.loads(raw_msg)

            backend_msg = None
            if isinstance(parsed, dict):
                backend_msg = parsed.get("message")

            error_blob = str(backend_msg or raw_msg)

            if (
                "extra_forbidden" in error_blob
                and "max_completion_tokens" in error_blob
                and "Extra inputs are not permitted" in error_blob
            ):
                return ErrorInsight(
                    title="Unsupported Parameter: max_completion_tokens",
                    problem=(
                        "The backend rejected 'max_completion_tokens'. "
                        "This backend only supports 'max_tokens'."
                    ),
                    causes=[
                        "AIPerf generated 'max_completion_tokens' due to --output-tokens-mean.",
                        "The backend rejects 'max_completion_tokens' because it only supports 'max_tokens'.",
                    ],
                    investigation=[
                        "Inspect request payloads in profile_export.jsonl.",
                        "Check the backend's supported parameters.",
                    ],
                    fixes=[
                        "Remove --output-tokens-mean.",
                        'Or use --extra-inputs "max_tokens:<value>".',
                        "Or run AIPerf with '--use-legacy-max-tokens' to force use of the legacy 'max_tokens' field instead of 'max_completion_tokens'.",
                    ],
                )

        return None


class DynamoSessionControlDetector:
    @staticmethod
    def detect(error_summary: list[ErrorDetailsCount]) -> ErrorInsight | None:
        if not error_summary or not isinstance(error_summary, list):
            return None

        for item in error_summary:
            err = getattr(item, "error_details", None)
            if err is None:
                continue

            raw_msg = err.message or ""
            parsed = None
            with contextlib.suppress(Exception):
                parsed = orjson.loads(raw_msg)

            backend_msg = None
            if isinstance(parsed, dict):
                backend_msg = parsed.get("message")

            error_blob = str(backend_msg or raw_msg).lower()

            # serde rejects the unknown enum value with e.g.
            # `unknown variant `bind`, expected `open` or `close``.
            if "unknown variant" in error_blob and "bind" in error_blob:
                return ErrorInsight(
                    title="Unsupported Dynamo session_control action: bind",
                    problem=(
                        "The Dynamo frontend rejected nvext.session_control with "
                        "action='bind'. This Dynamo build's SessionAction only "
                        "accepts 'open' and 'close' -- the 'bind' action was added "
                        "after the v1.2.x release line (first available in "
                        "v1.3.0-dev / upstream commit d97c889ba)."
                    ),
                    causes=[
                        "--session-routing dynamo_nvext emits action='bind' on every non-final turn.",
                        "The target Dynamo server predates the 'bind' action (e.g. v1.2.1).",
                    ],
                    investigation=[
                        "Check the Dynamo frontend version and its supported SessionAction values.",
                        "Inspect request payloads in profile_export.jsonl -> nvext.session_control.",
                    ],
                    fixes=[
                        "Upgrade Dynamo to a build that supports action='bind' (>= v1.3.0-dev, upstream commit d97c889ba).",
                        "Or switch to --session-routing dynamo_headers (header-based affinity; requires --router-session-affinity-ttl-secs on the Dynamo frontend).",
                    ],
                )

            # Current upstream Dynamo removed session_control and its NvExt
            # deserializer denies unknown fields, so the whole field is rejected.
            if "unknown field" in error_blob and "session_control" in error_blob:
                return ErrorInsight(
                    title="Dynamo build does not implement nvext.session_control",
                    problem=(
                        "The Dynamo frontend rejected nvext.session_control as an unknown "
                        "field. Current upstream Dynamo removed session_control; its NvExt "
                        "deserializer denies unknown fields, so every request fails."
                    ),
                    causes=[
                        "--session-routing dynamo_nvext emits nvext.session_control on every turn.",
                        "The target Dynamo build has no session_control support (e.g. current main).",
                    ],
                    investigation=[
                        "Check the Dynamo frontend version/build for session_control support.",
                        "Inspect request payloads in profile_export.jsonl -> nvext.session_control.",
                    ],
                    fixes=[
                        "Switch to --session-routing dynamo_headers and run the Dynamo "
                        "frontend with --router-session-affinity-ttl-secs.",
                        "Or drop --session-routing for an untagged baseline.",
                    ],
                )

        return None


class ConsoleApiErrorExporter(AIPerfLoggerMixin):
    """Displays helpful diagnostic panels for known API error patterns."""

    DETECTORS: ClassVar[list] = [
        MaxCompletionTokensDetector,
        DynamoSessionControlDetector,
    ]

    def __init__(self, exporter_config: ExporterConfig, **kwargs):
        super().__init__(**kwargs)
        self._results = exporter_config.results

    async def export(self, console: Console) -> None:
        error_summary: list[ErrorDetailsCount] | None = self._results.error_summary

        for detector in self.DETECTORS:
            insight = detector.detect(error_summary)
            if insight:
                panel = Panel(
                    self._format_text(insight),
                    title=insight.title,
                    border_style="bold yellow",
                    title_align="center",
                    padding=(0, 2),
                    expand=False,
                )
                console.print()
                console.print(panel)
                console.file.flush()

    def _format_text(self, insight: ErrorInsight) -> str:
        return (
            f"""\
[bold]{insight.problem}[/bold]

[bold]Possible Causes:[/bold]
  • """
            + "\n  • ".join(insight.causes)
            + """

[bold]Investigation Steps:[/bold]
  1. """
            + "\n  1. ".join(insight.investigation)
            + """

[bold]Suggested Fixes:[/bold]
  • """
            + "\n  • ".join(insight.fixes)
        )
