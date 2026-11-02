# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from rich.console import Console
from rich.panel import Panel

from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.exporters.exporter_config import ExporterConfig


class ConsoleTruncationNoticeExporter(AIPerfLoggerMixin):
    """Console notice for conversations ended early on a truncated turn.

    With end_conversation_on_truncation on, the request and conversation counts
    differ from a plain multi-turn run, so the reader needs to know how many
    conversations stopped early and how many turns were never sent. Shown only
    when at least one conversation ended early.
    """

    def __init__(self, exporter_config: ExporterConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self._results = exporter_config.results

    async def export(self, console: Console) -> None:
        ended = getattr(self._results, "conversations_ended_on_truncation", None)
        if not ended:
            return
        skipped = self._results.turns_skipped_on_truncation or 0
        panel = Panel(
            f"[bold]{ended:,}[/bold] conversations ended early after a turn was "
            f"truncated at the output-token cap; [bold]{skipped:,}[/bold] "
            "remaining turns were skipped.\n"
            "Each truncated turn is still measured. Raise --osl (or the dataset's "
            "max_tokens) to complete more conversations, or pass "
            "--no-end-conversation-on-truncation to send every turn.",
            title="Conversations Ended on Truncation",
            border_style="bold cyan",
            title_align="center",
            padding=(0, 2),
            expand=False,
        )
        console.print()
        console.print(panel)
        console.file.flush()
