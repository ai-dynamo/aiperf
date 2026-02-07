# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import textwrap
from collections import defaultdict

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from aiperf.common.models import ErrorDetails, ExitErrorInfo
from aiperf.common.tokenizer_display import (
    display_tokenizer_validation_error,
    extract_tokenizer_name_from_error,
    is_tokenizer_error,
)


def _group_errors_by_details(
    exit_errors: list[ExitErrorInfo],
) -> dict[ErrorDetails, list[ExitErrorInfo]]:
    """Group exit errors by their error details to deduplicate similar errors."""
    grouped: dict[ErrorDetails, list[ExitErrorInfo]] = defaultdict(list)
    for error in exit_errors:
        grouped[error.error_details].append(error)
    return dict(grouped)


def _is_wrapper_error(error: ExitErrorInfo) -> bool:
    """Check if an error is a wrapper/aggregate error that should be filtered."""
    msg = error.error_details.message.lower()
    return error.service_id == "system_controller" and (
        "failed to perform operation" in msg or "aiperfmultierror" in msg
    )


def _handle_tokenizer_errors(
    exit_errors: list[ExitErrorInfo], console: Console
) -> list[ExitErrorInfo]:
    """Display tokenizer errors with specialized output, return non-tokenizer errors."""
    remaining: list[ExitErrorInfo] = []
    displayed: set[str] = set()

    for error in exit_errors:
        cause_chain = error.error_details.cause_chain
        if not is_tokenizer_error(cause_chain):
            remaining.append(error)
            continue

        # Skip duplicates (by cause_chain or message)
        key = str(cause_chain) if cause_chain else error.error_details.message
        if key in displayed:
            continue
        displayed.add(key)

        # Extract tokenizer name from message or cause
        msg = error.error_details.message
        cause = str(error.error_details.cause) if error.error_details.cause else ""
        name = (
            extract_tokenizer_name_from_error(msg)
            or extract_tokenizer_name_from_error(cause)
            or "<unknown>"
        )

        display_tokenizer_validation_error(
            name, cause_chain=cause_chain, error_message=msg, console=console
        )

    # Filter wrapper errors if we displayed any tokenizer errors
    if displayed:
        remaining = [e for e in remaining if not _is_wrapper_error(e)]

    return remaining


def _format_field(label: str, value: str, prefix: str = "   ") -> Text:
    """Create a formatted label: value field for error display."""
    return Text.assemble(
        Text(f"{prefix}{label}: ", style="bold yellow"),
        Text(value, style="bold"),
    )


def _format_services(services: set[str]) -> str:
    """Format service names for display."""
    sorted_services = sorted(services)
    count = len(sorted_services)
    if count == 1:
        return sorted_services[0]
    if count <= 3:
        return f"{count} services: {', '.join(sorted_services)}"
    return f"{count} services: {', '.join(sorted_services[:2])}, etc."


def print_exit_errors(
    exit_errors: list[ExitErrorInfo] | None = None, console: Console | None = None
) -> None:
    """Display command errors to the user with deduplication of similar errors."""
    if not exit_errors:
        return
    console = console or Console()

    # Handle tokenizer errors with specialized display first
    exit_errors = _handle_tokenizer_errors(exit_errors, console)
    if not exit_errors:
        return

    grouped = _group_errors_by_details(exit_errors)
    wrap_width = max(console.size.width - 15, 20)

    def wrap_text(text: str) -> str:
        return textwrap.fill(text, width=wrap_width, subsequent_indent=" " * 11)

    summary: list[Text | str] = []
    for i, (details, errors) in enumerate(grouped.items()):
        operations = {e.operation for e in errors}
        services = {e.service_id or "N/A" for e in errors}

        operation = (
            next(iter(operations)) if len(operations) == 1 else "Multiple Operations"
        )
        service_label = "Services" if len(services) > 1 else "Service"

        summary.append(
            _format_field(service_label, _format_services(services), prefix="• ")
        )
        summary.append(Text("\n"))
        summary.append(_format_field("Operation", operation))
        summary.append(Text("\n"))
        summary.append(_format_field("Error", details.type or "Unknown"))
        summary.append(Text("\n"))
        summary.append(_format_field("Reason", wrap_text(details.message)))

        if details.cause:
            summary.append(Text("\n"))
            summary.append(_format_field("Cause", wrap_text(str(details.cause))))

        if details.details:
            summary.append(Text("\n"))
            summary.append(_format_field("Details", wrap_text(str(details.details))))

        if i < len(grouped) - 1:
            summary.append(Text("\n\n"))

    console.print()
    console.print(
        Panel(
            Text.assemble(*summary),
            border_style="bold red",
            title="AIPerf System Exit Errors",
            title_align="left",
        )
    )
    console.file.flush()
