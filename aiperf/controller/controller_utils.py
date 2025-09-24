# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import textwrap
from collections import defaultdict

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from aiperf.common.models import ErrorDetails, ExitErrorInfo


def _group_errors_by_details(
    exit_errors: list[ExitErrorInfo],
) -> dict[ErrorDetails, list[ExitErrorInfo]]:
    """Group exit errors by their error details to deduplicate similar errors."""
    grouped_errors: dict[ErrorDetails, list[ExitErrorInfo]] = defaultdict(list)
    for error in exit_errors:
        grouped_errors[error.error_details].append(error)
    return dict(grouped_errors)


def print_exit_errors(
    exit_errors: list[ExitErrorInfo] | None = None, console: Console | None = None
) -> None:
    """Display command errors to the user with deduplication of similar errors."""
    if not exit_errors:
        return
    console = console or Console()

    def _create_field(
        label: str, value: str, prefix: str = "   ", end: str = "\n"
    ) -> Text:
        """Helper to create a formatted field for error display."""
        return Text.assemble(
            Text(f"{prefix}{label}: ", style="bold yellow"),
            Text(f"{value}{end}", style="bold"),
        )

    # Group errors by their error details to deduplicate
    grouped_errors = _group_errors_by_details(exit_errors)

    summary = []
    for i, (error_details, error_list) in enumerate(grouped_errors.items()):
        # Show the first occurrence's operation, or a generic message if operations differ
        operations = {error.operation for error in error_list}
        operation_display = (
            next(iter(operations)) if len(operations) == 1 else "Multiple Operations"
        )

        # Collect affected services
        affected_services = sorted([error.service_id or "N/A" for error in error_list])
        service_count = len(affected_services)

        if service_count == 1:
            service_display = affected_services[0]
        elif service_count <= 3:
            # Show all services for service counts less than or equal to 3
            service_display = (
                f"{service_count} services: {', '.join(affected_services)}"
            )
        else:
            # Show first 2 + "etc." for service counts greater than 3
            shown_services = affected_services[:2]
            service_display = (
                f"{service_count} services: {', '.join(shown_services)}, etc."
            )

        summary.append(_create_field("Services", service_display, prefix="• "))
        summary.append(_create_field("Operation", operation_display))
        summary.append(_create_field("Error", error_details.type or "Unknown"))

        # Reason field with text wrapping
        wrapped_text = textwrap.fill(
            error_details.message,
            width=console.size.width - 15,  # Account for panel borders and indentation
            subsequent_indent=" " * 11,  # 11 spaces to align with "   Reason: "
        )

        # Add extra newline between errors, except for the last one
        end = "\n\n" if i < len(grouped_errors) - 1 else ""
        summary.append(_create_field("Reason", wrapped_text, end=end))

    console.print(
        Panel(
            Text.assemble(*summary),
            border_style="bold red",
            title="AIPerf System Exit Errors",
            title_align="left",
        )
    )
    console.file.flush()
