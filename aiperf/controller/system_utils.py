# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from aiperf.common.messages import CommandErrorResponse
from aiperf.common.messages.command_messages import CommandResponse
from aiperf.common.models.error_models import ErrorDetails


def extract_errors(
    responses: list[CommandResponse | ErrorDetails],
) -> list[CommandErrorResponse | ErrorDetails]:
    """Extract errors from a list of command responses."""
    return [
        response
        for response in responses
        if isinstance(response, CommandErrorResponse | ErrorDetails)
    ]


def display_command_errors(
    title: str, errors: list[CommandErrorResponse | ErrorDetails]
) -> None:
    """Display command errors to the user."""
    if not errors:
        return
    summary = []
    for error in errors:
        if isinstance(error, CommandErrorResponse):
            summary.append(
                Text.assemble(
                    Text("•", style="bold red"),
                    f" Service: {error.service_id}: Command: {error.command}\n",
                )
            )
            summary.append(
                Text.assemble(
                    Text(f"\t{error.error.type}:", style="bold red"),
                    f" {error.error.message}\n",
                )
            )
        else:
            summary.append(
                Text.assemble(
                    Text(f"• {error.type}:", style="bold red"), f" {error.message}\n"
                )
            )
    # Remove the trailing newline from the last summary item
    summary[-1]._text[-1] = summary[-1]._text[-1].rstrip("\n")
    console = Console()
    console.print(
        Panel(
            Text.assemble(*summary),
            border_style="bold red",
            title=f"Error: {title}",
            title_align="left",
        )
    )
    console.file.flush()
