# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING

# NOTE: Do as little imports as possible in this file to ensure the CLI is fast to start up

if TYPE_CHECKING:
    from rich.align import AlignMethod
    from rich.console import RenderableType
    from rich.style import StyleType


def warn_command_not_implemented(command: str) -> None:
    """Warn the user that the subcommand is not implemented."""
    raise_startup_error_and_exit(
        f"Command [bold red]{command}[/bold red] is not yet implemented",
        title="Not Implemented",
    )


def raise_startup_error_and_exit(
    message: "RenderableType",
    text_color: "StyleType | None" = None,
    title: str = "Error",
    exit_code: int = 1,
    border_style: "StyleType" = "bold red",
    title_align: "AlignMethod" = "left",
) -> None:
    """Raise a startup error and exit the program.

    Args:
        message: The message to display. Can be a string or a rich renderable.
        text_color: The text color to use.
        title: The title of the error.
        exit_code: The exit code to use.
        border_style: The border style to use.
        title_align: The alignment of the title.
    """
    import sys

    from rich.console import Console
    from rich.panel import Panel

    if isinstance(message, str):
        message = f"[{text_color}]{message}[/{text_color}]" if text_color else message

    console = Console()
    console.print(
        Panel(
            renderable=message,
            title=title,
            title_align=title_align,
            border_style=border_style,
        )
    )
    console.file.flush()

    sys.exit(exit_code)


class exit_on_error(AbstractContextManager):
    """Context manager that exits the program if an error occurs.

    Args:
        *exceptions: The exceptions to exit on. If no exceptions are provided, all exceptions will be caught.
        message: The message to display. Can be a string or a rich renderable. Will be formatted with the exception as `{e}`.
        text_color: The text color to use.
        title: The title of the error.
        exit_code: The exit code to use.
        show_traceback: Whether to show the full exception traceback. Defaults to True.
    """

    def __init__(
        self,
        *exceptions: type[BaseException],
        message: "RenderableType" = "{e}",
        text_color: "StyleType | None" = None,
        title: str = "Error",
        exit_code: int = 1,
        show_traceback: bool = True,
    ):
        self.message: RenderableType = message
        self.text_color: StyleType | None = text_color
        self.title: str = title
        self.exit_code: int = exit_code
        self.exceptions: tuple[type[BaseException], ...] = exceptions
        self.show_traceback: bool = show_traceback

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            return

        if (
            not self.exceptions
            and not isinstance(exc_value, (SystemExit | KeyboardInterrupt))
        ) or issubclass(exc_type, self.exceptions):
            from rich.console import Console

            console = Console()

            # Only show full traceback if requested
            # Don't show locals as they are very noisy and not useful for most errors
            if self.show_traceback:
                console.print_exception(
                    show_locals=False,
                    max_frames=10,
                    word_wrap=True,
                    width=console.width,
                )
                console.file.flush()

            message = (
                self.message.format(e=exc_value)
                if isinstance(self.message, str)
                else self.message
            )
            raise_startup_error_and_exit(
                message,
                text_color=self.text_color,
                title=self.title,
                exit_code=self.exit_code,
            )


def print_developer_mode_warning() -> None:
    """Print a warning message to the console if developer mode is enabled."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text

    console = Console()
    panel = Panel(
        Text(
            "Developer Mode is active. This is a developer-only feature. Use at your own risk.",
            style="yellow",
        ),
        title="AIPerf Developer Mode",
        border_style="bold yellow",
        title_align="left",
    )
    console.print(panel)
    console.file.flush()


def warn_osl_without_ignore_eos() -> None:
    """Log a warning when --osl is used without ignore_eos or min_tokens in extra inputs."""
    import logging

    logger = logging.getLogger("aiperf")
    logger.warning(
        "Using --osl without ignore_eos or min_tokens in --extra-inputs. "
        "Output sequence length cannot be guaranteed unless the server honors these parameters. "
        "Consider: --extra-inputs ignore_eos:true (generate until max_tokens) "
        "or --extra-inputs min_tokens:<value> (set minimum output length)."
    )
