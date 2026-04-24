# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preflight check runner for Kubernetes E2E tests.

Uses Rich formatting when stderr is a TTY, plain text otherwise.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field


class _PlainConsole:
    """Minimal Console-like interface that writes plain text to stderr."""

    def print(self, text: str = "") -> None:
        import re

        # Strip rich markup tags like [bold blue], [/bold blue], [dim], etc.
        plain = re.sub(r"\[/?[^\]]*\]", "", text)
        print(plain, file=sys.stderr)


def _make_console() -> _PlainConsole:
    """Create a console appropriate for the current terminal."""
    if sys.stderr.isatty():
        from rich.console import Console

        return Console(stderr=True)  # type: ignore[return-value]
    return _PlainConsole()


@dataclass
class _CheckResult:
    """Result of a single preflight check."""

    name: str
    """Human-readable check name."""

    passed: bool
    """Whether the check passed."""

    message: str
    """Detail message describing the result."""

    elapsed: float
    """Time taken for this check in seconds."""


@dataclass
class PreflightChecker:
    """Runs and displays preflight checks using rich formatting.

    Usage::

        checker = PreflightChecker("GPU KUBERNETES E2E TEST")
        checker.set_mode("external cluster", context="my-ctx")

        c = checker.check("kubectl connectivity")
        # ... run check ...
        c.pass_("cluster reachable")

        checker.finish()  # raises if any check failed
    """

    title: str
    """Title displayed in the preflight banner."""

    skip_message: str = ""
    """Message shown when checks fail, suggesting how to skip."""

    _results: list[_CheckResult] = field(default_factory=list, init=False)
    """Accumulated check results."""

    _console: _PlainConsole = field(default_factory=_make_console, init=False)
    """Console for formatted output."""

    _mode_text: str = field(default="", init=False)
    """Mode description line shown below the title."""

    _start_time: float = field(default=0.0, init=False)
    """Timestamp when preflight sequence started."""

    _current_step: int = field(default=0, init=False)
    """Current step counter."""

    _total_steps: int = field(default=0, init=False)
    """Total expected number of checks."""

    def set_mode(self, mode: str, **details: str) -> None:
        """Set the mode line shown below the title.

        Args:
            mode: Mode description (e.g. "external cluster", "minikube").
            **details: Key-value pairs shown after the mode.
        """
        parts = [f"[bold]{mode}[/bold]"]
        for k, v in details.items():
            parts.append(f"[dim]{k}=[/dim]{v}")
        self._mode_text = "  ".join(parts)

    def start(self, total_steps: int) -> None:
        """Begin the preflight sequence.

        Args:
            total_steps: Total number of checks to run.
        """
        self._total_steps = total_steps
        self._start_time = time.perf_counter()

        self._console.print()
        self._console.print(f"[bold blue]{self.title} PREFLIGHT CHECKS[/bold blue]")
        self._console.print(f"[blue]{'=' * 60}[/blue]")
        if self._mode_text:
            self._console.print(f"  Mode: {self._mode_text}")

    def check(self, name: str) -> _CheckContext:
        """Start a preflight check.

        Call ``pass_()`` or ``fail()`` on the returned context to record
        the result.

        Args:
            name: Human-readable check name.

        Returns:
            A context object with pass_/fail methods.
        """
        self._current_step += 1
        return _CheckContext(checker=self, name=name, step=self._current_step)

    def _record(self, result: _CheckResult, step: int) -> None:
        """Record a check result and print its status line."""
        self._results.append(result)
        step_label = f"[dim][{step}/{self._total_steps}][/dim]"
        if result.passed:
            mark = "[bold green]\u2713[/bold green]"
        else:
            mark = "[bold red]\u2717[/bold red]"
        self._console.print(
            f"  {step_label} {mark} [cyan]{result.name}[/cyan]: "
            f"{result.message}  [dim]({result.elapsed:.2f}s)[/dim]"
        )

    def finish(self, error_cls: type[Exception] | None = None) -> None:
        """Print summary and raise if any check failed.

        Args:
            error_cls: Exception class to raise on failure. If None, does
                not raise (caller handles failure).
        """
        elapsed_total = time.perf_counter() - self._start_time
        failed = sum(1 for r in self._results if not r.passed)
        total = len(self._results)
        all_passed = failed == 0

        self._console.print(f"[blue]{'-' * 60}[/blue]")

        if all_passed:
            self._console.print(
                f"[bold green]\u2713 All {total} checks passed[/bold green]"
                f"  [dim]({elapsed_total:.2f}s)[/dim]"
            )
        else:
            self._console.print(
                f"[bold red]\u2717 {failed}/{total} checks failed[/bold red]"
                f"  [dim]({elapsed_total:.2f}s)[/dim]"
            )
            if self.skip_message:
                self._console.print(f"  [dim]{self.skip_message}[/dim]")

            if error_cls is not None:
                raise error_cls(
                    f"Preflight checks failed ({failed}/{total})"
                    " - see above for details"
                )

        self._console.print()


@dataclass
class _CheckContext:
    """Context object for a single preflight check."""

    checker: PreflightChecker
    """Parent checker that owns this context."""

    name: str
    """Check name displayed in output."""

    step: int
    """Step number within the preflight sequence."""

    _start: float = field(default_factory=time.perf_counter, init=False)
    """Timestamp when this check started."""

    def pass_(self, message: str) -> None:
        """Mark this check as passed.

        Args:
            message: Success detail message.
        """
        elapsed = time.perf_counter() - self._start
        self.checker._record(_CheckResult(self.name, True, message, elapsed), self.step)

    def fail(self, message: str) -> None:
        """Mark this check as failed.

        Args:
            message: Failure detail message.
        """
        elapsed = time.perf_counter() - self._start
        self.checker._record(
            _CheckResult(self.name, False, message, elapsed), self.step
        )
