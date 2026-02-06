# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tokenizer error display and detection."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel

if TYPE_CHECKING:
    from aiperf.common.aiperf_logger import AIPerfLogger


@dataclass(slots=True)
class TokenizerErrorInsight:
    title: str
    causes: list[str]
    investigation: list[str]
    fixes: list[str]


# Exception type -> insight mapping
_INSIGHTS: dict[str, TokenizerErrorInsight] = {
    "GatedRepoError": TokenizerErrorInsight(
        title="Gated Repository",
        causes=["Model is gated - requires accepting terms on HuggingFace"],
        investigation=["Visit [cyan]huggingface.co/<model>[/cyan] to request access"],
        fixes=["Accept terms, then: [green]huggingface-cli login[/green]"],
    ),
    "RevisionNotFoundError": TokenizerErrorInsight(
        title="Invalid Git Revision",
        causes=["The specified revision (branch/tag/commit) does not exist"],
        investigation=["Check revisions at [cyan]huggingface.co/<model>/tree[/cyan]"],
        fixes=[
            "Remove [green]--tokenizer-revision[/green]",
            "Or use: [green]--tokenizer-revision main[/green]",
        ],
    ),
    "EntryNotFoundError": TokenizerErrorInsight(
        title="Missing Tokenizer Files",
        causes=[
            "Repository missing tokenizer files",
            "Model-only repo without tokenizer",
        ],
        investigation=["Check [cyan]huggingface.co/<model>/tree/main[/cyan]"],
        fixes=["Use a different tokenizer that matches your model"],
    ),
    "LocalEntryNotFoundError": TokenizerErrorInsight(
        title="Offline - Files Not Cached",
        causes=["Cannot connect to HuggingFace Hub and files not cached"],
        investigation=["Test: [cyan]curl -I https://huggingface.co[/cyan]"],
        fixes=["Pre-download online, then: [green]export HF_HUB_OFFLINE=1[/green]"],
    ),
    "RepositoryNotFoundError": TokenizerErrorInsight(
        title="Repository Not Found",
        causes=["Name misspelled", "Repository private or removed"],
        investigation=[
            "Search [link=https://huggingface.co/models]huggingface.co/models[/link]"
        ],
        fixes=["Use full ID: [green]--tokenizer org-name/model-name[/green]"],
    ),
    "HfHubHTTPError": TokenizerErrorInsight(
        title="HuggingFace Hub Error",
        causes=["Network or API error communicating with HuggingFace Hub"],
        investigation=["Test: [cyan]curl -I https://huggingface.co[/cyan]"],
        fixes=["Check network connectivity", "Retry later"],
    ),
    "ModuleNotFoundError": TokenizerErrorInsight(
        title="Missing Python Package",
        causes=["A required package is not installed"],
        investigation=["Check error for module name"],
        fixes=["Install: [green]pip install <package>[/green]"],
    ),
    "PermissionError": TokenizerErrorInsight(
        title="Cache Permission Error",
        causes=["Cannot write to cache", "Stale lock file"],
        investigation=["Check: [cyan]ls -la ~/.cache/huggingface/[/cyan]"],
        fixes=["Fix: [green]chmod -R u+rw ~/.cache/huggingface/[/green]"],
    ),
    "TimeoutError": TokenizerErrorInsight(
        title="Network Timeout",
        causes=["Network timeout", "Large files downloading slowly"],
        investigation=["Check network speed"],
        fixes=["Pre-download and use: [green]--tokenizer ./local-path[/green]"],
    ),
    "OSError": TokenizerErrorInsight(
        title="Tokenizer Load Error",
        causes=["File system error", "Network error", "Invalid tokenizer files"],
        investigation=["Check error message for details"],
        fixes=["Clear cache and retry", "Check network connectivity"],
    ),
    "AmbiguousTokenizerNameError": TokenizerErrorInsight(
        title="Ambiguous Tokenizer Name",
        causes=["Name matched multiple HuggingFace tokenizers"],
        investigation=["Check error message for suggested matches"],
        fixes=["Use full ID: [green]--tokenizer org-name/model-name[/green]"],
    ),
}

_FALLBACK_INSIGHT = TokenizerErrorInsight(
    title="Tokenizer Configuration Error",
    causes=["Tokenizer failed to load for an unexpected reason"],
    investigation=["Review error message", "Test with transformers directly"],
    fixes=[
        "Test: [cyan]python -c \"from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('...')\"[/cyan]"
    ],
)

_MODULE_NOT_FOUND_RE = re.compile(r"no module named ['\"]([^'\"]+)['\"]", re.IGNORECASE)
_TOKENIZER_NAME_PATTERNS = [
    re.compile(r"can't load tokenizer for ['\"]([^'\"]+)['\"]", re.IGNORECASE),
    re.compile(r"['\"]([^'\"]+)['\"] is not a local folder", re.IGNORECASE),
    re.compile(r"failed to load tokenizer[:\s]+['\"]?([^\s'\"]+)", re.IGNORECASE),
    re.compile(r"from_pretrained\(['\"]([^'\"]+)['\"]", re.IGNORECASE),
    re.compile(r"tokenizer ['\"]([^'\"]+)['\"]", re.IGNORECASE),
]

_TOKENIZER_EXCEPTION_TYPES = {
    "GatedRepoError",
    "RevisionNotFoundError",
    "EntryNotFoundError",
    "LocalEntryNotFoundError",
    "RepositoryNotFoundError",
    "HfHubHTTPError",
    "TokenizerError",
    "AmbiguousTokenizerNameError",
}

_EXPLANATION = (
    "[italic yellow]AIPerf needs a tokenizer for accurate client-side token "
    "counting and synthetic prompt generation.[/italic yellow]"
)
_SERVER_TOKEN_FIX = (
    "Skip tokenizer (non-synthetic data only): [green]--use-server-token-count[/green]"
)


def _detect_error(
    cause_chain: list[str] | None, error_message: str | None = None
) -> TokenizerErrorInsight:
    """Detect error type from cause_chain (iterates in reverse to find root cause first)."""
    if not cause_chain:
        return _FALLBACK_INSIGHT

    for type_name in reversed(cause_chain):
        if (
            type_name == "ModuleNotFoundError"
            and error_message
            and (match := _MODULE_NOT_FOUND_RE.search(error_message))
        ):
            module = match.group(1).split(".")[0]
            return TokenizerErrorInsight(
                title=f"Missing Package: {module}",
                causes=[f"The [cyan]{module}[/cyan] package is not installed"],
                investigation=[f"Check: [cyan]pip show {module}[/cyan]"],
                fixes=[f"Install: [green]pip install {module}[/green]"],
            )
        if type_name in _INSIGHTS:
            return _INSIGHTS[type_name]

    return _FALLBACK_INSIGHT


def is_tokenizer_error(cause_chain: list[str] | None = None) -> bool:
    return bool(cause_chain) and any(
        t in _TOKENIZER_EXCEPTION_TYPES for t in cause_chain
    )


def extract_tokenizer_name_from_error(error_message: str) -> str | None:
    for pattern in _TOKENIZER_NAME_PATTERNS:
        if match := pattern.search(error_message):
            return match.group(1)
    return None


@dataclass(slots=True)
class TokenizerDisplayEntry:
    original_name: str
    resolved_name: str
    was_resolved: bool


def log_tokenizer_validation_results(
    results: list[TokenizerDisplayEntry],
    logger: AIPerfLogger,
    elapsed_seconds: float | None = None,
) -> None:
    if not results:
        return

    for entry in results:
        if entry.was_resolved:
            logger.info(
                f"✓ Tokenizer {entry.resolved_name} detected for {entry.original_name}"
            )
        else:
            logger.info(f"✓ Tokenizer {entry.resolved_name} detected")

    total = len(results)
    resolved = sum(1 for e in results if e.was_resolved)
    parts = [f"{total} tokenizer{'s' if total > 1 else ''} validated"]
    if resolved > 0:
        parts.append(f"{resolved} resolved")
    if elapsed_seconds is not None:
        parts.append(f"{elapsed_seconds:.1f}s")
    logger.info(" • ".join(parts))


def _display_panel(title: str, content: str, console: Console | None = None) -> None:
    console = console or Console()
    console.print()
    console.print(
        Panel(
            content,
            title=f"[bold yellow]{title}[/bold yellow]",
            border_style="yellow",
            title_align="center",
            padding=(1, 2),
            expand=False,
        )
    )
    console.file.flush()


def _format_downloads(count: int) -> str:
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    if count >= 1_000:
        return f"{count / 1_000:.1f}K"
    return str(count)


def display_tokenizer_ambiguous_name(
    name: str,
    suggestions: list[tuple[str, int]],
    console: Console | None = None,
) -> None:
    suggestions_text = "\n".join(
        f"  • [cyan]{model_id}[/cyan] [dim]({_format_downloads(downloads)} downloads)[/dim]"
        for model_id, downloads in suggestions[:5]
    )
    specify_fix = (
        f"  • Specify explicitly: [green]--tokenizer {suggestions[0][0]}[/green]\n"
        if suggestions
        else ""
    )
    content = (
        f"[bold]'[white]{name}[/white]' matched multiple HuggingFace tokenizers[/bold]\n\n"
        f"{_EXPLANATION}\n\n"
        f"[bold]Did you mean one of these?[/bold]\n{suggestions_text or '  [dim]no suggested tokenizers[/dim]'}\n\n"
        f"[bold]Suggested Fixes:[/bold]\n"
        f"{specify_fix}"
        f"  • {_SERVER_TOKEN_FIX}"
    )
    _display_panel("Ambiguous Tokenizer Name", content, console)


def display_tokenizer_validation_error(
    name: str,
    cause_chain: list[str] | None = None,
    error_message: str | None = None,
    console: Console | None = None,
) -> None:
    insight = _detect_error(cause_chain, error_message)
    fixes = [*insight.fixes, _SERVER_TOKEN_FIX]

    content = (
        f"[bold]Failed to load tokenizer '[cyan]{name}[/cyan]'[/bold]\n\n"
        f"{_EXPLANATION}\n\n"
        "[bold]Possible Causes:[/bold]\n  • " + "\n  • ".join(insight.causes) + "\n\n"
        "[bold]Investigation Steps:[/bold]\n  "
        + "\n  ".join(
            f"{i + 1}. {step}" for i, step in enumerate(insight.investigation)
        )
        + "\n\n"
        "[bold]Suggested Fixes:[/bold]\n  • " + "\n  • ".join(fixes)
    )
    _display_panel(insight.title, content, console)
