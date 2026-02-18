#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate CLI documentation by introspecting the cyclopts application.

Usage:
    ./tools/generate_cli_docs.py
    ./tools/generate_cli_docs.py --check
    ./tools/generate_cli_docs.py --verbose
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow direct execution: add repo root to path for 'tools' package imports
if __name__ == "__main__" and "tools" not in sys.modules:
    sys.path.insert(0, str(Path(__file__).parent.parent))

import ast
import inspect
import re
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from inspect import isclass
from io import StringIO
from typing import Any, get_origin

from rich.console import Console

from tools._core import (
    CONSTRAINT_SYMBOLS,
    SPDX_HEADER_MD,
    CLIExtractionError,
    GeneratedFile,
    Generator,
    GeneratorResult,
    main,
    normalize_text,
    print_step,
    print_warning,
)

# =============================================================================
# Configuration
# =============================================================================

OUTPUT_FILE = Path("docs/cli_options.md")

# NumPy-style docstring section headers that terminate description extraction.
# Google-style ("Args:", "Examples:") are handled separately with startswith().
_DOCSTRING_SECTIONS = frozenset(
    {
        "parameters",
        "returns",
        "raises",
        "notes",
        "references",
        "yields",
        "attributes",
        "see also",
        "warnings",
    }
)

# =============================================================================
# Data Models
# =============================================================================


@dataclass
class Param:
    """CLI parameter info."""

    name: str
    long_opts: str
    short: str
    description: str
    required: bool
    type_suffix: str
    default: str = ""
    choices: list[str] | None = None
    choice_descs: dict[str, str] | None = None
    constraints: list[str] | None = None


# =============================================================================
# Extraction Helpers
# =============================================================================


def _get_enum_docstrings(enum_class: type[Enum]) -> dict[str, str]:
    """Extract docstrings for enum members from source."""
    try:
        source = inspect.getsource(enum_class)
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == enum_class.__name__:
                docs = {}
                for i, item in enumerate(node.body):
                    if (
                        isinstance(item, ast.Assign)
                        and item.targets
                        and isinstance(item.targets[0], ast.Name)
                    ):
                        name = item.targets[0].id
                        # Check next node for docstring
                        if i + 1 < len(node.body):
                            next_item = node.body[i + 1]
                            if (
                                isinstance(next_item, ast.Expr)
                                and isinstance(next_item.value, ast.Constant)
                                and isinstance(next_item.value.value, str)
                            ):
                                docs[name] = next_item.value.value.strip()
                return docs
    except (OSError, TypeError, SyntaxError):
        pass
    return {}


def _build_constraints(
    model_class: type, visited: set | None = None
) -> dict[str, list[str]]:
    """Build constraint map from Pydantic model."""
    from pydantic import BaseModel

    # Sort order: lower bounds first (>, ≥), then upper bounds (<, ≤)
    constraint_order = {">": 0, "≥": 1, "<": 2, "≤": 3, "min:": 4, "max:": 5}

    visited = visited or set()
    if model_class in visited:
        return {}
    visited.add(model_class)

    result: dict[str, list[str]] = {}

    for name, info in model_class.model_fields.items():
        constraints = []
        for attr, sym in CONSTRAINT_SYMBOLS.items():
            val = getattr(info, attr, None)
            if val is not None:
                constraints.append(f"{sym} {val}")
            if hasattr(info, "metadata"):
                for meta in info.metadata or []:
                    val = getattr(meta, attr, None)
                    if val is not None and f"{sym} {val}" not in constraints:
                        constraints.append(f"{sym} {val}")
        if constraints:
            # Sort: lower bounds first, then upper bounds
            constraints.sort(key=lambda c: constraint_order.get(c.split()[0], 99))
            result[name] = constraints

        # Recurse into nested models
        ann = info.annotation
        args = getattr(ann, "__args__", ())
        for arg in [ann, *args]:
            if isinstance(arg, type) and issubclass(arg, BaseModel):
                for k, v in _build_constraints(arg, visited).items():
                    result.setdefault(k, v)

    return result


def _extract_text(obj: Any) -> str:
    """Extract plain text from cyclopts InlineText."""
    from cyclopts.help import InlineText

    if isinstance(obj, InlineText):
        buf = StringIO()
        Console(file=buf, record=True, width=1000).print(obj)
        return normalize_text(buf.getvalue().replace("\r", ""))
    return str(obj) if obj else ""


def _type_suffix(hint: Any) -> str:
    """Get type suffix for display."""
    mapping = {
        bool: "",
        int: " <int>",
        float: " <float>",
        list: " <list>",
        tuple: " <list>",
        set: " <list>",
    }
    return mapping.get(hint, mapping.get(get_origin(hint), " <str>"))


def _extract_param(arg: Any, constraints: dict[str, list[str]]) -> Param:
    """Extract parameter info from cyclopts argument."""
    from cyclopts.field_info import FieldInfo

    # Split names
    long_opts = [n for n in arg.names if n.startswith("--")]
    short_opts = [n for n in arg.names if n.startswith("-") and n not in long_opts]

    # Default value
    default = ""
    if arg.show_default:
        val = arg.field_info.default
        if val is not FieldInfo.empty and val is not None:
            default = (
                str(arg.show_default(val)) if callable(arg.show_default) else str(val)
            )

    # Choices from enum
    choices = None
    choice_descs = None
    if arg.parameter.show_choices:
        enum_cls = None
        if isclass(arg.hint) and issubclass(arg.hint, Enum):
            enum_cls = arg.hint
        elif get_origin(arg.hint) in (list, tuple, set):
            args = getattr(arg.hint, "__args__", ())
            if args and isclass(args[0]) and issubclass(args[0], Enum):
                enum_cls = args[0]

        if enum_cls:
            choices = [f"`{m.value}`" for m in enum_cls]
            docs = _get_enum_docstrings(enum_cls)
            if docs:
                choice_descs = {
                    f"`{m.value}`": normalize_text(docs.get(n, ""))
                    for n, m in enum_cls.__members__.items()
                }

    # Constraints
    field_name = arg.names[0].lstrip("-").replace("-", "_")

    return Param(
        name=arg.names[0].lstrip("-").replace("-", " ").title()
        + (" _(Required)_" if arg.required else ""),
        long_opts=" --".join(long_opts),
        short=" ".join(short_opts),
        description=_extract_text(arg.parameter.help),
        required=arg.required,
        type_suffix=_type_suffix(arg.hint),
        default=default,
        choices=choices,
        choice_descs=choice_descs,
        constraints=constraints.get(field_name),
    )


def extract_commands(app: Any) -> list[tuple[str, str]]:
    """Extract command names and descriptions."""
    skip = {"--help", "-h", "--version"}
    commands = []
    for name, cmd in app._commands.items():
        if name in skip:
            continue
        help_text = cmd.help if hasattr(cmd, "help") else ""
        if callable(help_text):
            help_text = help_text()
        if help_text:
            help_text = _extract_text(help_text).split("\n")[0].strip()
        commands.append((name, help_text))
    return commands


def extract_params(app: Any, subcommand: str) -> dict[str, list[Param]]:
    """Extract parameters for a subcommand."""
    from typing import get_type_hints

    from cyclopts.bind import normalize_tokens
    from pydantic import BaseModel

    tokens = normalize_tokens(subcommand)
    _, apps, _ = app.parse_commands(tokens)

    # Build constraints from type hints
    constraints: dict[str, list[str]] = {}
    func = apps[-1].default_command
    if func:
        for hint in get_type_hints(func, include_extras=False).values():
            if isinstance(hint, type) and issubclass(hint, BaseModel):
                constraints.update(_build_constraints(hint))
            for arg in getattr(hint, "__args__", ()):
                if isinstance(arg, type) and issubclass(arg, BaseModel):
                    constraints.update(_build_constraints(arg))

    # Extract params by group
    groups: dict[str, list[Param]] = defaultdict(list)
    for arg in (
        apps[-1].assemble_argument_collection(parse_docstring=True).filter_by(show=True)
    ):
        groups[arg.parameter.group[0].name].append(_extract_param(arg, constraints))

    return dict(groups)


# =============================================================================
# Markdown Generation
# =============================================================================


def _format_param(param: Param) -> list[str]:
    """Format a parameter as markdown."""
    # Header
    opts = []
    if param.short:
        opts.append(f"`{param.short}`")
    for opt in param.long_opts.split(" --"):
        if opt := opt.strip():
            opts.append(
                f"`{'--' if not opt.startswith('--') else ''}{opt.lower().replace(' ', '-')}`"
            )

    if not opts:
        return []

    type_str = f" `{param.type_suffix.strip()}`" if param.type_suffix else ""
    req = " _(Required)_" if param.required else ""
    lines = [f"#### {', '.join(opts)}{type_str}{req}", ""]

    # Body
    lines.append(f"{normalize_text(param.description).rstrip('.')}.")

    if param.type_suffix in ("", " <bool>") and "--no-" not in param.long_opts:
        lines.append("<br>_Flag (no value required)_")

    if param.constraints:
        lines.append(f"<br>_Constraints: {', '.join(param.constraints)}_")

    if param.choices:
        if param.choice_descs:
            lines.extend(
                ["", "**Choices:**", "", "| | | |", "|-------|:-------:|-------------|"]
            )
            for choice in param.choices:
                desc = param.choice_descs.get(choice, "")
                val = choice.strip("`")
                is_default = False
                if param.default and param.default != "False":
                    ds = str(param.default)
                    if ds.startswith("[") and ds.endswith("]"):
                        vals = re.findall(r"\.(\w+)", ds) + re.findall(r"'([^']+)'", ds)
                        is_default = any(val.lower() == v.lower() for v in vals)
                    else:
                        is_default = val == param.default
                marker = "_default_" if is_default else ""
                lines.append(f"| {choice} | {marker} | {desc} |")
        else:
            lines.append(f"<br>_Choices: [{', '.join(param.choices)}]_")
            if param.default and param.default != "False":
                lines.append(f"<br>_Default: `{param.default}`_")
    elif param.default and param.default != "False":
        lines.append(f"<br>_Default: `{param.default}`_")

    lines.append("")
    return lines


def generate_markdown(app: Any, data: dict[str, dict[str, list[Param]]]) -> str:
    """Generate full markdown documentation."""
    lines = [
        *SPDX_HEADER_MD,
        "",
        "# Command Line Options",
        "",
    ]

    # TOC
    if data:
        lines.extend(["## `aiperf` Commands", ""])
        for name, desc in extract_commands(app):
            if name in data:
                lines.extend(
                    [
                        f"### [`{name}`](#aiperf-{name.lower().replace(' ', '-')})",
                        "",
                        desc,
                        "",
                    ]
                )
                groups = data[name]
                if len(groups) > 1 or list(groups.keys())[0] not in (
                    "Parameters",
                    "Options",
                    "General",
                ):
                    links = [
                        f"[{g}](#{g.lower().replace(' ', '-').replace('(', '').replace(')', '')})"
                        for g in groups
                    ]
                    lines.extend([" • ".join(links), ""])

    # Command sections
    for cmd_name, groups in data.items():
        lines.extend(["<hr>", "", f"## `aiperf {cmd_name}`", ""])

        # Command help text
        cmd = app._commands.get(cmd_name)
        if cmd and hasattr(cmd, "help"):
            help_text = cmd.help() if callable(cmd.help) else cmd.help
            if help_text:
                text = _extract_text(help_text)
                text_lines = text.split("\n")

                # Extract description (before docstring sections)
                desc_lines = []
                examples_idx = None
                for i, line in enumerate(text_lines):
                    stripped = line.strip().lower()
                    if stripped.startswith("examples:"):
                        examples_idx = i
                        break
                    if stripped.startswith("args:"):
                        break
                    if stripped in _DOCSTRING_SECTIONS:
                        break
                    desc_lines.append(line)

                desc = "\n".join(desc_lines).strip()
                if desc:
                    for para in desc.split("\n\n"):
                        if para.strip():
                            lines.extend([normalize_text(para), ""])

                # Extract examples
                if examples_idx is not None:
                    end_idx = len(text_lines)
                    for i in range(examples_idx + 1, len(text_lines)):
                        s = text_lines[i].strip().lower()
                        if s.startswith("args:") or s in _DOCSTRING_SECTIONS:
                            end_idx = i
                            break
                    example_lines = text_lines[examples_idx + 1 : end_idx]
                    # Strip leading/trailing blank lines but preserve internal ones
                    while example_lines and not example_lines[0].strip():
                        example_lines.pop(0)
                    while example_lines and not example_lines[-1].strip():
                        example_lines.pop()
                    if example_lines:
                        non_blank = [ln for ln in example_lines if ln.strip()]
                        min_indent = min(len(ln) - len(ln.lstrip()) for ln in non_blank)
                        lines.extend(["**Examples:**", "", "```bash"])
                        lines.extend(
                            ln[min_indent:] if ln.strip() else ""
                            for ln in example_lines
                        )
                        lines.extend(["```", ""])

        # Parameters
        skip_header = len(groups) == 1 and list(groups.keys())[0] in (
            "Parameters",
            "Options",
            "General",
        )
        for group_name, params in groups.items():
            if not skip_header:
                lines.extend([f"### {group_name}", ""])
            for param in params:
                lines.extend(_format_param(param))

    return "\n".join(line.rstrip() for line in lines)


# =============================================================================
# Generator
# =============================================================================


class CLIDocsGenerator(Generator):
    """Generate CLI documentation."""

    name = "CLI Documentation"
    description = "Generate CLI documentation for AIPerf"

    def generate(self) -> GeneratorResult:
        # Import app
        try:
            sys.path.insert(0, "src")
            from aiperf.cli import app
        except ImportError as e:
            raise CLIExtractionError(
                "Failed to import aiperf.cli",
                {
                    "error": str(e),
                    "hint": "Ensure aiperf is installed: uv pip install -e .",
                },
            ) from e

        # Extract commands
        commands = extract_commands(app)
        if self.verbose:
            print_step(f"Found {len(commands)} commands")

        # Extract params for each command
        data: dict[str, dict[str, list[Param]]] = {}
        for cmd_name, _ in commands:
            try:
                data[cmd_name] = extract_params(app, cmd_name)
                if self.verbose:
                    count = sum(len(p) for p in data[cmd_name].values())
                    print_step(f"Extracted `{cmd_name}` ({count} params)")
            except Exception as e:
                print_warning(f"Could not extract '{cmd_name}': {e}")

        total_params = sum(sum(len(p) for p in cmd.values()) for cmd in data.values())

        return GeneratorResult(
            files=[GeneratedFile(OUTPUT_FILE, generate_markdown(app, data))],
            summary=f"[bold]{len(data)}[/] commands with [bold]{total_params}[/] parameters",
        )


if __name__ == "__main__":
    main(CLIDocsGenerator)
