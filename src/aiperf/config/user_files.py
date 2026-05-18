# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""User-defined templated output files materialized into the run directory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Annotated, Any, Literal

import jinja2
import orjson
import yaml
from pydantic import Field, model_validator

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.exceptions import AIPerfError
from aiperf.config.base import BaseConfig

if TYPE_CHECKING:
    # Avoid runtime import cycle: config.config imports config.artifacts which
    # imports this module. The annotation is enough for type-checkers.
    from aiperf.config.config import BenchmarkConfig

_logger = AIPerfLogger(__name__)

# C0 controls + DEL; matches POSIX portable filename character set negation.
_FORBIDDEN_PATH_CHARS = frozenset(chr(c) for c in range(32)) | {"\x7f"}

_JINJA_MARKERS = ("{{", "{%", "{#")


# Strict-undefined env duplicated from loader/jinja.py: that one is for load-time
# config expansion, this one is for run-time user_files materialization. Keep them
# separate so changes to one don't bleed into the other.
_USER_FILES_ENV = jinja2.Environment(
    undefined=jinja2.StrictUndefined,
    autoescape=False,
    keep_trailing_newline=True,
)


class UserFileError(AIPerfError):
    """Raised when a user_files entry fails validation, render, or write."""


class UserFile(BaseConfig):
    """One user-declared output file rendered into the run directory before benchmark start.

    Path is relative to the run directory; subdirectories are allowed; absolute
    paths and any segment equal to '..' are rejected. Content is rendered with
    jinja2 against a documented context (variables: + system-injected names).
    """

    path: Annotated[
        str,
        Field(
            description=(
                "Output path relative to the run directory. Subdirectories allowed. "
                "Absolute paths and any segment equal to '..' are rejected."
            ),
        ),
    ]

    format: Annotated[
        Literal["json", "yaml", "text"] | None,
        Field(
            default=None,
            description=(
                "Serialization format. If omitted: 'text' when content is a string, "
                "'json' otherwise."
            ),
        ),
    ] = None

    content: Annotated[
        Any,
        Field(
            description=(
                "Templated value. Structured (dict/list/scalar) for json/yaml; "
                "string for text. Jinja2 expressions in any string leaf are "
                "rendered with the user_files context."
            ),
        ),
    ]

    @model_validator(mode="after")
    def _validate_path(self) -> UserFile:
        if not self.path:
            raise ValueError("user_files entry has empty path")
        if any(c in _FORBIDDEN_PATH_CHARS for c in self.path):
            raise ValueError(
                f"user_files path contains control characters: {self.path!r}"
            )
        # Always POSIX semantics: paths resolve under the run directory inside
        # the operator pod, regardless of where the YAML was authored.
        p = PurePosixPath(self.path)
        if p.is_absolute():
            raise ValueError(f"user_files absolute path rejected: {self.path!r}")
        if any(part == ".." for part in p.parts):
            raise ValueError(f"user_files path '..' rejected: {self.path!r}")
        return self

    @model_validator(mode="after")
    def _resolve_format(self) -> UserFile:
        if self.format is None:
            self.format = "text" if isinstance(self.content, str) else "json"
        if self.format in {"json", "yaml"} and isinstance(self.content, str):
            raise ValueError(
                f"user_files path={self.path!r}: format={self.format!r} "
                "requires structured content (dict/list/scalar); got str. "
                "Wrap in a dict or set format: text."
            )
        if self.format == "text" and not isinstance(self.content, str):
            raise ValueError(
                f"user_files path={self.path!r}: format='text' requires string content; "
                f"got {type(self.content).__name__}."
            )
        return self


@dataclass(frozen=True, slots=True)
class RunMeta:
    """Run-time identity for a benchmark execution.

    Built once at run start and passed into build_user_file_context.
    """

    epoch: str
    """Run epoch (e.g. '1714000000')."""

    job_name: str
    """AIPerfJob name in k8s; --artifact-dir basename locally."""

    namespace: str
    """K8s namespace; empty string locally."""


def build_user_file_context(
    config: BenchmarkConfig,
    run_meta: RunMeta,
    run_dir: Path,
    *,
    variables: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the jinja2 context dict for user_files rendering.

    Args:
        config: Resolved BenchmarkConfig (the swept body). Provides
            ``endpoint.urls`` and ``get_model_names()`` for system-injected
            names. Does NOT carry ``variables`` — pass them via the
            keyword-only ``variables`` argument.
        run_meta: Identity for the run (epoch, job_name, namespace).
        run_dir: Absolute path to the run directory on local disk.
        variables: Envelope-level Jinja variables (from
            ``AIPerfConfig.variables`` or ``BenchmarkRun.variables``).
            User-provided names. None or empty dict means no user variables.

    Returns:
        A dict combining user-provided variables with system-injected names.
        On collision, injected wins and a WARNING is logged.

    Side effects:
        Logs WARNING for each shadowed user variable.
    """
    user_vars = dict(variables or {})
    # BenchmarkConfig.models is a ModelsAdvanced, not a list — go through the
    # canonical helper which flattens .items[*].name into a list[str].
    models = config.get_model_names()
    endpoint_urls = config.endpoint.urls or []
    injected = {
        "epoch": run_meta.epoch,
        "job_name": run_meta.job_name,
        "namespace": run_meta.namespace,
        "model": models[0] if models else "",
        "endpoint_url": endpoint_urls[0] if endpoint_urls else "",
        "artifact_dir": str(run_dir),
    }
    for name in injected:
        if name in user_vars:
            _logger.warning(
                "variable %r in artifacts.user_files context shadowed by "
                "system-injected name; rename to avoid",
                name,
            )
    return {**user_vars, **injected}


def materialize_user_files(
    files: list[UserFile],
    run_dir: Path,
    context: dict[str, Any],
) -> None:
    """Render and write all user_files to the run directory.

    Aborts on first failure; partial writes may have already happened on disk
    when this raises (acceptable: caller treats this as a fatal pre-run error
    and the run dir is owned by this run).

    Args:
        files: User-declared file specs from artifacts.user_files.
        run_dir: Absolute path to the run directory.
        context: Jinja2 context dict from build_user_file_context.

    Raises:
        UserFileError: On render failure, path-escape, or write failure. The
            message names the offending file path.
    """
    if not files:
        return
    run_dir_resolved = run_dir.resolve()
    for entry in files:
        rendered = _render_content(entry, context)
        target = run_dir / entry.path
        # Resolve and check parent BEFORE mkdir so a symlinked intermediate
        # directory cannot cause us to create directories outside run_dir.
        parent_resolved = target.parent.resolve()
        try:
            parent_resolved.relative_to(run_dir_resolved)
        except ValueError as exc:
            raise UserFileError(
                f"user_files path={entry.path!r} parent resolves to "
                f"{parent_resolved} which is outside run dir {run_dir_resolved}"
            ) from exc
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise UserFileError(
                f"user_files mkdir failed: path={entry.path!r} "
                f"parent={target.parent} errno={exc!s}"
            ) from exc
        target_resolved = target.resolve()
        try:
            target_resolved.relative_to(run_dir_resolved)
        except ValueError as exc:
            raise UserFileError(
                f"user_files path={entry.path!r} resolved to {target_resolved} "
                f"which is outside run dir {run_dir_resolved}"
            ) from exc
        try:
            _write(entry, target_resolved, rendered)
        except OSError as exc:
            raise UserFileError(
                f"user_files write failed: path={entry.path!r} "
                f"resolved={target_resolved} errno={exc!s}"
            ) from exc


def _render_content(entry: UserFile, context: dict[str, Any]) -> Any:
    """Recursively render jinja2 strings in entry.content with strict undefined."""
    coerce = entry.format != "text"
    try:
        return _render_recursive(entry.content, context, coerce=coerce)
    except jinja2.UndefinedError as exc:
        raise UserFileError(
            f"user_files render failed: path={entry.path!r} undefined variable: "
            f"{exc!s}. Available context keys: {sorted(context.keys())}"
        ) from exc
    except jinja2.TemplateError as exc:
        raise UserFileError(
            f"user_files render failed: path={entry.path!r} jinja2 error: {exc!s}"
        ) from exc


def _render_recursive(value: Any, context: dict[str, Any], *, coerce: bool) -> Any:
    if isinstance(value, str):
        # Skip plain strings without jinja markers — huge speedup on large dicts,
        # AND preserves user-authored literals (a string "42" stays "42" rather
        # than being coerced to int 42 by the json/yaml path).
        if not any(m in value for m in _JINJA_MARKERS):
            return value
        rendered = _USER_FILES_ENV.from_string(value).render(**context)
        return _coerce_scalar(rendered) if coerce else rendered
    if isinstance(value, dict):
        return {
            k: _render_recursive(v, context, coerce=coerce) for k, v in value.items()
        }
    if isinstance(value, list):
        return [_render_recursive(v, context, coerce=coerce) for v in value]
    return value


# Mirrors loader/jinja.py::_coerce_rendered — keep them in sync if either changes.
def _coerce_scalar(rendered: str) -> Any:
    """Coerce a rendered string to bool/int/float when unambiguous.

    Matches loader/jinja.py::_coerce_rendered so structured output (json/yaml)
    treats ``"{{ n }}"`` with ``n=42`` as the int ``42`` rather than ``"42"``.
    """
    low = rendered.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        return int(rendered)
    except ValueError:
        pass
    try:
        return float(rendered)
    except ValueError:
        pass
    return rendered


def _write(entry: UserFile, target: Path, rendered: Any) -> None:
    if entry.format == "json":
        target.write_bytes(orjson.dumps(rendered, option=orjson.OPT_INDENT_2))
        return
    if entry.format == "yaml":
        target.write_text(
            yaml.safe_dump(
                rendered,
                sort_keys=False,
                default_flow_style=False,
            )
        )
        return
    # text: write rendered string verbatim. _USER_FILES_ENV has
    # keep_trailing_newline=True so the user's exact content (including or
    # excluding a trailing newline) round-trips unchanged.
    text = rendered if isinstance(rendered, str) else str(rendered)
    target.write_text(text)
