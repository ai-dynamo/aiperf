# `artifacts.user_files` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add user-defined templated output files to `ArtifactsConfig.user_files`, materialized into the run directory before benchmark start, with strict jinja templating.

**Architecture:** New `aiperf.config.user_files` module owning the model, error type, context builder, and materializer. Wires in at the existing artifact-dir creation site (`config/resolvers.py:resolve_artifact_dir`) so both `aiperf profile` (local) and the Kube controller pod get it for free. Existing jinja renderer is tightened to `StrictUndefined` globally; in-tree templates audited and fixed.

**Tech Stack:** Python 3.10+, Pydantic v2, jinja2, pytest, orjson, PyYAML.

**Spec:** `docs/superpowers/specs/2026-04-24-artifacts-user-files-design.md`

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/aiperf/config/user_files.py` | **Create** | `UserFile` model, `UserFileError`, `RunMeta`, `build_user_file_context`, `materialize_user_files` |
| `src/aiperf/config/artifacts.py` | Modify | Add `user_files: list[UserFile]` field on `ArtifactsConfig` |
| `src/aiperf/config/config.py` | Modify | Add `variables: dict[str, Any]` field on `AIPerfConfig` |
| `src/aiperf/config/loader/core.py` | Modify | Stop popping `variables` from data dict (line 187) |
| `src/aiperf/config/loader/jinja.py` | Modify | Switch renderer to `jinja2.Environment(undefined=StrictUndefined, autoescape=False)`; add `artifacts.user_files` to `SKIP_TEMPLATE_FIELDS` |
| `src/aiperf/config/resolvers.py` | Modify | Call `materialize_user_files` immediately after `artifact_dir.mkdir(...)` (line 76) |
| `tests/unit/config/test_user_files.py` | **Create** | Model, validation, format inference, builder, materializer unit tests |
| `tests/unit/config/loader/test_jinja_strict_undefined.py` | **Create** | Regression: undefined variable → `ConfigurationError` with name in message |
| `tests/component_integration/test_user_files_e2e.py` | **Create** | End-to-end via `aiperf profile -c config.yaml`; failure paths abort the run |
| `src/aiperf/config/templates/embeddings.yaml` | Audit | Verify every `{{ x }}` resolves; fix or document defaults |
| `src/aiperf/config/templates/latency_test.yaml` | Audit | (same) |
| `src/aiperf/config/templates/request_cancellation.yaml` | Audit | (same) |
| `src/aiperf/config/templates/jinja2_variables.yaml` | Audit | (same) |
| `tests/**/*.yaml` | Audit | Grep for `{{ ... }}`; fix typos that previously rendered empty |
| `docs/kubernetes/user-files.md` | **Create** | User-facing reference: schema, injected names, examples |
| `docs/kubernetes/configuration.md` | Modify | Add link to `user-files.md` |
| `docs/dev/patterns.md` | Modify | Add a one-paragraph "User-defined templated outputs" pattern |
| `CLAUDE.md` + `.github/copilot-instructions.md` + `.cursor/rules/python.mdc` | Modify | One-line note on `artifacts.user_files` |
| `llms.txt` | Modify | Link the new `user-files.md` page |

---

## Task 1: Preserve `variables:` block on resolved config

**Why first:** The user_files renderer needs access to user variables at run-time. Currently they're popped at load time. This unblocks every later render-time task.

**Files:**
- Modify: `src/aiperf/config/config.py` (add `variables` field to `AIPerfConfig`)
- Modify: `src/aiperf/config/loader/core.py:187` (remove `data.pop("variables", None)`)
- Test: `tests/unit/config/test_variables_persist.py` (new)

- [ ] **Step 1.1: Write the failing test**

```python
# tests/unit/config/test_variables_persist.py
"""Variables block must persist on the resolved config so run-time renderers can use it."""
from aiperf.config.loader import load_config_from_string


def test_variables_block_persists_on_resolved_config():
    yaml_str = """
variables:
  isl: 1024
  osl: 512
models:
  - test/model
endpoint:
  type: chat
  urls: ["http://localhost:8000"]
"""
    config = load_config_from_string(yaml_str)
    assert config.variables == {"isl": 1024, "osl": 512}


def test_variables_default_empty_when_not_declared():
    yaml_str = """
models:
  - test/model
endpoint:
  type: chat
  urls: ["http://localhost:8000"]
"""
    config = load_config_from_string(yaml_str)
    assert config.variables == {}
```

- [ ] **Step 1.2: Run test to verify it fails**

```bash
uv run pytest tests/unit/config/test_variables_persist.py -n auto -v
```
Expected: FAIL with `AttributeError` or pydantic `ValidationError` on the `variables` attr.

- [ ] **Step 1.3: Add `variables` field to `AIPerfConfig`**

In `src/aiperf/config/config.py`, locate the `AIPerfConfig` class definition and add (preserving existing field ordering and Pydantic conventions for the file):

```python
variables: Annotated[
    dict[str, Any],
    Field(
        default_factory=dict,
        description=(
            "User-defined variables for jinja2 templating in this config. "
            "Resolved at config load and preserved for run-time renderers "
            "(e.g. artifacts.user_files)."
        ),
    ),
] = {}
```

If `Any` isn't already imported in `config.py`, add `from typing import Any`. If `Annotated` and `Field` aren't already imported there, follow the existing imports in that file.

- [ ] **Step 1.4: Stop popping `variables` in the loader**

In `src/aiperf/config/loader/core.py` around line 187, delete the line:

```python
data.pop("variables", None)
```

The `variables` key now flows into `_validate_config_dict` and lands on the resolved `AIPerfConfig.variables` field.

- [ ] **Step 1.5: Run test to verify it passes**

```bash
uv run pytest tests/unit/config/test_variables_persist.py -n auto -v
```
Expected: PASS, both tests green.

- [ ] **Step 1.6: Run the full unit-test suite to catch regressions**

```bash
uv run pytest tests/unit/ -n auto
```
Expected: All green. If any test breaks because it asserted absence of `variables` on a resolved config, that's the regression and the test is wrong — adjust it.

- [ ] **Step 1.7: Commit**

```bash
git add src/aiperf/config/config.py src/aiperf/config/loader/core.py tests/unit/config/test_variables_persist.py
git commit -s -m "$(cat <<'EOF'
feat(config): preserve variables block on resolved AIPerfConfig

Variables were popped at load time after the load-time render pass. They
now persist on AIPerfConfig.variables so run-time renderers (notably
artifacts.user_files in the next change) can resolve them.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Switch jinja renderer to `StrictUndefined`

**Files:**
- Modify: `src/aiperf/config/loader/jinja.py` (renderer call)
- Test: `tests/unit/config/loader/test_jinja_strict_undefined.py` (new)
- Audit: in-tree templates (see step 2.5)

- [ ] **Step 2.1: Write the failing regression test**

```python
# tests/unit/config/loader/test_jinja_strict_undefined.py
"""Strict mode: undefined jinja2 variables raise ConfigurationError naming the variable."""
import pytest

from aiperf.common.exceptions import ConfigurationError
from aiperf.config.loader.jinja import render_jinja2_templates


def test_undefined_variable_raises_configuration_error():
    data = {"foo": "{{ undefined_var }}"}
    with pytest.raises(ConfigurationError) as exc_info:
        render_jinja2_templates(data, context={})
    assert "undefined_var" in str(exc_info.value)


def test_defined_variable_renders_normally():
    data = {"foo": "{{ defined }}"}
    result = render_jinja2_templates(data, context={"defined": 42})
    assert result == {"foo": 42}
```

- [ ] **Step 2.2: Run test to verify it fails**

```bash
uv run pytest tests/unit/config/loader/test_jinja_strict_undefined.py -n auto -v
```
Expected: FAIL on first test (lenient mode renders empty string, no exception).

- [ ] **Step 2.3: Switch renderer to `StrictUndefined`**

In `src/aiperf/config/loader/jinja.py`, replace the bare `jinja2.Template(data)` usage with a module-level `Environment`:

```python
# Near the top of the file with other module-level constants:
_JINJA_ENV = jinja2.Environment(
    undefined=jinja2.StrictUndefined,
    autoescape=False,
    keep_trailing_newline=True,
)


# In _render_template_string (around line 84), replace:
#   template = jinja2.Template(data)
#   rendered = template.render(**context)
# with:
template = _JINJA_ENV.from_string(data)
rendered = template.render(**context)
```

The `except jinja2.TemplateError as e:` block already wraps `UndefinedError` (it's a subclass), so the existing `ConfigurationError` raise picks up strict failures unchanged. Verify by reading the catch block (line 86) — no other change required there.

- [ ] **Step 2.4: Run regression test**

```bash
uv run pytest tests/unit/config/loader/test_jinja_strict_undefined.py -n auto -v
```
Expected: Both tests PASS.

- [ ] **Step 2.5: Audit in-tree templates**

Run the audit grep:

```bash
grep -rn "{{" src/aiperf/config/templates/ tests/ dev/deploy/ dev/aiperf-runs/ 2>/dev/null | grep -v ".git/"
```

For each match, verify the referenced variable is resolvable (declared in the template's own `variables:` block, declared in the test fixture's context, or supplied via `${VAR}` env-substitution upstream of the jinja pass). Fix typos by either:
- Renaming the reference to match the declared variable, or
- Adding the missing variable with a sensible default in the same file's `variables:` block.

Concretely-known files to check (from spec section 7.2):
- `src/aiperf/config/templates/embeddings.yaml`
- `src/aiperf/config/templates/latency_test.yaml`
- `src/aiperf/config/templates/request_cancellation.yaml`
- `src/aiperf/config/templates/jinja2_variables.yaml`

If any template uses a name that's expected to be supplied externally (e.g. by CLI flag), document the contract by adding `# Required: X must be defined upstream` as a YAML comment near the reference. Do not silence with a `default('')` filter — the whole point of this change is that empty defaults must be explicit choices.

- [ ] **Step 2.6: Run full unit suite to surface any latent breakage**

```bash
uv run pytest tests/unit/ -n auto
```
Expected: All green. Any newly-failing test is a config that depended on lenient rendering — fix the config (or fixture), not the renderer.

- [ ] **Step 2.7: Commit**

```bash
git add src/aiperf/config/loader/jinja.py tests/unit/config/loader/test_jinja_strict_undefined.py src/aiperf/config/templates/ tests/
git commit -s -m "$(cat <<'EOF'
feat(config): tighten jinja2 renderer to StrictUndefined

Undefined variables previously rendered as empty strings — silent bugs
that downstream parsers had to catch. Strict mode raises
ConfigurationError naming the variable. In-tree templates and fixtures
audited and fixed for resolvability under strict mode.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `UserFile` model + path/format validation

**Files:**
- Create: `src/aiperf/config/user_files.py`
- Test: `tests/unit/config/test_user_files.py`

- [ ] **Step 3.1: Write the failing model tests**

```python
# tests/unit/config/test_user_files.py
"""UserFile model: path validation, format inference, content typing."""
import pytest
from pydantic import ValidationError

from aiperf.config.user_files import UserFile


# --- Path validation ----------------------------------------------------------

@pytest.mark.parametrize(
    "path",
    [
        "input_config.json",
        "meta/notes.md",
        "deep/nested/info.txt",
    ],
)
def test_valid_paths_accepted(path):
    f = UserFile(path=path, content="ok")
    assert f.path == path


@pytest.mark.parametrize(
    "path,reason_substring",
    [
        ("/etc/passwd", "absolute"),
        ("../escape.json", ".."),
        ("foo/../bar.json", ".."),
        ("", "empty"),
        ("with\x00null.json", "control"),
    ],
)
def test_invalid_paths_rejected(path, reason_substring):
    with pytest.raises(ValidationError) as exc_info:
        UserFile(path=path, content="ok")
    assert reason_substring in str(exc_info.value).lower()


# --- Format inference ---------------------------------------------------------

def test_format_inferred_text_for_string_content():
    f = UserFile(path="x.txt", content="hello")
    assert f.format == "text"


def test_format_inferred_json_for_dict_content():
    f = UserFile(path="x.json", content={"a": 1})
    assert f.format == "json"


def test_format_inferred_json_for_list_content():
    f = UserFile(path="x.json", content=[1, 2, 3])
    assert f.format == "json"


def test_explicit_yaml_format_with_dict_content():
    f = UserFile(path="x.yaml", format="yaml", content={"a": 1})
    assert f.format == "yaml"


# --- Format/content mismatch --------------------------------------------------

def test_json_format_with_string_content_rejected():
    with pytest.raises(ValidationError) as exc_info:
        UserFile(path="x.json", format="json", content="raw string")
    assert "structured" in str(exc_info.value).lower()


def test_text_format_with_dict_content_rejected():
    with pytest.raises(ValidationError) as exc_info:
        UserFile(path="x.txt", format="text", content={"a": 1})
    assert "string" in str(exc_info.value).lower()
```

- [ ] **Step 3.2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/config/test_user_files.py -n auto -v
```
Expected: FAIL — module does not exist yet.

- [ ] **Step 3.3: Create `user_files.py` with the `UserFile` model**

```python
# src/aiperf/config/user_files.py
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""User-defined templated output files materialized into the run directory.

See docs/kubernetes/user-files.md for the user-facing reference.
"""
from __future__ import annotations

from pathlib import PurePosixPath
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from aiperf.common.exceptions import AIPerfError
from aiperf.config.base import BaseConfig


_FORBIDDEN_PATH_CHARS = frozenset(chr(c) for c in range(32)) | {"\x7f"}


class UserFileError(AIPerfError):
    """Raised when a user_files entry fails validation, render, or write."""


class UserFile(BaseConfig):
    """One user-declared output file rendered into the run directory before benchmark start.

    Path is relative to the run directory; subdirectories are allowed; absolute
    paths and any segment equal to '..' are rejected. Content is rendered with
    jinja2 against a documented context (variables: + system-injected names);
    see docs/kubernetes/user-files.md.
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
    def _validate_path(self) -> "UserFile":
        if not self.path:
            raise ValueError("user_files entry has empty path")
        if any(c in _FORBIDDEN_PATH_CHARS for c in self.path):
            raise ValueError(f"user_files path contains control characters: {self.path!r}")
        p = PurePosixPath(self.path)
        if p.is_absolute():
            raise ValueError(f"user_files absolute path rejected: {self.path!r}")
        if any(part == ".." for part in p.parts):
            raise ValueError(f"user_files path '..' rejected: {self.path!r}")
        return self

    @model_validator(mode="after")
    def _resolve_format(self) -> "UserFile":
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
```

If `BaseConfig` lives at a different import path in this repo, follow what `src/aiperf/config/artifacts.py` does (note its `class ArtifactsConfig(BaseConfig):` line and its imports). Match that exactly.

- [ ] **Step 3.4: Run tests to verify they pass**

```bash
uv run pytest tests/unit/config/test_user_files.py -n auto -v
```
Expected: All parametrized cases PASS.

- [ ] **Step 3.5: Commit**

```bash
git add src/aiperf/config/user_files.py tests/unit/config/test_user_files.py
git commit -s -m "$(cat <<'EOF'
feat(config): add UserFile model with path and format validation

New aiperf.config.user_files module hosts the UserFile pydantic model.
Path validation rejects absolute, '..', empty, and control chars.
Format is inferred from content type (text for str, json otherwise) and
mismatches between explicit format and content type are rejected at
load time.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `RunMeta` + context builder + materializer

**Files:**
- Modify: `src/aiperf/config/user_files.py`
- Modify: `tests/unit/config/test_user_files.py`

- [ ] **Step 4.1: Write the failing tests**

Append to `tests/unit/config/test_user_files.py`:

```python
import json
from pathlib import Path

import pytest
import yaml

from aiperf.config.user_files import (
    RunMeta,
    UserFile,
    UserFileError,
    build_user_file_context,
    materialize_user_files,
)


# --- build_user_file_context --------------------------------------------------

def _stub_config(variables=None, model="m", url="http://x"):
    """Minimal duck-typed config for build_user_file_context."""
    from types import SimpleNamespace
    return SimpleNamespace(
        variables=variables or {},
        benchmark=SimpleNamespace(
            models=[model],
            endpoint=SimpleNamespace(urls=[url]),
        ),
    )


def test_context_includes_injected_names(tmp_path):
    config = _stub_config(variables={"isl": 1024})
    meta = RunMeta(epoch="1714", job_name="run-1", namespace="ns")
    ctx = build_user_file_context(config, meta, run_dir=tmp_path)
    assert ctx["epoch"] == "1714"
    assert ctx["job_name"] == "run-1"
    assert ctx["namespace"] == "ns"
    assert ctx["model"] == "m"
    assert ctx["endpoint_url"] == "http://x"
    assert ctx["artifact_dir"] == str(tmp_path)
    assert ctx["isl"] == 1024


def test_collision_injected_wins_and_warns(caplog):
    config = _stub_config(variables={"epoch": "user-supplied"})
    meta = RunMeta(epoch="1714", job_name="r", namespace="")
    with caplog.at_level("WARNING"):
        ctx = build_user_file_context(config, meta, run_dir=Path("/tmp"))
    assert ctx["epoch"] == "1714"
    assert any("epoch" in r.message and "shadow" in r.message.lower() for r in caplog.records)


# --- materialize_user_files ---------------------------------------------------

def test_materialize_json_renders_int_as_int(tmp_path):
    files = [UserFile(path="a.json", format="json", content={"n": "{{ x }}"})]
    materialize_user_files(files, run_dir=tmp_path, context={"x": 42})
    data = json.loads((tmp_path / "a.json").read_text())
    assert data == {"n": 42}  # not "42"


def test_materialize_yaml_round_trip(tmp_path):
    files = [UserFile(path="a.yaml", format="yaml", content={"k": "{{ v }}"})]
    materialize_user_files(files, run_dir=tmp_path, context={"v": "hello"})
    data = yaml.safe_load((tmp_path / "a.yaml").read_text())
    assert data == {"k": "hello"}


def test_materialize_text_preserves_newlines(tmp_path):
    files = [UserFile(path="notes.md", content="line {{ n }}\nend")]
    materialize_user_files(files, run_dir=tmp_path, context={"n": 1})
    assert (tmp_path / "notes.md").read_text() == "line 1\nend"


def test_materialize_subdir_creates_intermediate_dirs(tmp_path):
    files = [UserFile(path="meta/sub/a.json", content={"a": 1})]
    materialize_user_files(files, run_dir=tmp_path, context={})
    assert (tmp_path / "meta" / "sub" / "a.json").exists()


def test_materialize_undefined_variable_raises_with_path(tmp_path):
    files = [UserFile(path="a.txt", content="{{ missing }}")]
    with pytest.raises(UserFileError) as exc_info:
        materialize_user_files(files, run_dir=tmp_path, context={})
    msg = str(exc_info.value)
    assert "missing" in msg
    assert "a.txt" in msg


def test_materialize_overwrites_existing(tmp_path):
    (tmp_path / "a.txt").write_text("old")
    files = [UserFile(path="a.txt", content="new")]
    materialize_user_files(files, run_dir=tmp_path, context={})
    assert (tmp_path / "a.txt").read_text() == "new"


def test_materialize_symlink_escape_rejected(tmp_path):
    outside = tmp_path.parent / "outside"
    outside.mkdir(exist_ok=True)
    (tmp_path / "evil").symlink_to(outside)
    files = [UserFile(path="evil/a.txt", content="x")]
    with pytest.raises(UserFileError) as exc_info:
        materialize_user_files(files, run_dir=tmp_path, context={})
    assert "outside run dir" in str(exc_info.value).lower() or "escape" in str(exc_info.value).lower()


def test_materialize_write_failure_raises(tmp_path):
    import os
    (tmp_path / "a.txt").write_text("seed")
    os.chmod(tmp_path, 0o500)  # read+exec only
    try:
        files = [UserFile(path="b.txt", content="x")]
        with pytest.raises(UserFileError):
            materialize_user_files(files, run_dir=tmp_path, context={})
    finally:
        os.chmod(tmp_path, 0o700)
```

- [ ] **Step 4.2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/config/test_user_files.py -n auto -v
```
Expected: FAIL on every new test — symbols don't exist.

- [ ] **Step 4.3: Add `RunMeta`, `build_user_file_context`, `materialize_user_files` to `user_files.py`**

Append to `src/aiperf/config/user_files.py`:

```python
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jinja2
import orjson
import yaml

from aiperf.common.logging import AIPerfLogger  # match the project's logger pattern; see existing modules

_logger = logging.getLogger(__name__)


# Reuse the same StrictUndefined env shape as loader/jinja.py for consistency.
_USER_FILES_ENV = jinja2.Environment(
    undefined=jinja2.StrictUndefined,
    autoescape=False,
    keep_trailing_newline=True,
)


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


_INJECTED_NAMES = frozenset({"epoch", "job_name", "namespace", "model", "endpoint_url", "artifact_dir"})


def build_user_file_context(
    config: Any,
    run_meta: RunMeta,
    run_dir: Path,
) -> dict[str, Any]:
    """Build the jinja2 context dict for user_files rendering.

    Args:
        config: Resolved AIPerfConfig (must expose .variables, .benchmark.models[0],
            .benchmark.endpoint.urls[0]).
        run_meta: Identity for the run (epoch, job_name, namespace).
        run_dir: Absolute path to the run directory on local disk.

    Returns:
        A dict combining user variables (tier 1) with system-injected names (tier 2).
        On collision, injected wins and a WARNING is logged.

    Side effects:
        Logs WARNING for each shadowed user variable.
    """
    user_vars = dict(getattr(config, "variables", {}) or {})
    injected = {
        "epoch": run_meta.epoch,
        "job_name": run_meta.job_name,
        "namespace": run_meta.namespace,
        "model": config.benchmark.models[0] if config.benchmark.models else "",
        "endpoint_url": config.benchmark.endpoint.urls[0] if config.benchmark.endpoint.urls else "",
        "artifact_dir": str(run_dir),
    }
    for name in injected:
        if name in user_vars:
            _logger.warning(
                "variable %r in artifacts.user_files context shadowed by system-injected name; rename to avoid",
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
        target = (run_dir / entry.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target_resolved = target.resolve()
        try:
            target_resolved.relative_to(run_dir_resolved)
        except ValueError as exc:
            raise UserFileError(
                f"user_files path={entry.path!r} resolved to {target_resolved} which is outside run dir {run_dir_resolved}"
            ) from exc
        try:
            _write(entry, target_resolved, rendered)
        except OSError as exc:
            raise UserFileError(
                f"user_files write failed: path={entry.path!r} resolved={target_resolved} errno={exc!s}"
            ) from exc


def _render_content(entry: UserFile, context: dict[str, Any]) -> Any:
    """Recursively render jinja2 strings in entry.content with strict undefined."""
    try:
        return _render_recursive(entry.content, context, entry.path)
    except jinja2.UndefinedError as exc:
        raise UserFileError(
            f"user_files render failed: path={entry.path!r} undefined variable: {exc!s}. "
            f"Available context keys: {sorted(context.keys())}"
        ) from exc
    except jinja2.TemplateError as exc:
        raise UserFileError(
            f"user_files render failed: path={entry.path!r} jinja2 error: {exc!s}"
        ) from exc


def _render_recursive(value: Any, context: dict[str, Any], path: str) -> Any:
    if isinstance(value, str):
        if "{{" not in value and "{%" not in value:
            return value
        return _USER_FILES_ENV.from_string(value).render(**context)
    if isinstance(value, dict):
        return {k: _render_recursive(v, context, path) for k, v in value.items()}
    if isinstance(value, list):
        return [_render_recursive(v, context, path) for v in value]
    return value


def _write(entry: UserFile, target: Path, rendered: Any) -> None:
    if entry.format == "json":
        target.write_bytes(orjson.dumps(rendered, option=orjson.OPT_INDENT_2))
        return
    if entry.format == "yaml":
        target.write_text(yaml.safe_dump(rendered, sort_keys=False, default_flow_style=False))
        return
    # text
    text = rendered if isinstance(rendered, str) else str(rendered)
    if not text.endswith("\n"):
        text = text + "\n"
    target.write_text(text)
```

Notes for the implementer:
- If `aiperf.common.logging.AIPerfLogger` is the project's preferred logger pattern (it is — see `CLAUDE.md`'s "Lambda for expensive logs" rule and the existing `loader/jinja.py` style), prefer it over `logging.getLogger`. Match the surrounding files in `src/aiperf/config/`.
- The strict-undefined render path is duplicated here intentionally: `loader/jinja.py` is for load-time, this module is for run-time. The two could share an env later if a refactor shows up clean, but don't pre-emptively factor.
- `_render_recursive` skips strings with no `{{` for speed (huge wins on large dicts).

- [ ] **Step 4.4: Run tests to verify they pass**

```bash
uv run pytest tests/unit/config/test_user_files.py -n auto -v
```
Expected: All tests PASS, including the symlink-escape and chmod-write-failure cases.

- [ ] **Step 4.5: Commit**

```bash
git add src/aiperf/config/user_files.py tests/unit/config/test_user_files.py
git commit -s -m "$(cat <<'EOF'
feat(config): add user_files context builder and materializer

RunMeta dataclass + build_user_file_context (variables + injected names
with collision warning) + materialize_user_files (render → path-escape
check → write, fail-fast on first error). Strict jinja, format-aware
serialization (json via orjson, yaml via PyYAML, text raw with trailing
newline). UserFileError surfaces path and cause in every message.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Wire `user_files` into `ArtifactsConfig` + skip at load-time render

**Files:**
- Modify: `src/aiperf/config/artifacts.py` (add field)
- Modify: `src/aiperf/config/loader/jinja.py` (extend `SKIP_TEMPLATE_FIELDS`)
- Test: `tests/unit/config/test_artifacts_user_files.py` (new)

- [ ] **Step 5.1: Write the failing test**

```python
# tests/unit/config/test_artifacts_user_files.py
"""ArtifactsConfig.user_files: wired in, surviving load, not rendered at load time."""
from aiperf.config.loader import load_config_from_string


def test_user_files_default_empty():
    yaml_str = """
models:
  - test/model
endpoint:
  type: chat
  urls: ["http://localhost:8000"]
"""
    config = load_config_from_string(yaml_str)
    assert config.artifacts.user_files == []


def test_user_files_round_trips_through_config_load():
    yaml_str = """
variables:
  isl: 1024
artifacts:
  user_files:
    - path: input_config.json
      format: json
      content:
        isl: "{{ isl }}"
        note: "fixed string"
models:
  - test/model
endpoint:
  type: chat
  urls: ["http://localhost:8000"]
"""
    config = load_config_from_string(yaml_str)
    assert len(config.artifacts.user_files) == 1
    entry = config.artifacts.user_files[0]
    assert entry.path == "input_config.json"
    assert entry.format == "json"
    # Critical: load-time render must NOT have evaluated {{ isl }} on user_files content.
    # The template string survives verbatim for run-time render.
    assert entry.content == {"isl": "{{ isl }}", "note": "fixed string"}
```

- [ ] **Step 5.2: Run test to verify it fails**

```bash
uv run pytest tests/unit/config/test_artifacts_user_files.py -n auto -v
```
Expected: FAIL on first test — `ArtifactsConfig` has no `user_files` field.

- [ ] **Step 5.3: Add `user_files` field to `ArtifactsConfig`**

In `src/aiperf/config/artifacts.py`, around the `ArtifactsConfig` class (line 44 area), add:

```python
from aiperf.config.user_files import UserFile  # at the top of the file
```

And as a new field within `ArtifactsConfig`:

```python
user_files: Annotated[
    list[UserFile],
    Field(
        default_factory=list,
        description=(
            "User-defined templated files materialized into the run directory before "
            "the benchmark begins. See docs/kubernetes/user-files.md."
        ),
    ),
] = []
```

If there's a circular-import risk (artifacts.py imported by user_files.py transitively), keep the import inside the field type via `from __future__ import annotations` (already common in the project) and use the string form `list["UserFile"]`.

- [ ] **Step 5.4: Add `user_files` content path to load-time skip set**

In `src/aiperf/config/loader/jinja.py`, locate `SKIP_TEMPLATE_FIELDS` and add the path:

```python
SKIP_TEMPLATE_FIELDS = frozenset({
    # ... existing entries ...
    "artifacts.user_files",
})
```

The skip semantics in this codebase apply to entire subtrees — the existing matcher prefix-matches, so `artifacts.user_files.0.content.foo` will be skipped under `artifacts.user_files`. Verify by reading `_render_template_string`'s skip check (around the top of `_render_template_string` or its caller).

If the existing matcher does NOT prefix-match (read it carefully), adjust the skip mechanism to handle the wildcard properly: anything whose path starts with `artifacts.user_files.` must be skipped, including arbitrarily-nested string leaves. Add a unit test in `test_jinja_strict_undefined.py` covering that behavior.

- [ ] **Step 5.5: Run test to verify it passes**

```bash
uv run pytest tests/unit/config/test_artifacts_user_files.py -n auto -v
```
Expected: Both tests PASS, including the critical assertion that `{{ isl }}` survives unrendered in `entry.content`.

- [ ] **Step 5.6: Run full unit suite**

```bash
uv run pytest tests/unit/ -n auto
```
Expected: All green.

- [ ] **Step 5.7: Commit**

```bash
git add src/aiperf/config/artifacts.py src/aiperf/config/loader/jinja.py tests/unit/config/test_artifacts_user_files.py
git commit -s -m "$(cat <<'EOF'
feat(config): wire artifacts.user_files into ArtifactsConfig

Add user_files: list[UserFile] field to ArtifactsConfig and skip the
artifacts.user_files subtree during load-time jinja rendering — content
is rendered at run start where epoch/job_name/artifact_dir are known.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Wire the run-start call site

**Files:**
- Modify: `src/aiperf/config/resolvers.py` (around line 76, after `artifact_dir.mkdir`)

- [ ] **Step 6.1: Read the existing function**

Read `src/aiperf/config/resolvers.py` lines 50-90 to understand the `resolve_artifact_dir` function shape — it owns directory creation for both local and kube paths. Confirm that `run.cfg.artifacts.user_files` is reachable from `run.cfg`.

- [ ] **Step 6.2: Determine `RunMeta` derivation**

Inside `resolve_artifact_dir`, after `artifact_dir.mkdir(parents=True, exist_ok=True)`, derive `RunMeta`:

- `epoch`: prefer parsing from the artifact_dir path if it follows the `{epoch}_{job_name}` convention (see `src/aiperf/operator/results_layout.py` for the format); fall back to `str(int(time.time()))` for local runs.
- `job_name`: take it from `run.cfg.benchmark.job_name` if such a field exists; else from `artifact_dir.name` (or its parent, depending on layout); fall back to a deterministic hash of the run config.
- `namespace`: `run.cfg.runtime.kubernetes.namespace` if it exists, else "" (local).

If any of those derivations require code that doesn't exist yet (e.g. a `job_name` field on benchmark config), use the fallbacks consistently — the design tolerates an empty `namespace` and a synthesized `job_name` for local runs.

- [ ] **Step 6.3: Call `materialize_user_files` after directory creation**

In `src/aiperf/config/resolvers.py` immediately after the existing `artifact_dir.mkdir(parents=True, exist_ok=True)` (line 76), add:

```python
from aiperf.config.user_files import (
    RunMeta,
    build_user_file_context,
    materialize_user_files,
)

if run.cfg.artifacts.user_files:
    run_meta = RunMeta(
        epoch=_derive_epoch(artifact_dir),
        job_name=_derive_job_name(run, artifact_dir),
        namespace=_derive_namespace(run),
    )
    context = build_user_file_context(run.cfg, run_meta, run_dir=artifact_dir)
    materialize_user_files(run.cfg.artifacts.user_files, run_dir=artifact_dir, context=context)
```

Add the three private `_derive_*` helpers in the same file. They are tiny and not worth their own module — they are local to this resolver and tested via the e2e tests in Task 7.

- [ ] **Step 6.4: Defer integration test to Task 7**

Integration of the call site is covered end-to-end by the component-integration test
in Task 7 (which actually runs `aiperf profile -c config.yaml` and asserts the file
appears in the run dir). No separate unit test required for this step — the unit
tests for `materialize_user_files` (Task 4) plus the e2e test (Task 7) collectively
exercise the call path. If you find a logic-only branch in `_derive_epoch` /
`_derive_job_name` / `_derive_namespace` that the e2e test won't hit, add a focused
unit test for it; otherwise skip.

- [ ] **Step 6.5: Run unit tests**

```bash
uv run pytest tests/unit/ -n auto
```
Expected: All green.

- [ ] **Step 6.6: Commit**
git commit -s -m "$(cat <<'EOF'
feat(config): materialize user_files at artifact-dir creation

Hook materialize_user_files into resolve_artifact_dir, immediately after
artifact_dir.mkdir. RunMeta derived locally (epoch from path or wall
clock; job_name from artifact dir or benchmark config; namespace from
runtime config or empty). Both aiperf profile (local) and the controller
pod hit the same call site.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: End-to-end component-integration test

**Files:**
- Create: `tests/component_integration/test_user_files_e2e.py`

- [ ] **Step 7.1: Write the test**

```python
# tests/component_integration/test_user_files_e2e.py
"""End-to-end: user_files render and write during a real aiperf profile run."""
import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.component_integration
def test_user_files_e2e_renders_json_and_text(tmp_path, mock_server_url):
    """A config with two user_files entries produces correctly-rendered files."""
    config_path = tmp_path / "config.yaml"
    artifact_dir = tmp_path / "artifacts"
    config_path.write_text(f"""
variables:
  isl: 1024
  osl: 512
artifacts:
  dir: {artifact_dir}
  user_files:
    - path: input_config.json
      format: json
      content:
        isl: "{{{{ isl }}}}"
        osl: "{{{{ osl }}}}"
        endpoint: "{{{{ endpoint_url }}}}"
    - path: notes.txt
      content: |
        run for {{{{ model }}}}
models:
  - test/mock-model
endpoint:
  type: chat
  urls: ["{mock_server_url}"]
loadgen:
  request_count: 3
""")
    result = subprocess.run(
        [sys.executable, "-m", "aiperf", "profile", "-c", str(config_path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"

    # Find the run dir (artifacts/<something>/...)
    run_dirs = [p for p in artifact_dir.rglob("input_config.json")]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0].parent

    data = json.loads((run_dir / "input_config.json").read_text())
    assert data == {"isl": 1024, "osl": 512, "endpoint": mock_server_url}

    notes = (run_dir / "notes.txt").read_text()
    assert "test/mock-model" in notes


@pytest.mark.component_integration
def test_user_files_missing_variable_aborts_run(tmp_path, mock_server_url):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"""
artifacts:
  dir: {tmp_path / "artifacts"}
  user_files:
    - path: a.json
      content:
        x: "{{{{ does_not_exist }}}}"
models:
  - test/m
endpoint:
  type: chat
  urls: ["{mock_server_url}"]
""")
    result = subprocess.run(
        [sys.executable, "-m", "aiperf", "profile", "-c", str(config_path)],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode != 0
    assert "does_not_exist" in (result.stdout + result.stderr)
```

The `mock_server_url` fixture is the standard project fixture for component-integration tests. If it's named differently in this codebase, mirror what `tests/component_integration/test_*.py` use today (grep for the most common mock-server fixture name).

- [ ] **Step 7.2: Run the test**

```bash
uv run pytest tests/component_integration/test_user_files_e2e.py -m component_integration -n auto -v
```
Expected: PASS.

- [ ] **Step 7.3: Commit**

```bash
git add tests/component_integration/test_user_files_e2e.py
git commit -s -m "$(cat <<'EOF'
test(config): component-integration coverage for artifacts.user_files

End-to-end via aiperf profile -c config.yaml: JSON + text user_files
render with structured types preserved (int as int) and the run aborts
with a clear error when a variable is undefined.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: User-facing documentation

**Files:**
- Create: `docs/kubernetes/user-files.md`
- Modify: `docs/kubernetes/configuration.md` (add link)
- Modify: `docs/dev/patterns.md` (add pattern)
- Modify: `llms.txt` (link)

- [ ] **Step 8.1: Write `docs/kubernetes/user-files.md`**

Create the file with this exact content (project markdown style — no emojis):

````markdown
# User-Defined Output Files (`artifacts.user_files`)

`artifacts.user_files` lets you declare arbitrary templated output files that are
materialized into the run directory before the benchmark begins. Files are rendered
with jinja2 against the user `variables:` block plus a small set of system-injected
names. The same mechanism works for `aiperf profile` (local) and `AIPerfJob`
(Kubernetes) — both load the same config block.

## Quickstart

```yaml
variables:
  isl: 1024
  osl: 512

artifacts:
  user_files:
    - path: input_config.json
      format: json                 # optional; inferred from content type
      content:
        isl: "{{ isl }}"
        osl: "{{ osl }}"
        endpoint: "{{ endpoint_url }}"
        model: "{{ model }}"

    - path: meta/notes.md          # subdirectories allowed
      content: |
        Run {{ job_name }} started at {{ epoch }}.
        Targeting {{ model }} @ {{ endpoint_url }}.

models:
  - my-org/my-model
endpoint:
  type: chat
  urls: ["http://my-frontend:8000"]
```

Result in the run directory:

```
{artifact_dir}/{epoch}_{job_name}/
├── input_config.json
├── meta/
│   └── notes.md
└── ... (standard AIPerf artifacts)
```

## Schema

Each entry is:

| Field | Type | Required | Description |
|---|---|---|---|
| `path` | string | yes | Output path **relative** to the run directory. Subdirectories OK. Absolute paths and any segment equal to `..` are rejected. |
| `format` | `json` \| `yaml` \| `text` | no | Serialization format. If omitted: `text` when `content` is a string, `json` otherwise. |
| `content` | structured or string | yes | Templated value. Dict/list/scalar for `json`/`yaml`; string for `text`. Jinja2 expressions in any string leaf are rendered. |

Format/content compatibility:
- `format: json` or `format: yaml` requires structured `content` (dict/list/scalar).
- `format: text` requires string `content`.

## Templating context

Inside `content`, you can reference:

**1. User-declared variables** — anything you put in the top-level `variables:` block of your config.

**2. System-injected names** (stable API):

| Name | Type | Meaning |
|---|---|---|
| `epoch` | str | Run epoch identifier (e.g. `"1714000000"`). |
| `job_name` | str | AIPerfJob name in Kubernetes; `--artifact-dir` basename locally. |
| `namespace` | str | Kubernetes namespace; empty string locally. |
| `model` | str | First entry of `benchmark.models`. |
| `endpoint_url` | str | First entry of `benchmark.endpoint.urls`. |
| `artifact_dir` | str | Absolute path to the run directory. |

**Collision rule:** if a user `variables:` key shadows an injected name, the injected name wins and a `WARNING` is logged at startup. Rename your variable.

## Errors

These are all fatal — the benchmark does not start.

| Failure | Cause | Where you see it |
|---|---|---|
| Path validation | Absolute path, `..` segment, empty path, control chars | Config load (pydantic `ValidationError`) |
| Format/content mismatch | e.g. `format: json` with `content: "string"` | Config load (pydantic `ValidationError`) |
| Undefined variable | Template references a name not in context | Run start (`UserFileError`); message names the file path and the variable |
| Path escape | Resolved path is not inside the run directory | Run start (`UserFileError`) |
| Write failure | Disk full, permission denied, etc. | Run start (`UserFileError`); message includes resolved path and OS error |

In Kubernetes, controller-pod failures surface on `status.phase: Failed` with the
error in `status.conditions`.

## Use cases

- **Sidecar metadata** — produce an `input_config.json` for downstream tooling that expects
  the dynamo-style deployment-shape file.
- **Run notes** — write a `notes.md` summarizing what this run is for, who triggered it.
- **Manifests** — emit a manifest a downstream pipeline will read.

## Limitations (v1)

- **Pre-run only.** Files render before the benchmark starts. Post-run files that include
  results are tracked as a future extension.
- **Files always overwrite.** No `overwrite: false` safety net.
- **No `required: false`.** Every declared file must materialize successfully or the run aborts.
- **Strict undefined.** A typo in `{{ varaibles_name }}` is a hard error, not a silent empty string.
````

- [ ] **Step 8.2: Link from `docs/kubernetes/configuration.md`**

Find a sensible spot (the section that lists `artifacts:` fields, or the "see also" footer) and add:

```markdown
- [User-defined output files](user-files.md) — `artifacts.user_files` for templated sidecar files.
```

- [ ] **Step 8.3: Add pattern to `docs/dev/patterns.md`**

Append a short section near the templating-related patterns:

````markdown
### User-defined templated outputs

Configs may declare arbitrary output files via `artifacts.user_files`. Each entry
is `{path, format, content}`; content is rendered with jinja2 (StrictUndefined)
against `variables:` plus a documented set of injected names (`epoch`,
`job_name`, `namespace`, `model`, `endpoint_url`, `artifact_dir`). Files
materialize into the run directory before the benchmark starts. See
`docs/kubernetes/user-files.md` for the full reference.
````

- [ ] **Step 8.4: Link from `llms.txt`**

Add a line under the kubernetes docs section:

```
- docs/kubernetes/user-files.md - User-declared templated output files (artifacts.user_files)
```

- [ ] **Step 8.5: Commit**

```bash
git add docs/kubernetes/user-files.md docs/kubernetes/configuration.md docs/dev/patterns.md llms.txt
git commit -s -m "$(cat <<'EOF'
docs: reference for artifacts.user_files

User-facing reference at docs/kubernetes/user-files.md covers schema,
injected context names, errors, and use cases. Linked from
configuration.md, patterns.md, and llms.txt.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Three-file sync update

The project's CLAUDE.md three-file sync rule requires `CLAUDE.md`, `.github/copilot-instructions.md`, and `.cursor/rules/python.mdc` to stay identical (modulo the cursor file's frontmatter).

**Files:**
- Modify: `CLAUDE.md`
- Modify: `.github/copilot-instructions.md`
- Modify: `.cursor/rules/python.mdc`

- [ ] **Step 9.1: Add a one-line note in the appropriate section**

In each of the three files, find the "Documentation Updates" table (or the most relevant section listing artifacts/configuration patterns) and add a row or one-liner referencing `artifacts.user_files`. Example one-liner to slot under "Coding Standards" or near the artifacts discussion:

```markdown
- User-defined templated output files: declare via `artifacts.user_files` (see `docs/kubernetes/user-files.md`).
```

Make sure all three files end up with the exact same change (only file-specific headers/frontmatter differ).

- [ ] **Step 9.2: Diff to confirm sync**

```bash
diff CLAUDE.md .github/copilot-instructions.md
diff CLAUDE.md .cursor/rules/python.mdc
```

Expected: only file-specific header/frontmatter differences. If anything else differs, normalize.

- [ ] **Step 9.3: Commit**

```bash
git add CLAUDE.md .github/copilot-instructions.md .cursor/rules/python.mdc
git commit -s -m "$(cat <<'EOF'
docs(claude): note artifacts.user_files in three-file sync

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Final verification

- [ ] **Step 10.1: Format and lint**

```bash
ruff format . && ruff check --fix .
```
Expected: clean exit. Stage and amend if anything reformatted.

- [ ] **Step 10.2: Pre-commit on touched files**

```bash
pre-commit run --files \
  src/aiperf/config/user_files.py \
  src/aiperf/config/artifacts.py \
  src/aiperf/config/config.py \
  src/aiperf/config/resolvers.py \
  src/aiperf/config/loader/jinja.py \
  src/aiperf/config/loader/core.py \
  tests/unit/config/test_user_files.py \
  tests/unit/config/test_artifacts_user_files.py \
  tests/unit/config/loader/test_jinja_strict_undefined.py \
  tests/component_integration/test_user_files_e2e.py \
  docs/kubernetes/user-files.md \
  CLAUDE.md .github/copilot-instructions.md .cursor/rules/python.mdc
```
Expected: all green. If anything fails, fix in place and amend the relevant prior commit (or a new commit).

- [ ] **Step 10.3: Run unit suite**

```bash
uv run pytest tests/unit/ -n auto
```
Expected: All green.

- [ ] **Step 10.4: Run component-integration suite (just the new test)**

```bash
uv run pytest tests/component_integration/test_user_files_e2e.py -m component_integration -n auto -v
```
Expected: All green.

- [ ] **Step 10.5: Verify ergonomics check baseline is not regressed**

```bash
make check-ergonomics
make check-ruff-baselined
```
Expected: clean — new files must not add baseline entries.

- [ ] **Step 10.6: Final summary commit (if any cleanup)**

If steps 10.1-10.5 introduced any further changes, commit them under a focused message. Otherwise, no commit needed.

---

## Self-Review Checklist

After implementation:

- [ ] Every spec section maps to at least one task. Cross-checked: section 5.1 → Task 5; 5.2 → Task 3; 5.3 → Task 1; 5.4 → Task 4; section 6 → Tasks 4 + 6; section 7 → Task 2; section 8 → Tasks 4, 7; section 9 → Tasks 3, 4, 5, 7; section 10 → Tasks 8, 9.
- [ ] No `TBD` / `TODO` / `implement later` placeholders.
- [ ] Every step that changes code shows the actual code.
- [ ] Type names match across tasks: `UserFile`, `UserFileError`, `RunMeta`, `build_user_file_context`, `materialize_user_files` are consistent throughout.
- [ ] Field name `artifacts.user_files` is consistent (not `extra_files`, not `sidecar_files`).
- [ ] All pytest invocations use `-n auto` (per repo convention).
- [ ] Commits are signed (`-s`) and follow the repo's `feat:` / `docs:` / `test:` prefix style.
