# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI parsing regressions for the ``aiperf kube`` mapping and JSON flags.

``--annotations``, ``--labels``, ``--env-vars`` and ``--env-from-secrets`` are
``dict[str, str]`` fields that cyclopts binds by dot-notation. A bare token used
to reach ``Argument._json`` with an empty ``keys`` tuple and crash with
``IndexError: tuple index out of range`` from inside cyclopts. Both spellings
must now work, malformed input must produce a usage error rather than a
traceback, and ``--secret-mounts`` must accept the same shapes ``--tolerations``
does.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml
from cyclopts import App
from cyclopts.exceptions import CoercionError
from pytest import param

from aiperf.cli_commands.kube._mapping_flags import normalize_mapping_flag_tokens
from aiperf.config.kube import KubeOptions

_REPO_ROOT = Path(__file__).resolve().parents[4]

_MIN_CONFIG = """\
models: [m]
endpoint: {urls: [http://x], type: chat, streaming: true}
datasets: [{name: main, type: synthetic, prompts: {isl: 64, osl: 32}}]
phases:
  - {name: profiling, type: concurrency, requests: 10, concurrency: 1}
"""

# Every ``KubeOptions`` field cyclopts binds via dot-notation, paired with the
# ``PodTemplateConfig`` key each one lands on in a rendered AIPerfJob.
_MAPPING_FLAGS = (
    param("--annotations", "annotations", "annotations", id="annotations"),
    param("--labels", "labels", "labels", id="labels"),
    param("--env-vars", "env_vars", "env", id="env-vars"),
    param("--env-from-secrets", "env_from_secrets", "env", id="env-from-secrets"),
)


def _parse_kube_options(*tokens: str) -> KubeOptions:
    """Bind CLI tokens to ``KubeOptions`` exactly as a kube subcommand does.

    ``parse_args`` runs the full argument-binding path -- the one the cyclopts
    ``IndexError`` lived in -- without executing a command body.
    """
    app = App(name="probe", config=(normalize_mapping_flag_tokens,))

    @app.default
    def _probe(*, kube_options: KubeOptions | None = None) -> None: ...

    _, bound, _ = app.parse_args(tokens, exit_on_error=False, print_error=False)
    return bound.arguments["kube_options"]


def _run_generate(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Render an AIPerfJob through the real CLI in a subprocess."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(_MIN_CONFIG)
    return subprocess.run(
        [
            "uv",
            "run",
            "aiperf",
            "kube",
            "generate",
            "-f",
            str(config_file),
            "--image",
            "aiperf:test",
            "--operator",
            *args,
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        check=False,
        text=True,
        timeout=120,
    )


@pytest.mark.parametrize("flag,field,_pod_key", _MAPPING_FLAGS)
def test_mapping_flag_key_value_spelling_is_accepted(
    flag: str, field: str, _pod_key: str
) -> None:
    """The intuitive ``--flag KEY=VALUE`` spelling binds instead of crashing."""
    options = _parse_kube_options(flag, "ALPHA=one/two")

    assert getattr(options, field) == {"ALPHA": "one/two"}


@pytest.mark.parametrize("flag,field,_pod_key", _MAPPING_FLAGS)
def test_mapping_flag_dot_notation_spelling_still_works(
    flag: str, field: str, _pod_key: str
) -> None:
    """Dot-notation is the documented spelling and must keep binding."""
    options = _parse_kube_options(f"{flag}.ALPHA", "one/two")

    assert getattr(options, field) == {"ALPHA": "one/two"}


@pytest.mark.parametrize("flag,field,_pod_key", _MAPPING_FLAGS)
def test_mapping_flag_repeated_key_value_pairs_merge(
    flag: str, field: str, _pod_key: str
) -> None:
    """Repeating the flag accumulates entries rather than raising a repeat error."""
    options = _parse_kube_options(flag, "ALPHA=a/a", flag, "BETA=b/b")

    assert getattr(options, field) == {"ALPHA": "a/a", "BETA": "b/b"}


@pytest.mark.parametrize("flag,field,_pod_key", _MAPPING_FLAGS)
def test_mapping_flag_json_object_is_accepted(
    flag: str, field: str, _pod_key: str
) -> None:
    """A JSON object token binds, matching --node-selector and --tolerations."""
    options = _parse_kube_options(flag, '{"ALPHA": "a/a", "BETA": "b/b"}')

    assert getattr(options, field) == {"ALPHA": "a/a", "BETA": "b/b"}


@pytest.mark.parametrize("flag,_field,_pod_key", _MAPPING_FLAGS)
@pytest.mark.parametrize(
    "value",
    [
        param("ALPHA", id="bare-key"),
        param("=oops", id="empty-key"),
        param("{not json", id="truncated-json"),
        param('["a"]', id="json-array"),
    ],
)  # fmt: skip
def test_mapping_flag_malformed_token_names_the_dot_notation_spelling(
    flag: str, _field: str, _pod_key: str, value: str
) -> None:
    """Unparseable tokens raise a usage error naming every accepted spelling."""
    with pytest.raises(CoercionError) as excinfo:
        _parse_kube_options(flag, value)

    message = str(excinfo.value)
    assert f"{flag}.KEY VALUE" in message
    assert f"{flag} KEY=VALUE" in message


def test_mapping_flag_json_rejects_non_string_values() -> None:
    """``dict[str, str]`` targets cannot hold nested JSON containers."""
    with pytest.raises(CoercionError, match="string keys"):
        _parse_kube_options("--labels", '{"tier": {"nested": "no"}}')


def test_mapping_flag_value_may_contain_equals_signs() -> None:
    """Only the first ``=`` separates key from value."""
    options = _parse_kube_options("--env-vars", "FLAGS=--a=1 --b=2")

    assert options.env_vars == {"FLAGS": "--a=1 --b=2"}


@pytest.mark.parametrize(
    "tokens",
    [
        param(['{"name": "s", "mount_path": "/mnt"}'], id="single-json-object"),
        param(['[{"name": "s", "mount_path": "/mnt"}]'], id="json-array-one-token"),
    ],
)  # fmt: skip
def test_secret_mounts_accepts_the_same_shapes_as_tolerations(
    tokens: list[str],
) -> None:
    """``--secret-mounts`` no longer rejects the array shape ``--tolerations`` takes."""
    options = _parse_kube_options("--secret-mounts", *tokens)

    assert [mount.name for mount in options.secret_mounts] == ["s"]
    assert [mount.mount_path for mount in options.secret_mounts] == ["/mnt"]


@pytest.mark.parametrize(
    "flag,tokens",
    [
        param(
            "--secret-mounts",
            ['{"name": "a", "mount_path": "/a"}', '{"name": "b", "mount_path": "/b"}'],
            id="secret-mounts",
        ),
        param(
            "--tolerations",
            ['{"key": "a", "operator": "Exists"}', '{"key": "b", "operator": "Exists"}'],
            id="tolerations",
        ),
    ],
)  # fmt: skip
def test_json_list_flags_accumulate_across_repeats(
    flag: str, tokens: list[str]
) -> None:
    """Both JSON list flags accept repeated occurrences identically."""
    options = _parse_kube_options(flag, tokens[0], flag, tokens[1])

    field = "secret_mounts" if flag == "--secret-mounts" else "tolerations"
    assert len(getattr(options, field)) == 2


@pytest.mark.parametrize(
    "flag",
    [
        param("--secret-mounts", id="secret-mounts"),
        param("--tolerations", id="tolerations"),
    ],
)  # fmt: skip
def test_json_list_flags_explain_the_required_shape(flag: str) -> None:
    """A malformed token names the flag and shows a well-formed example."""
    with pytest.raises(Exception) as excinfo:  # noqa: PT011 - pydantic wraps the cause
        _parse_kube_options(flag, "notjson")

    message = str(excinfo.value)
    assert flag in message
    assert "JSON object or array" in message


@pytest.mark.parametrize("flag,_field,pod_key", _MAPPING_FLAGS)
def test_cli_subprocess_accepts_both_mapping_spellings(
    tmp_path: Path, flag: str, _field: str, pod_key: str
) -> None:
    """End-to-end proof: neither spelling crashes the real CLI process.

    Argument binding is where the cyclopts ``IndexError`` lived, so this runs
    the installed entry point rather than calling into it in-process.
    """
    for args in ((flag, "ALPHA=one/two"), (f"{flag}.ALPHA", "one/two")):
        result = _run_generate(tmp_path, *args)

        assert result.returncode == 0, result.stderr
        assert "Traceback" not in result.stderr
        assert "IndexError" not in result.stderr
        pod_template: dict[str, Any] = yaml.safe_load(result.stdout)["spec"][
            "podTemplate"
        ]
        assert pod_template[pod_key], f"{flag} {args} produced no {pod_key}"


def test_cli_subprocess_malformed_mapping_token_is_a_usage_error(
    tmp_path: Path,
) -> None:
    """The former ``IndexError`` crash is now a non-zero usage error."""
    result = _run_generate(tmp_path, "--env-from-secrets", "OPENAI_API_KEY")

    assert result.returncode != 0
    assert "Traceback" not in result.stderr
    assert "IndexError" not in result.stderr
    assert "--env-from-secrets.KEY VALUE" in result.stderr.replace("\n", " ")
