# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Config errors must not print a presigned signature, even a truncated tail of one."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError
from pytest import param
from rich.console import Console

import aiperf.cli_utils as cli_utils
from aiperf.cli_utils import exit_on_error
from aiperf.config.loader.core import load_config
from aiperf.config.loader.errors import ConfigurationError, describe_validation_error

SIGNATURE = "SECRETSIGNATURE1234567890abcdef"
PRESIGNED = (
    "https://bucket.s3.amazonaws.com/t.jsonl"
    f"?X-Amz-Credential=AKIAXX&X-Amz-Signature={SIGNATURE}"
)

TWO_DATASETS_YAML = f"""\
schemaVersion: "2.0"
benchmark:
  model: m
  endpoint:
    url: http://localhost:8000
  datasets:
    - name: a
      type: file
      format: mooncake_trace
      path: /tmp/x.jsonl
    - name: b
      type: file
      format: mooncake_trace
      path: {PRESIGNED}
  phases:
    type: concurrency
    concurrency: 1
    requests: 10
"""


class _Paths(BaseModel):
    paths: list[int]


def _validation_error(value: object) -> ValidationError:
    with pytest.raises(ValidationError) as exc_info:
        _Paths.model_validate({"paths": value})
    return exc_info.value


@pytest.mark.parametrize(
    "message,context",
    [
        param(f"bad path {PRESIGNED} here", None, id="message"),
        param("bad path", f"from {PRESIGNED}", id="context"),
    ],
)  # fmt: skip
def test_configuration_error_redacts_presigned_url(
    message: str, context: str | None
) -> None:
    error = ConfigurationError(message, context=context)

    assert SIGNATURE not in str(error)
    assert "X-Amz-Signature=<redacted>" in str(error)


def test_describe_validation_error_hides_input_carrying_credential() -> None:
    error = _validation_error(["a" * 40, PRESIGNED])
    assert SIGNATURE[-10:] in str(error)

    described = describe_validation_error(error)

    assert SIGNATURE[-10:] not in described
    assert "2 validation errors for _Paths" in described
    assert "paths.0" in described
    assert "[type=int_parsing]" in described


def test_describe_validation_error_keeps_plain_input_unchanged() -> None:
    error = _validation_error(["not-a-number"])

    assert describe_validation_error(error) == str(error)


def test_load_config_two_datasets_does_not_leak_signature_tail(
    tmp_path: Path,
) -> None:
    config_file = tmp_path / "two.yaml"
    config_file.write_text(TWO_DATASETS_YAML)

    with pytest.raises(ConfigurationError) as exc_info:
        load_config(config_file)

    assert SIGNATURE[-10:] in str(exc_info.value.__cause__)
    assert SIGNATURE[-10:] not in str(exc_info.value)
    assert "[type=too_long]" in str(exc_info.value)


def test_exit_on_error_does_not_print_signature_tail(monkeypatch) -> None:
    buffer = io.StringIO()
    monkeypatch.setattr(
        cli_utils, "console", Console(file=buffer, force_terminal=False, width=200)
    )

    with pytest.raises(SystemExit), exit_on_error(show_traceback=False):
        raise _validation_error(["a" * 40, PRESIGNED])

    assert SIGNATURE[-10:] not in buffer.getvalue()
    assert "paths.1" in buffer.getvalue()


def test_configuration_error_keeps_already_redacted_message_intact() -> None:
    message = "Cannot use https://h/t.jsonl?X-Amz-Signature=<redacted>: denied"

    assert str(ConfigurationError(message)) == message
