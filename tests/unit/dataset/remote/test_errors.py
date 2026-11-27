# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from aiperf.common.redact import REDACTED_VALUE
from aiperf.config.loader.errors import ConfigurationError
from aiperf.dataset.remote.errors import (
    RemoteErrorKind,
    RemoteSourceError,
    RemoteStage,
    changed_after_pin,
)


def test_message_shape_and_redaction() -> None:
    err = RemoteSourceError(
        ref="https://h/t.jsonl?X-Amz-Signature=secret",
        kind=RemoteErrorKind.AUTH_DENIED,
        stage=RemoteStage.ACCESS,
        what="the server denied access (HTTP 403)",
        hint="Generate a new presigned URL",
    )
    assert isinstance(err, ConfigurationError)
    assert str(err).startswith(
        f"Cannot use https://h/t.jsonl?X-Amz-Signature={REDACTED_VALUE}: the server"
    )
    assert "secret" not in str(err)
    assert err.kind == RemoteErrorKind.AUTH_DENIED and err.stage == RemoteStage.ACCESS
    assert "secret" not in err.ref


def test_library_text_in_what_and_hint_is_redacted() -> None:
    err = RemoteSourceError(
        ref="hf://org/repo",
        kind=RemoteErrorKind.TRANSIENT,
        stage=RemoteStage.MATERIALIZE,
        what="request to https://h/t.jsonl?X-Amz-Signature=leak1&a=b failed",
        hint="see https://user:pw123@h/x?token=leak2",
    )
    text = str(err)
    assert "leak1" not in text and "leak2" not in text and "pw123" not in text
    assert REDACTED_VALUE in text
    assert "a=b" in text


def test_changed_after_pin_factory() -> None:
    err = changed_after_pin(
        "https://h/t.jsonl?X-Amz-Signature=secret", RemoteStage.MATERIALIZE, "the file"
    )
    assert isinstance(err, RemoteSourceError)
    assert err.stage == RemoteStage.MATERIALIZE
    assert "the file changed after AIPerf pinned its version" in str(err)
    assert "Run again" in str(err)
    assert "secret" not in str(err)
