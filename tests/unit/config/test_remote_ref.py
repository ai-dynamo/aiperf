# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import orjson
import pytest
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator
from pytest import param

from aiperf.common.redact import REDACTED_VALUE, redact_url
from aiperf.config.dataset.remote_ref import (
    RemoteRef,
    coerce_remote_ref,
    is_presigned,
    is_remote_ref,
    parse_remote_ref,
)


@pytest.mark.parametrize(
    "value,scheme,location,path,revision,is_container",
    [
        param("hf://org/repo", "hf", "org/repo", None, None, True, id="hf-short-root"),
        param("hf://datasets/org/repo", "hf", "org/repo", None, None, True, id="hf-long-root"),
        param("hf://org/repo/", "hf", "org/repo", None, None, True, id="hf-trailing-slash"),
        param("hf://org/repo@v1/data/t.jsonl", "hf", "org/repo", "data/t.jsonl", "v1", False, id="hf-rev-path"),
        param("hf://org/repo@refs%2Fpr%2F1", "hf", "org/repo", None, "refs/pr/1", True, id="hf-encoded-revision"),
        param("s3://bucket/traces/t.jsonl", "s3", "bucket", "traces/t.jsonl", None, False, id="s3-file"),
        param("s3://bucket/traces/", "s3", "bucket", "traces/", None, True, id="s3-prefix"),
        param("s3://bucket", "s3", "bucket", None, None, True, id="s3-bucket-root"),
        param("s3://bucket/k.jsonl?versionId=abc", "s3", "bucket", "k.jsonl", "abc", False, id="s3-version"),
        param("s3://bucket/my traces/día 1.jsonl", "s3", "bucket", "my traces/día 1.jsonl", None, False, id="s3-space-unicode"),
        param("https://h.example/x/t.jsonl", "https", "https://h.example/x/t.jsonl", None, None, False, id="https"),
        param("http://h.example/t.jsonl", "http", "http://h.example/t.jsonl", None, None, False, id="http"),
    ],
)  # fmt: skip
def test_parse_table(value, scheme, location, path, revision, is_container) -> None:
    parsed = parse_remote_ref(value)
    assert (parsed.scheme, parsed.location, parsed.path, parsed.revision, parsed.is_container) == (
        scheme, location, path, revision, is_container,
    )  # fmt: skip


@pytest.mark.parametrize(
    "value,match",
    [
        param("hf://org", "must name <org>/<repo>", id="hf-no-repo"),
        param("hf://models/org/repo", "only Hugging Face dataset repos", id="hf-model-repo"),
        param("hf://org/repo@", "empty repo name or revision", id="hf-empty-rev"),
        param("s3://", "must name a bucket", id="s3-no-bucket"),
        param("s3://b/k?foo=1", "only versionId is allowed", id="s3-bad-query"),
        param("gs://b/k", r"Supported schemes: hf://, s3://, https://, http://", id="unknown-scheme"),
        param("https://", "has no host", id="https-no-host"),
    ],
)  # fmt: skip
def test_parse_errors(value: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        parse_remote_ref(value)


@pytest.mark.parametrize(
    "value",
    [
        param("gs://b/k?X-Amz-Signature=secret", id="unknown-scheme"),
        param("hf://org?token=secret", id="hf-no-repo"),
        param("hf://models/org/repo?token=secret", id="hf-model-repo"),
        param("hf://org/@/?token=secret", id="hf-empty-rev"),
        param("s3:///k?token=secret", id="s3-no-bucket"),
        param("s3://b/k?foo=1&X-Amz-Signature=secret", id="s3-bad-query"),
        param("https://?token=secret", id="https-no-host"),
    ],
)  # fmt: skip
def test_parse_errors_never_contain_the_secret(value: str) -> None:
    with pytest.raises(ValueError) as exc_info:
        parse_remote_ref(value)
    assert "secret" not in str(exc_info.value)


@pytest.mark.parametrize(
    "value,expected",
    [
        param("hf://a/b", True, id="hf"),
        param("C:\\data\\t.jsonl", False, id="windows-backslash"),
        param("C:/data/t.jsonl", False, id="windows-slash"),
        param("./t.jsonl", False, id="relative"),
        param(None, False, id="none"),
    ],
)  # fmt: skip
def test_is_remote_ref(value: object, expected: bool) -> None:
    assert is_remote_ref(value) is expected


def test_both_hf_forms_share_canonical() -> None:
    assert (
        RemoteRef(ref="hf://org/repo@v1/x.jsonl").canonical
        == RemoteRef(ref="hf://datasets/org/repo@v1/x.jsonl").canonical
        == "hf://datasets/org/repo@v1/x.jsonl"
    )


def test_presigned_url_is_redacted_and_canonical_drops_query() -> None:
    url = "https://b.s3.amazonaws.com/t.jsonl?X-Amz-Date=1&X-Amz-Signature=secret&X-Amz-Credential=AKIA"
    ref = RemoteRef(ref=url)
    assert is_presigned(url)
    assert "secret" not in ref.display and "AKIA" not in ref.display
    assert REDACTED_VALUE in ref.display
    assert ref.canonical == "https://b.s3.amazonaws.com/t.jsonl"
    assert redact_url("https://h/t.jsonl?page=2") == "https://h/t.jsonl?page=2"


def test_is_presigned_is_false_for_plain_query() -> None:
    assert not is_presigned("https://h/t.jsonl?page=2")


def test_repr_and_str_show_only_the_redacted_display() -> None:
    ref = RemoteRef(ref="https://h/t.jsonl?X-Amz-Signature=secret")
    assert "secret" not in repr(ref)
    assert "secret" not in str(ref)
    assert ref.display in repr(ref)
    assert str(ref) == ref.display


def test_validation_error_for_a_bad_presigned_ref_has_no_secret() -> None:
    # The holder must hide inputs too: pydantic echoes the failing field value
    # using the outermost model's config, which RemoteRef cannot control.
    class Holder(BaseModel):
        model_config = ConfigDict(hide_input_in_errors=True)

        ref: RemoteRef

        @field_validator("ref", mode="before")
        @classmethod
        def _coerce(cls, value: object) -> object:
            return coerce_remote_ref(value)

    bad = "gs://b/k?X-Amz-Signature=secret"
    with pytest.raises(ValidationError) as direct:
        RemoteRef(ref=bad)
    with pytest.raises(ValidationError) as from_str:
        Holder(ref=bad)
    with pytest.raises(ValidationError) as from_dict:
        Holder(ref={"ref": bad})
    for exc_info in (direct, from_str, from_dict):
        assert "secret" not in str(exc_info.value)
        assert "secret" not in repr(exc_info.value.errors(include_input=False))


def test_serializer_unresolved_is_plain_string_resolved_is_dict() -> None:
    class Holder(BaseModel):
        ref: RemoteRef

    unresolved = Holder(ref=RemoteRef(ref="hf://org/repo"))
    assert orjson.loads(unresolved.model_dump_json())["ref"] == "hf://org/repo"
    resolved = Holder(ref=RemoteRef(ref="hf://org/repo").with_version("abc"))
    assert orjson.loads(resolved.model_dump_json())["ref"] == {
        "ref": "hf://org/repo",
        "resolved_version": "abc",
    }


def test_serializer_never_emits_presigned_secret() -> None:
    ref = RemoteRef(ref="https://h/t.jsonl?X-Amz-Signature=secret").with_version(
        "etag:1"
    )
    assert "secret" not in orjson.dumps(ref.model_dump(mode="json")).decode()


def test_coerce_remote_ref_round_trips_and_passes_through_local() -> None:
    ref = coerce_remote_ref("s3://b/k")
    assert isinstance(ref, RemoteRef)
    assert coerce_remote_ref(ref) is ref
    assert (
        coerce_remote_ref(
            {"ref": "s3://b/k", "resolved_version": "v:1"}
        ).resolved_version
        == "v:1"
    )
    assert coerce_remote_ref("local.jsonl") == "local.jsonl"


def test_remote_ref_is_frozen_and_validates() -> None:
    ref = RemoteRef(ref="hf://org/repo")
    with pytest.raises(ValidationError):
        RemoteRef(ref="gs://b/k")
    with pytest.raises(ValidationError):
        ref.ref = "hf://x/y"  # type: ignore[misc]
