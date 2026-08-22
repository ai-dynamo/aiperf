# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib

import aiperf_k8s_operator.upload_auth as upload_auth

derive_upload_public_key = upload_auth.derive_upload_public_key
verify_upload_signature = upload_auth.verify_upload_signature

BOOTSTRAP = b"private sidecar bootstrap"
ARTIFACT_DIGEST = hashlib.sha256(b"{}").hexdigest()
OBJECT_UID = "9d2f3e2a-1111-4222-8333-abcdefabcdef"
PUBLIC_KEY = "8uFXJpCIj094psVHjvxpu5_YA6Ruivm9sb8z4GNRlTo"
SIGNATURE = (
    "FywHe55FNtgNfyb8KjEVXfxqsGYnRM2RiZk9mSRIRW9kP78zGr9YckvfwTzwu7JbedO"
    "37u6GPTbNmEu-DagECg"
)


def test_upload_key_and_signature_match_the_native_wire_fixture() -> None:
    assert (
        derive_upload_public_key(
            BOOTSTRAP, "bench", "job-1", "run-1", OBJECT_UID
        )
        == PUBLIC_KEY
    )
    assert verify_upload_signature(
        PUBLIC_KEY,
        SIGNATURE,
        "bench",
        "job-1",
        "run-1",
        OBJECT_UID,
        "artifact",
        "summary.json",
        ARTIFACT_DIGEST,
        2,
    )


def test_upload_signature_cannot_authorize_a_different_identity_or_content() -> None:
    fields = [
        "bench",
        "job-1",
        "run-1",
        OBJECT_UID,
        "artifact",
        "summary.json",
        ARTIFACT_DIGEST,
        2,
    ]
    for index, wrong in [
        (0, "other-namespace"),
        (2, "other-run"),
        (3, "9d2f3e2a-1111-4222-8333-000000000000"),
        (5, "other.json"),
        (6, hashlib.sha256(b"tampered").hexdigest()),
        (7, 3),
    ]:
        candidate = fields.copy()
        candidate[index] = wrong
        assert not verify_upload_signature(PUBLIC_KEY, SIGNATURE, *candidate)


def test_upload_signature_rejects_missing_or_malformed_authority() -> None:
    fields = (
        "bench",
        "job-1",
        "run-1",
        OBJECT_UID,
        "artifact",
        "summary.json",
        ARTIFACT_DIGEST,
        2,
    )
    assert not verify_upload_signature(PUBLIC_KEY, "", *fields)
    assert not verify_upload_signature("not-a-public-key", SIGNATURE, *fields)


def test_distinct_read_capability_stores_only_a_one_way_digest() -> None:
    encode = getattr(upload_auth, "encode_results_read_token", None)
    digest = getattr(upload_auth, "results_read_token_sha256", None)
    verify = getattr(upload_auth, "verify_results_read_token", None)
    assert callable(encode) and callable(digest) and callable(verify)

    token = encode(bytes(range(32)))
    assert token == "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8"
    assert (
        digest(bytes(range(32)))
        == "630dcd2966c4336691125448bbb25b4ff412a49c732db2c8abc1b8581bd710dd"
    )
    assert verify(
        "630dcd2966c4336691125448bbb25b4ff412a49c732db2c8abc1b8581bd710dd",
        token,
    )
    assert not verify(
        "630dcd2966c4336691125448bbb25b4ff412a49c732db2c8abc1b8581bd710dd",
        token + "A",
    )
