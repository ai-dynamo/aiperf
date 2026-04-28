# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re

import pytest
from pytest import param

from aiperf.sweep_controller.k8s_executor import (
    _sanitize_for_label,
    derive_child_name,
    is_my_child,
    needs_trial_suffix,
)


def test_needs_trial_suffix_logic():
    assert needs_trial_suffix(multi_run_trials=5, has_convergence=False) is True
    assert needs_trial_suffix(multi_run_trials=1, has_convergence=False) is False
    assert needs_trial_suffix(multi_run_trials=None, has_convergence=True) is True
    assert needs_trial_suffix(multi_run_trials=None, has_convergence=False) is False


def test_derive_child_name_no_trial_suffix():
    assert (
        derive_child_name("my-sweep", var_idx=7, trial=0, with_trial_suffix=False)
        == "my-sweep-v07"
    )


def test_derive_child_name_with_trial_suffix():
    assert (
        derive_child_name("my-sweep", var_idx=7, trial=4, with_trial_suffix=True)
        == "my-sweep-v07-t4"
    )


def test_is_my_child_owner_ref_match():
    child = {
        "metadata": {
            "ownerReferences": [{"uid": "abc-123", "kind": "AIPerfSweep"}],
            "labels": {"aiperf.nvidia.com/sweep": "my-sweep"},
        }
    }
    assert is_my_child(child, sweep_uid="abc-123", sweep_name="my-sweep") is True


def test_is_my_child_rejects_label_mismatch():
    child = {
        "metadata": {
            "ownerReferences": [{"uid": "abc-123"}],
            "labels": {"aiperf.nvidia.com/sweep": "different-sweep"},
        }
    }
    assert is_my_child(child, sweep_uid="abc-123", sweep_name="my-sweep") is False


def test_is_my_child_rejects_uid_mismatch():
    child = {
        "metadata": {
            "ownerReferences": [{"uid": "wrong-uid"}],
            "labels": {"aiperf.nvidia.com/sweep": "my-sweep"},
        }
    }
    assert is_my_child(child, sweep_uid="abc-123", sweep_name="my-sweep") is False


# =============================================================================
# DNS-1123 label-value hardening regression-locks (third-pass fix).
# K8s label values: ``(([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9])?`` and at most
# 63 chars. We additionally lowercase. The function MUST re-strip leading/
# trailing non-alnum AFTER the 63-char truncation, otherwise a value like
# ``concurrency=`` × 30 ends in ``=`` after sub but the truncated head ends
# in a ``-`` that fails validation.
# =============================================================================

_LABEL_VALUE_STRICT = r"[a-z0-9]([a-z0-9._-]{0,61}[a-z0-9])?"


@pytest.mark.parametrize(
    "value, expected",
    [
        param("simple", "simple", id="passthrough"),
        param("UPPER", "upper", id="lowercase"),
        param("a.b.c", "a.b.c", id="dots-preserved"),
        param("a_b_c", "a_b_c", id="underscores-preserved"),
        param("c=64", "c-64", id="equals-replaced"),
        param("", "v", id="empty-falls-back"),
        param("___", "v", id="all-underscores-fall-back"),
        param("---", "v", id="all-hyphens-fall-back"),
        param("...", "v", id="all-dots-fall-back"),
    ],
)  # fmt: skip
def test_sanitize_for_label_basic(value: str, expected: str) -> None:
    assert _sanitize_for_label(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        param("a" * 70, id="very-long"),
        param("ab-" * 25, id="hyphen-run-truncates-on-hyphen"),
        param("ab_" * 25, id="underscore-run-truncates-on-underscore"),
        param("ab." * 25, id="dot-run-truncates-on-dot"),
        param("c=" * 40, id="equals-collapse-then-truncate"),
        param("phases.profiling.concurrency=" + "x" * 100, id="long-with-prefix"),
    ],
)  # fmt: skip
def test_sanitize_for_label_strict_after_truncation(value: str) -> None:
    """After truncation to 63 chars, the result still starts AND ends alnum."""
    out = _sanitize_for_label(value)
    assert len(out) <= 63
    assert re.fullmatch(_LABEL_VALUE_STRICT, out), (
        f"sanitized {out!r} from {value!r} is not a valid k8s label value "
        f"(must start and end with [a-z0-9] and be <=63 chars)"
    )


@pytest.mark.parametrize(
    "value",
    [
        param("c=64", id="metric-equals"),
        param("phases.profiling.concurrency=8", id="dotted-path-equals-int"),
        param("model=Qwen/Qwen3-0.6B", id="model-path"),
        param("rate=10.5", id="float-rate"),
        param("seed=42", id="seed"),
    ],
)  # fmt: skip
def test_sanitize_for_label_realistic_variation_labels(value: str) -> None:
    """Round-trip the kinds of labels the orchestrator actually emits."""
    out = _sanitize_for_label(value)
    assert re.fullmatch(_LABEL_VALUE_STRICT, out), out
    assert out  # never empty
