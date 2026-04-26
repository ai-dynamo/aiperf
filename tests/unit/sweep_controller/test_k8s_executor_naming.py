# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.sweep_controller.k8s_executor import (
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
        == "my-sweep-v0007"
    )


def test_derive_child_name_with_trial_suffix():
    assert (
        derive_child_name("my-sweep", var_idx=7, trial=4, with_trial_suffix=True)
        == "my-sweep-v0007-t04"
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
