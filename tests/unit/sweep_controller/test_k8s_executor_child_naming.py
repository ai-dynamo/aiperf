# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.sweep_controller.k8s_executor import build_child_name


def test_child_name_omits_sweep_epoch() -> None:
    # Epoch lives on the sweep-run-epoch label, not in the name.
    assert (
        build_child_name(
            sweep_name="satsweep",
            variation_index=7,
            trial_index=4,
        )
        == "satsweep-v07-t4"
    )


def test_child_name_no_trial_omits_trial_segment() -> None:
    assert (
        build_child_name(
            sweep_name="satsweep",
            variation_index=0,
            trial_index=None,
        )
        == "satsweep-v00"
    )
