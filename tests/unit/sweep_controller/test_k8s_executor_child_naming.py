# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.sweep_controller.k8s_executor import build_child_name


def test_child_name_embeds_sweep_epoch() -> None:
    # 9-11 digit decimal epoch
    assert (
        build_child_name(
            sweep_name="satsweep",
            sweep_run_epoch="1714069323",
            variation_index=7,
            trial_index=4,
        )
        == "satsweep-e1714069323-v0007-t04"
    )


def test_child_name_no_trial_omits_trial_segment() -> None:
    assert (
        build_child_name(
            sweep_name="satsweep",
            sweep_run_epoch="1714069323",
            variation_index=0,
            trial_index=None,
        )
        == "satsweep-e1714069323-v0000"
    )
