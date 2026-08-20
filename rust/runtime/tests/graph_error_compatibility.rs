// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public graph error compatibility regressions.

use aiperf_runtime::graph::conditional::ConditionalError;
use aiperf_runtime::graph::scheduler::MixedAnchorFanInError;

#[test]
fn public_tuple_errors_support_external_construction_and_field_access() {
    let conditional = ConditionalError("conditional".to_owned());
    assert_eq!(conditional.0, "conditional");

    let fan_in = MixedAnchorFanInError("fan-in".to_owned());
    assert_eq!(fan_in.0, "fan-in");
}
