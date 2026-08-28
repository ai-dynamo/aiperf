// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Priority and effective_priority tests (Task 13).

use aiperf_plugin_host::{
    discovery::DiscoverySource,
    priority::{effective_priority, source_kind_ordinal},
};

#[test]
fn effective_priority_distribution_lowest() {
    let dist = effective_priority(source_kind_ordinal(&DiscoverySource::Distribution), 0);
    let bundle = effective_priority(
        source_kind_ordinal(&DiscoverySource::HermeticBundle("/b".into())),
        0,
    );
    assert!(
        dist < bundle,
        "distribution must have lower effective priority than hermetic bundle"
    );
}

#[test]
fn authored_priority_breaks_ties_within_tier() {
    let ordinal = source_kind_ordinal(&DiscoverySource::ExplicitDirectory("/".into()));
    let low = effective_priority(ordinal, -10);
    let high = effective_priority(ordinal, 100);
    assert!(low < high);
}

#[test]
fn source_tier_dominates_authored_priority() {
    // A distribution plugin with authored priority=1000 must lose to
    // a hermetic bundle with authored priority=-1000.
    let dist = effective_priority(source_kind_ordinal(&DiscoverySource::Distribution), 1000);
    let bundle = effective_priority(
        source_kind_ordinal(&DiscoverySource::HermeticBundle("/b".into())),
        -1000,
    );
    assert!(
        dist < bundle,
        "source tier must dominate: dist={dist} bundle={bundle}"
    );
}
