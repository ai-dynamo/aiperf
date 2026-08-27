// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transport-neutral dispatch seam.
//!
//! The seam itself is boundary-owned and lives in [`aiperf_core::dispatch`];
//! this module is the compatibility path for runtime code and downstream crates
//! that already import `crate::dispatch::sink::*`.

pub use aiperf_core::dispatch::{
    Dispatchable, ObservedEndpointMetrics, ObservedRoundTripMetrics, ObservedSpecDecodeAcceptance,
    ObservedTokenKind, ObservedTransportRoute, ObservedUsage, RequestObserver, RequestSink,
    TransportFallbackReason, TransportRoute,
};
