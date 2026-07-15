// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Typed `transport` section of the native `BenchmarkConfig`.
//!
//! Wire shape ported from `src/aiperf/orchestrator/rust_wire.py::_inline_transport`
//! (a discriminated union keyed by `type`, with only explicitly-set fields
//! present). The default HTTP transport projects to exactly `{"type":"http"}`.

use serde::{Deserialize, Serialize};

/// The typed inline transport selection (discriminated by `type`).
///
/// Only `http` is fully modeled today; the gRPC and DynoSim transports carry
/// additional inline knobs and are added when those paths are exercised. Serde's
/// internal tagging emits the `type` discriminator.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Transport {
    /// Native HTTP/1.1 or HTTP/2 transport.
    Http,
    /// Native gRPC transport (KServe OIP / Riva).
    Grpc,
    /// Offline virtual-clock Dynamo replay.
    DynosimOffline,
    /// Online wall-clock Dynamo replay.
    DynosimOnline,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn http_projects_type_only() {
        assert_eq!(
            serde_json::to_value(Transport::Http).unwrap(),
            serde_json::json!({"type": "http"})
        );
    }

    #[test]
    fn dynosim_offline_wire_spelling() {
        assert_eq!(
            serde_json::to_value(Transport::DynosimOffline).unwrap(),
            serde_json::json!({"type": "dynosim_offline"})
        );
    }
}
