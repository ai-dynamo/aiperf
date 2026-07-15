// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! The typed native `BenchmarkConfig` — the runner-consumed `cfg` tree.
//!
//! Each section (`endpoint`, `datasets`, `phases`, `transport`, …) is added as a
//! fully-typed struct as it is ported from `src/aiperf/config/*.py` (input keys)
//! and `src/aiperf/orchestrator/rust_wire.py` (wire shape). Serializing this
//! struct yields the exact `run.cfg` subtree the runner consumes.
//!
//! `deny_unknown_fields` is intentionally omitted: deserializing a Python golden
//! through this type drops the sections not yet ported, which is exactly the
//! parity filter (see `crate::model`). Fields present here are fully typed.

use serde::{Deserialize, Serialize};

/// The canonical benchmark configuration (runner-consumed projection).
///
/// Empty today; grows one typed section per port task. `#[serde(default)]` on
/// every field so a partially-populated config (and a golden filtered through
/// this type) both round-trip.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    // Sections are added here as they are ported, e.g.:
    //   pub endpoint: endpoint::Endpoint,
    //   pub datasets: Vec<dataset::Dataset>,
    //   pub phases: Vec<phase::Phase>,
    // Each is a fully-typed struct whose Serialize output matches the golden.
}
