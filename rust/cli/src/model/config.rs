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

use super::endpoint::Endpoint;
use super::models::Models;

/// The canonical benchmark configuration (runner-consumed projection).
///
/// Grows one typed section per port task. Sections not yet ported are simply
/// absent from this struct; a Python golden deserialized through it drops them
/// (no `deny_unknown_fields`), which is the parity filter. Every section field
/// is `Option` so a partial config (and a filtered golden) both round-trip;
/// `skip_serializing_if` keeps an unset section out of the serialized request
/// exactly as Python omits an unprojected section.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Model-selection policy (`cfg.models`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub models: Option<Models>,
    /// Default endpoint profile (`cfg.endpoint`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<Endpoint>,
    // Further sections (datasets, phases, transport, tokenizer, metrics,
    // artifacts, runtime, slos, sidecars, export, …) are added here as ported.
}
