// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Concrete authored-phase identity for named multi-phase workflows.
//!
//! Aggregate warmup/profiling inclusion still uses [`crate::timing::Phase`].
//! Components that must distinguish two profiling phases (or two warmups) use
//! this identity instead of the binary phase enum alone.

use crate::engine::protocol::{PhaseRoleSpec, PhaseSpec};
use crate::timing::PhaseKind;

/// Concrete identity of one authored workflow phase.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhaseIdentity {
    /// Zero-based position in the authored phase sequence.
    pub phase_index: usize,
    /// Stable user-authored workflow phase name.
    pub phase_name: String,
    /// Semantic warmup or profiling role.
    pub phase_kind: PhaseKind,
    /// Dense zero-based position among profiling phases only.
    pub profiling_index: Option<usize>,
}

/// Build a concrete phase identity from one resolved phase specification.
pub(crate) fn phase_identity_from_spec(
    spec: &PhaseSpec,
    phase_index: usize,
    profiling_index: Option<usize>,
) -> PhaseIdentity {
    let common = spec.common();
    PhaseIdentity {
        phase_index,
        phase_name: common.name.clone(),
        phase_kind: match common.semantic_role() {
            PhaseRoleSpec::Warmup => PhaseKind::Warmup,
            PhaseRoleSpec::Profiling => PhaseKind::Profiling,
        },
        profiling_index,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::protocol::PhaseSpec;
    use crate::timing::PhaseKind;

    fn named_concurrency_phase(name: &str) -> PhaseSpec {
        serde_json::from_value(serde_json::json!({
            "type": "concurrency",
            "name": name,
            "exclude_from_results": false,
            "concurrency": 1,
        }))
        .unwrap()
    }

    #[test]
    fn phase_identity_from_named_profiling_phase() {
        let spec = named_concurrency_phase("storm");
        let id = phase_identity_from_spec(&spec, 2, Some(1));
        assert_eq!(id.phase_name, "storm");
        assert_eq!(id.phase_index, 2);
        assert_eq!(id.profiling_index, Some(1));
        assert!(matches!(id.phase_kind, PhaseKind::Profiling));
    }
}
