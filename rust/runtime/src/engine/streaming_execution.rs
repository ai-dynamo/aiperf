// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Registered `shadow_replay` workload for the native streaming plane.
//!
//! `shadow_replay` is the workload identity under which a bounded streaming
//! run is authored. Its factory owns *capability agreement*: every refusal a
//! streaming run can make on the basis of what was selected — rather than what
//! was observed — happens here, before any source is polled, any endpoint is
//! contacted, and any checkpoint participant is initialized.
//!
//! Three refusals are frozen into this generation:
//!
//! - Exactly one profiling phase. A warmup phase, or a second profiling phase,
//!   fails during validation. Generation one has no phase-handoff story for a
//!   streaming source, and discovering that mid-run would strand a partially
//!   consumed immutable snapshot.
//! - A Dynamo source/format composition is refused while its reconstruction
//!   factories are absent. The refusal is here rather than inside the pipeline
//!   so an unsupported selection never reaches execution.
//! - A resume whose committed synthesis authority disagrees with the authored
//!   plan is refused before participant initialization. The frozen execution
//!   identity binds the authored synthesis profile and never changes; the
//!   authority is the separate resumable fact, and a mismatch means the
//!   restored state describes a different content lineage.

use std::{collections::BTreeSet, sync::Arc};

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;

use crate::{
    engine::{
        protocol::PhaseSpec,
        registry::{
            ValidatedWorkloadConfig, WorkloadDescriptor, WorkloadFactory, WorkloadRequirements,
        },
    },
    extensions::AIPerfRegistry,
    streaming::identity::ContentDigest,
};

/// Registry metadata for the executable streaming shadow-replay workload.
pub static SHADOW_REPLAY_WORKLOAD_DESCRIPTOR: WorkloadDescriptor = WorkloadDescriptor {
    id: "shadow_replay",
    description: "Bounded streaming shadow replay over the native session and action plane",
};

/// Source and format factory identifiers whose Dynamo composition is not yet
/// executable, because canonical content reconstruction has no registered
/// factory in this generation.
const DYNAMO_COMPOSITION_IDS: &[&str] = &["dynamo", "streaming_dynamo"];

/// Whether the run's checkpoint and session authority is bound to a synthesis
/// profile.
///
/// The *frozen execution identity* binds the authored synthesis profile and
/// never changes. This authority is the separate, resumable fact: resume
/// compares it before participant initialization and never mutates the frozen
/// plan.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SynthesisAuthority {
    /// No synthesis factory participated; content came from the source verbatim.
    Unbound,
    /// A synthesis factory participated under the named session-program digest.
    Bound {
        /// Digest of the session program that bound the authority.
        session_program_digest: ContentDigest,
    },
}

impl SynthesisAuthority {
    /// Refuse a restored authority that disagrees with the authored plan.
    ///
    /// Called before any participant is initialized, so a mismatched resume
    /// leaves every participant untouched and the frozen plan byte-unchanged.
    pub fn accept_restored(&self, restored: &Self) -> Result<()> {
        if self == restored {
            return Ok(());
        }
        bail!(
            "restored synthesis authority {restored:?} does not match the authored plan's \
             {self:?}; the committed state describes a different content lineage"
        )
    }
}

/// Strict authored configuration for a shadow-replay run.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShadowReplayWorkloadConfigV2 {
    /// Number of local execution workers.
    pub worker_count: usize,
    /// Streaming source factory identifier.
    pub source: String,
    /// Streaming format factory identifier.
    pub format: String,
    /// Streaming session program identifier.
    pub session_program: String,
    /// Ordered phase objects; generation one accepts exactly one profiling phase.
    pub phases: Vec<PhaseSpec>,
    /// Resolved reliability policy digest recorded in execution-plan agreement.
    ///
    /// Present in the plan so a pipeline whose reporter resolves a different
    /// policy is refused before the first source poll rather than running under
    /// a policy nobody agreed to.
    pub reliability_policy_digest: ContentDigest,
    /// Authored synthesis authority for this run.
    #[serde(default = "unbound_authority")]
    pub synthesis_authority: SynthesisAuthority,
}

fn unbound_authority() -> SynthesisAuthority {
    SynthesisAuthority::Unbound
}

// `PhaseSpec` is not `Debug`, and the authored phase objects are not what a
// diagnostic needs anyway: the selection and phase count are.
impl std::fmt::Debug for ShadowReplayWorkloadConfigV2 {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ShadowReplayWorkloadConfigV2")
            .field("worker_count", &self.worker_count)
            .field("source", &self.source)
            .field("format", &self.format)
            .field("session_program", &self.session_program)
            .field("phase_count", &self.phases.len())
            .finish_non_exhaustive()
    }
}

/// Refuse anything but exactly one profiling phase.
///
/// Generation one has no phase-handoff story for an immutable streaming
/// snapshot, so warmup plus live profile — and two profiling phases — are
/// capability failures rather than run-time surprises.
pub fn ensure_single_profiling_phase(phases: &[PhaseSpec]) -> Result<()> {
    let warmup_count = phases
        .iter()
        .filter(|phase| phase.common().is_warmup())
        .count();
    if warmup_count > 0 {
        bail!(
            "shadow_replay accepts no warmup phase, but {warmup_count} were authored; a streaming \
             source cannot be replayed across a phase handoff in this generation"
        );
    }
    match phases.len() {
        1 => Ok(()),
        0 => bail!("shadow_replay requires exactly one profiling phase, but none was authored"),
        other => bail!(
            "shadow_replay accepts exactly one profiling phase, but {other} were authored"
        ),
    }
}

/// Refuse a Dynamo source/format composition while its factories are absent.
pub fn ensure_supported_composition(source: &str, format: &str) -> Result<()> {
    for (role, id) in [("source", source), ("format", format)] {
        if DYNAMO_COMPOSITION_IDS.contains(&id) {
            bail!(
                "shadow_replay cannot execute the Dynamo {role} {id:?}: canonical content \
                 reconstruction has no registered factory in this generation"
            );
        }
    }
    Ok(())
}

/// Refuse a plan whose recorded policy digest is not the resolved one.
///
/// This is the amendment's agreement check: the digest travels in the frozen
/// plan, so a reporter resolved from a different policy is caught before the
/// pipeline polls a source or issues a request.
pub fn ensure_reliability_policy_agreement(
    planned: &ContentDigest,
    resolved: &ContentDigest,
) -> Result<()> {
    if planned == resolved {
        return Ok(());
    }
    bail!(
        "resolved reliability policy digest does not match the digest recorded in the frozen \
         execution plan; no adapter or workload default may replace the agreed policy"
    )
}

/// Registered factory for the executable streaming workload.
#[derive(Debug, Default)]
struct ShadowReplayWorkloadFactoryV2;

impl WorkloadFactory for ShadowReplayWorkloadFactoryV2 {
    fn descriptor(&self) -> &'static WorkloadDescriptor {
        &SHADOW_REPLAY_WORKLOAD_DESCRIPTOR
    }

    fn validate(&self, authored: &RawValue) -> Result<Box<dyn ValidatedWorkloadConfig>> {
        let config: ShadowReplayWorkloadConfigV2 = serde_json::from_str(authored.get())
            .map_err(|error| anyhow::anyhow!("shadow_replay workload config: {error}"))?;
        if config.worker_count == 0 {
            bail!("shadow_replay requires at least one worker");
        }
        ensure_single_profiling_phase(&config.phases)?;
        ensure_supported_composition(&config.source, &config.format)?;
        Ok(Box::new(config))
    }

    fn requirements(&self, _config: &dyn ValidatedWorkloadConfig) -> Result<WorkloadRequirements> {
        // Streaming shadow replay reaches an ordinary inference endpoint, so it
        // asks for no transport feature the scheduled workload does not.
        Ok(WorkloadRequirements::default())
    }
}

/// Register the executable streaming workload.
pub fn register_streaming_workloads(registry: &mut AIPerfRegistry) -> Result<()> {
    registry.register_workload(Arc::new(ShadowReplayWorkloadFactoryV2))
}

/// Identifiers a shadow-replay run selected, for capability resolution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShadowReplaySelection {
    /// Streaming source factory identifier.
    pub source: String,
    /// Streaming format factory identifier.
    pub format: String,
    /// Streaming session program identifier.
    pub session_program: String,
}

impl ShadowReplaySelection {
    /// Refuse any selected identifier the registry cannot resolve.
    ///
    /// Capability agreement resolves by identifier rather than by probing, so
    /// an unregistered factory fails closed against the compiled inventory.
    pub fn ensure_registered(&self, registered: &BTreeSet<String>) -> Result<()> {
        for (role, id) in [
            ("source", &self.source),
            ("format", &self.format),
            ("session program", &self.session_program),
        ] {
            if !registered.contains(id) {
                bail!("shadow_replay selected the unregistered {role} {id:?}");
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::protocol::{PhaseCommonSpec, PhaseRoleSpec};

    fn phase(name: &str, kind: Option<PhaseRoleSpec>) -> PhaseSpec {
        PhaseSpec::Concurrency {
            common: PhaseCommonSpec {
                name: name.to_owned(),
                kind,
                exclude_from_results: false,
                requests: Some(1),
                ..PhaseCommonSpec::default()
            },
            concurrency: 1,
        }
    }

    fn digest(byte: u8) -> ContentDigest {
        ContentDigest::from_bytes([byte; 32])
    }

    #[test]
    fn warmup_or_second_profiling_phase_is_refused_during_validation() {
        assert!(ensure_single_profiling_phase(&[phase("profiling", None)]).is_ok());
        assert!(
            ensure_single_profiling_phase(&[
                phase("warmup", Some(PhaseRoleSpec::Warmup)),
                phase("profiling", None),
            ])
            .is_err()
        );
        assert!(
            ensure_single_profiling_phase(&[phase("first", None), phase("second", None)]).is_err()
        );
        assert!(ensure_single_profiling_phase(&[]).is_err());
    }

    #[test]
    fn dynamo_composition_is_refused_during_capability_agreement() {
        assert!(ensure_supported_composition("local", "reference_jsonl").is_ok());
        assert!(ensure_supported_composition("local", "streaming_dynamo").is_err());
        assert!(ensure_supported_composition("dynamo", "reference_jsonl").is_err());
    }

    #[test]
    fn reliability_policy_digest_mismatch_is_refused() {
        assert!(ensure_reliability_policy_agreement(&digest(1), &digest(1)).is_ok());
        assert!(ensure_reliability_policy_agreement(&digest(1), &digest(2)).is_err());
    }

    #[test]
    fn restored_synthesis_authority_must_match_the_authored_plan() {
        let unbound = SynthesisAuthority::Unbound;
        let bound = SynthesisAuthority::Bound {
            session_program_digest: digest(7),
        };
        assert!(unbound.accept_restored(&unbound).is_ok());
        assert!(bound.accept_restored(&bound).is_ok());
        assert!(unbound.accept_restored(&bound).is_err());
        assert!(bound.accept_restored(&unbound).is_err());
    }
}
