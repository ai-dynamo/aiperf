// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! AgentX agentic-replay timing mode — a faithful port of Python's
//! `AgenticReplayStrategy` (`src/aiperf/timing/strategies/agentic_replay.py`).
//!
//! A scheduled-runtime [`Workload`]: it owns its own dispatch timing (per-lane
//! t\* sampling, warmup global-alignment, profiling offsets, cache-bust) and
//! reuses the shared [`ScheduledRuntime`] for transport/metrics/records/report.
//! It is selected per-phase by [`crate::engine::protocol::PhaseSpec::AgenticReplay`]
//! and runs as two instances — a WARMUP phase (dispatches the last pre-t\* turn of
//! each stream to prime server KV) and a PROFILING phase (resumes at t\* and
//! replays each stream at its recorded offsets).
//!
//! The pure timing kernel lives in [`crate::agentx::trajectory_source`] and the
//! byte-exact marker builder in [`crate::agentx::cache_bust`]; this module drives
//! them against the [`crate::multiturn::ConversationSource`] seam.

use std::cell::RefCell;
use std::rc::Rc;

use anyhow::Result;
use async_trait::async_trait;

use crate::agentx::cache_bust::CacheBustTarget;
use crate::multiturn::ConversationSource;
use crate::scheduled::{ScheduledRuntime, Workload};

/// Which phase an [`AgenticReplayWorkload`] instance drives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgenticPhase {
    /// Dispatch the warmup turn (n-1) of each stream to prime the server KV.
    Warmup,
    /// Resume each stream at t\* and replay its post-t\* turns at their offsets.
    Profiling,
}

/// Configuration for one agentic-replay phase instance (from the authored
/// [`PhaseSpec::AgenticReplay`](crate::engine::protocol::PhaseSpec) + run identity).
#[derive(Debug, Clone)]
pub struct AgenticReplayConfig {
    /// Which phase this instance drives.
    pub phase: AgenticPhase,
    /// Trajectory-start window lower ratio.
    pub start_min_ratio: f64,
    /// Trajectory-start window upper ratio.
    pub start_max_ratio: f64,
    /// Idle-gap cap in ms for warmup-lead / leading-idle capping (`None` = uncapped).
    pub idle_gap_cap_ms: Option<f64>,
    /// Anchor phase-start bursts at the earliest post-t\* request instead of spread.
    pub burst_phase_starts: bool,
    /// Base random seed for per-lane t\* sampling.
    pub random_seed: u64,
    /// The run's benchmark id (cache-bust digest input).
    pub benchmark_id: String,
    /// Cache-bust placement (scenario-locked to first-turn-prefix for the MVP).
    pub cache_bust_target: CacheBustTarget,
}

/// The agentic-replay workload: drives one phase's dispatch over a
/// [`ConversationSource`] of reconstructed WEKA trajectories.
pub struct AgenticReplayWorkload {
    #[allow(dead_code)]
    source: Rc<RefCell<Box<dyn ConversationSource>>>,
    #[allow(dead_code)]
    config: AgenticReplayConfig,
}

impl AgenticReplayWorkload {
    /// Build the workload over a reconstructed-trajectory conversation source.
    pub fn new(
        source: Box<dyn ConversationSource>,
        config: AgenticReplayConfig,
    ) -> Result<Self> {
        Ok(Self {
            source: Rc::new(RefCell::new(source)),
            config,
        })
    }
}

#[async_trait(?Send)]
impl Workload for AgenticReplayWorkload {
    fn name(&self) -> &'static str {
        "agentic_replay"
    }

    /// Authored per-turn dispatch times (not credit-paced), like fixed_schedule.
    fn has_credit_timestamps(&self) -> bool {
        false
    }

    async fn execute(&self, _runtime: Rc<ScheduledRuntime>) -> Result<()> {
        // Dispatch is implemented in the following increment (warmup global t\*-
        // alignment / profiling offset replay via schedule_at_ns + issue_turn),
        // once the SampledSession turn-building seam is wired. Emitting no work
        // here keeps the phase a clean no-op rather than a hang.
        tracing::warn!(
            phase = ?self.config.phase,
            "agentic_replay dispatch not yet implemented; phase issues no requests"
        );
        Ok(())
    }
}
