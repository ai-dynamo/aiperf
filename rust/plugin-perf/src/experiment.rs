// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Experiment identity and the AB/BA measurement state machine.
//!
//! A parity verdict is only meaningful for the exact pair of binaries, the
//! exact lockfile, and the exact machine it was measured on. This module makes
//! that binding explicit: [`ExperimentIdentity`] is content-addressed over
//! every one of those facts and is frozen before the first warmup pair runs, so
//! a result document cannot be re-pointed at a different build after the fact.
//!
//! [`ExperimentRunner`] owns the ordering and admission rules around the
//! measurement itself:
//!
//! - [`WARMUP_ITERATIONS`] pairs run before anything is retained, because the
//!   first pairs pay page-fault, dynamic-relocation, and cache-population costs
//!   that are properties of process startup rather than of the plugin boundary.
//! - Pairs alternate static-first and dynamic-first ([`PairSchedule::balanced`])
//!   so a monotone drift in machine state — thermal, frequency, or page-cache —
//!   cannot be attributed to whichever build happened to run second.
//! - A *product* error aborts the experiment immediately. A crash or a protocol
//!   violation is a result, not noise, and rerunning until it goes away would
//!   launder a real defect.
//! - An *invalidation* (the rig was too noisy to have measured anything) may be
//!   retried, but only within a bounded budget, and never more than
//!   [`MAX_CONSECUTIVE_INVALIDATIONS`] times in a row — at that point the rig,
//!   not the sample, is the problem.

use std::fmt;

use crate::stats::{MINIMUM_RETAINED_PAIRS, PairedSamples, WARMUP_ITERATIONS};

/// Consecutive invalidations tolerated before the rig is declared unusable.
pub const MAX_CONSECUTIVE_INVALIDATIONS: usize = 3;

/// Total invalidations tolerated across one experiment.
///
/// Larger than the consecutive limit because a rig that recovers after each
/// stumble is still measurable; a rig that never recovers is caught by the
/// consecutive rule first.
pub const MAX_TOTAL_INVALIDATIONS: usize = 10;

/// A 32-byte content digest naming a build input.
pub type Digest = [u8; 32];

/// Errors raised while constructing or freezing an experiment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExperimentError {
    /// The static and dynamic sides name the same binary, so there is nothing
    /// to compare.
    IdenticalBinaries {
        /// The digest both sides supplied.
        digest: Digest,
    },
    /// Fewer retained pairs requested than the gate can bootstrap.
    TooFewRetainedPairs {
        /// Pairs requested.
        found: usize,
        /// Pairs required.
        required: usize,
    },
    /// Fewer warmup pairs requested than the gate discards.
    TooFewWarmups {
        /// Warmups requested.
        found: usize,
        /// Warmups required.
        required: usize,
    },
    /// A field that must name something was left empty.
    MissingField {
        /// The field left empty.
        field: &'static str,
    },
    /// A frozen identity was asked to adopt a different build input.
    FrozenIdentity {
        /// The field the caller tried to replace.
        field: &'static str,
    },
}

impl fmt::Display for ExperimentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IdenticalBinaries { digest } => write!(
                f,
                "static and dynamic sides share digest {}; there is nothing to compare",
                hex(digest)
            ),
            Self::TooFewRetainedPairs { found, required } => write!(
                f,
                "requested {found} retained pairs, the parity gate requires {required}"
            ),
            Self::TooFewWarmups { found, required } => write!(
                f,
                "requested {found} warmup pairs, the parity gate requires {required}"
            ),
            Self::MissingField { field } => write!(f, "experiment field `{field}` must not be empty"),
            Self::FrozenIdentity { field } => write!(
                f,
                "experiment identity is frozen; `{field}` cannot be replaced after freezing"
            ),
        }
    }
}

impl std::error::Error for ExperimentError {}

/// Lowercase hex rendering of a digest, for identity strings and diagnostics.
#[must_use]
pub fn hex(digest: &Digest) -> String {
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        // Writing two nibbles per byte cannot fail for a `String` sink.
        out.push(char::from_digit(u32::from(byte >> 4), 16).unwrap_or('0'));
        out.push(char::from_digit(u32::from(byte & 0x0f), 16).unwrap_or('0'));
    }
    out
}

/// Everything a parity experiment is bound to before it starts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExperimentSpec {
    /// Digest of the statically linked comparator binary.
    pub static_binary_digest: Digest,
    /// Digest of the dynamically loading candidate binary.
    pub dynamic_binary_digest: Digest,
    /// Digest of the workspace `Cargo.lock` both were built from.
    pub cargo_lock_digest: Digest,
    /// Digest of the harness itself, so a harness change invalidates results.
    pub harness_digest: Digest,
    /// Model string of the CPU the experiment ran on.
    pub cpu_model: String,
    /// Socket, NUMA, and capacity summary of the machine.
    pub memory_topology: String,
    /// Toolchain version both binaries were built with.
    pub rust_version: String,
    /// RFC 3339 UTC instant the experiment was frozen at.
    pub timestamp_utc: String,
    /// Metric under comparison, such as `ttft_p50` or `e2e_p50`.
    pub metric: String,
    /// Warmup pairs to discard.
    pub warmups: usize,
    /// Pairs to retain in each orientation.
    pub retained_pairs: usize,
}

impl ExperimentSpec {
    /// Validates a spec's shape without freezing it.
    ///
    /// Returns an error when the two sides are the same binary, when a required
    /// descriptive field is empty, or when the requested pair counts fall below
    /// what the bootstrap needs.
    pub fn validate(&self) -> Result<(), ExperimentError> {
        if self.static_binary_digest == self.dynamic_binary_digest {
            return Err(ExperimentError::IdenticalBinaries {
                digest: self.static_binary_digest,
            });
        }
        for (field, value) in [
            ("cpu_model", &self.cpu_model),
            ("memory_topology", &self.memory_topology),
            ("rust_version", &self.rust_version),
            ("timestamp_utc", &self.timestamp_utc),
            ("metric", &self.metric),
        ] {
            if value.trim().is_empty() {
                return Err(ExperimentError::MissingField { field });
            }
        }
        if self.warmups < WARMUP_ITERATIONS {
            return Err(ExperimentError::TooFewWarmups {
                found: self.warmups,
                required: WARMUP_ITERATIONS,
            });
        }
        if self.retained_pairs < MINIMUM_RETAINED_PAIRS {
            return Err(ExperimentError::TooFewRetainedPairs {
                found: self.retained_pairs,
                required: MINIMUM_RETAINED_PAIRS,
            });
        }
        Ok(())
    }

    /// A valid spec with fixed synthetic inputs, for tests and examples.
    ///
    /// It names no real binary; it exists so state-machine behaviour can be
    /// exercised without building two products first.
    #[must_use]
    pub fn synthetic_fixture() -> Self {
        Self {
            static_binary_digest: [0x11; 32],
            dynamic_binary_digest: [0x22; 32],
            cargo_lock_digest: [0x33; 32],
            harness_digest: [0x44; 32],
            cpu_model: "synthetic-cpu".to_owned(),
            memory_topology: "synthetic-topology".to_owned(),
            rust_version: "synthetic-toolchain".to_owned(),
            timestamp_utc: "2026-08-27T00:00:00Z".to_owned(),
            metric: "ttft_p50".to_owned(),
            warmups: WARMUP_ITERATIONS,
            retained_pairs: MINIMUM_RETAINED_PAIRS,
        }
    }
}

/// The frozen, content-addressed name of one experiment.
///
/// `experiment_id` is BLAKE3 over the canonical newline-joined rendering of
/// every other field, in declaration order, so an external reader can recompute
/// it from the published document without this crate.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ExperimentIdentity {
    /// Lowercase-hex BLAKE3 over the canonical rendering of the fields below.
    pub experiment_id: String,
    /// Digest of the statically linked comparator binary.
    pub static_binary_digest: String,
    /// Digest of the dynamically loading candidate binary.
    pub dynamic_binary_digest: String,
    /// Digest of the workspace `Cargo.lock`.
    pub cargo_lock_digest: String,
    /// Digest of the harness that produced the result.
    pub harness_digest: String,
    /// Model string of the CPU the experiment ran on.
    pub cpu_model: String,
    /// Socket, NUMA, and capacity summary of the machine.
    pub memory_topology: String,
    /// Toolchain version both binaries were built with.
    pub rust_version: String,
    /// RFC 3339 UTC instant the experiment was frozen at.
    pub timestamp_utc: String,
    /// Metric under comparison.
    pub metric: String,
}

impl ExperimentIdentity {
    /// Freezes a validated spec into a content-addressed identity.
    pub fn from_spec(spec: &ExperimentSpec) -> Result<Self, ExperimentError> {
        spec.validate()?;
        Ok(Self::assemble(
            hex(&spec.static_binary_digest),
            hex(&spec.dynamic_binary_digest),
            hex(&spec.cargo_lock_digest),
            hex(&spec.harness_digest),
            spec.cpu_model.clone(),
            spec.memory_topology.clone(),
            spec.rust_version.clone(),
            spec.timestamp_utc.clone(),
            spec.metric.clone(),
        ))
    }

    /// The canonical pre-image `experiment_id` is computed over.
    #[must_use]
    pub fn canonical_preimage(&self) -> String {
        [
            self.static_binary_digest.as_str(),
            self.dynamic_binary_digest.as_str(),
            self.cargo_lock_digest.as_str(),
            self.harness_digest.as_str(),
            self.cpu_model.as_str(),
            self.memory_topology.as_str(),
            self.rust_version.as_str(),
            self.timestamp_utc.as_str(),
            self.metric.as_str(),
        ]
        .join("\n")
    }

    /// Rebinding a frozen identity to a different harness build.
    ///
    /// Succeeds only when the digest is unchanged, which makes the call a
    /// no-op confirmation. Any other digest names a different experiment and is
    /// refused, so results cannot be carried across a harness change.
    pub fn clone_with_harness_digest(&self, digest: Digest) -> Result<Self, ExperimentError> {
        if hex(&digest) != self.harness_digest {
            return Err(ExperimentError::FrozenIdentity {
                field: "harness_digest",
            });
        }
        Ok(self.clone())
    }

    #[allow(clippy::too_many_arguments)]
    fn assemble(
        static_binary_digest: String,
        dynamic_binary_digest: String,
        cargo_lock_digest: String,
        harness_digest: String,
        cpu_model: String,
        memory_topology: String,
        rust_version: String,
        timestamp_utc: String,
        metric: String,
    ) -> Self {
        let mut identity = Self {
            experiment_id: String::new(),
            static_binary_digest,
            dynamic_binary_digest,
            cargo_lock_digest,
            harness_digest,
            cpu_model,
            memory_topology,
            rust_version,
            timestamp_utc,
            metric,
        };
        identity.experiment_id = blake3::hash(identity.canonical_preimage().as_bytes())
            .to_hex()
            .to_string();
        identity
    }
}

/// Which build runs first in one measurement pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairOrder {
    /// Static build first, dynamic second.
    StaticFirst,
    /// Dynamic build first, static second.
    DynamicFirst,
}

impl PairOrder {
    /// Whether this pair runs the static build first.
    #[must_use]
    pub fn is_ab(self) -> bool {
        matches!(self, Self::StaticFirst)
    }

    /// Whether this pair runs the dynamic build first.
    #[must_use]
    pub fn is_ba(self) -> bool {
        matches!(self, Self::DynamicFirst)
    }
}

/// The ordered plan of pair orientations for one experiment.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PairSchedule {
    orders: Vec<PairOrder>,
}

impl PairSchedule {
    /// A schedule of `pairs` entries alternating static-first and dynamic-first.
    ///
    /// An even `pairs` count splits exactly evenly, which is what balances the
    /// design against monotone machine drift.
    #[must_use]
    pub fn balanced(pairs: usize) -> Self {
        let orders = (0..pairs)
            .map(|index| {
                if index % 2 == 0 {
                    PairOrder::StaticFirst
                } else {
                    PairOrder::DynamicFirst
                }
            })
            .collect();
        Self { orders }
    }

    /// Number of pairs in the schedule.
    #[must_use]
    pub fn len(&self) -> usize {
        self.orders.len()
    }

    /// Whether the schedule has no pairs.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.orders.is_empty()
    }

    /// Iterates the pair orientations in execution order.
    pub fn iter(&self) -> std::slice::Iter<'_, PairOrder> {
        self.orders.iter()
    }
}

impl<'a> IntoIterator for &'a PairSchedule {
    type Item = &'a PairOrder;
    type IntoIter = std::slice::Iter<'a, PairOrder>;

    fn into_iter(self) -> Self::IntoIter {
        self.orders.iter()
    }
}

/// Why a measured pair was discarded without counting as a result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum InvalidationReason {
    /// Retained samples were too dispersed to have measured anything.
    CvExceeded,
    /// The machine drifted mid-pair, so the two halves are not comparable.
    ThermalDrift,
    /// Another process contended for the measured cores.
    Interference,
}

impl fmt::Display for InvalidationReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CvExceeded => f.write_str("coefficient of variation exceeded"),
            Self::ThermalDrift => f.write_str("thermal drift within the pair"),
            Self::Interference => f.write_str("interference on the measured cores"),
        }
    }
}

/// Where an experiment is in its lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExperimentPhase {
    /// Discarding warmup pairs.
    Warmup,
    /// Retaining pairs for the bootstrap.
    Measuring,
    /// A verdict was reached.
    Complete,
    /// The experiment was abandoned and has no verdict.
    Failed,
}

/// What the runner decided about one attempted pair.
#[derive(Debug, Clone, PartialEq)]
pub enum AttemptOutcome {
    /// The pair was retained.
    Recorded {
        /// Retained pairs so far, in each orientation.
        retained: usize,
    },
    /// The pair was discarded; the experiment may continue.
    Invalidated {
        /// Why the pair was discarded.
        reason: InvalidationReason,
        /// Consecutive invalidations including this one.
        consecutive: usize,
    },
    /// The experiment is over and produced no verdict.
    ImmediateFailure {
        /// Why the experiment was abandoned.
        reason: String,
    },
    /// The experiment is over and the dynamic build lost measurable ground.
    ConfirmedRegression {
        /// The retention-ratio lower bound that failed the gate.
        lower_bound: f64,
    },
}

/// Drives one AB/BA parity experiment through its admission rules.
#[derive(Debug, Clone)]
pub struct ExperimentRunner {
    spec: ExperimentSpec,
    identity: ExperimentIdentity,
    schedule: PairSchedule,
    samples: PairedSamples,
    phase: ExperimentPhase,
    consecutive_invalidations: usize,
    total_invalidations: usize,
}

impl ExperimentRunner {
    /// Validates and freezes a spec, producing a runner ready for warmup.
    pub fn new(spec: ExperimentSpec) -> Result<Self, ExperimentError> {
        let identity = ExperimentIdentity::from_spec(&spec)?;
        let schedule = PairSchedule::balanced(spec.retained_pairs);
        Ok(Self {
            spec,
            identity,
            schedule,
            samples: PairedSamples::default(),
            phase: ExperimentPhase::Warmup,
            consecutive_invalidations: 0,
            total_invalidations: 0,
        })
    }

    /// The frozen identity of this experiment.
    #[must_use]
    pub fn freeze_identity(&self) -> ExperimentIdentity {
        self.identity.clone()
    }

    /// The spec this runner was frozen from.
    #[must_use]
    pub fn spec(&self) -> &ExperimentSpec {
        &self.spec
    }

    /// The planned pair orientations.
    #[must_use]
    pub fn schedule(&self) -> &PairSchedule {
        &self.schedule
    }

    /// Warmup pairs discarded before retention begins.
    #[must_use]
    pub fn warmup_count(&self) -> usize {
        self.spec.warmups
    }

    /// The current lifecycle phase.
    #[must_use]
    pub fn current_phase(&self) -> ExperimentPhase {
        self.phase
    }

    /// Retained samples collected so far.
    #[must_use]
    pub fn samples(&self) -> &PairedSamples {
        &self.samples
    }

    /// Records one retained pair, given both halves of both orientations.
    ///
    /// Arguments are in execution order: the AB pair's static then dynamic
    /// halves, then the BA pair's dynamic then static halves. Retaining a pair
    /// clears the consecutive-invalidation counter, since the rig demonstrably
    /// recovered.
    pub fn record_valid_pair(
        &mut self,
        ab_static_ns: f64,
        ab_dynamic_ns: f64,
        ba_dynamic_ns: f64,
        ba_static_ns: f64,
    ) -> AttemptOutcome {
        if matches!(self.phase, ExperimentPhase::Failed | ExperimentPhase::Complete) {
            return AttemptOutcome::ImmediateFailure {
                reason: "experiment already concluded".to_owned(),
            };
        }
        self.phase = ExperimentPhase::Measuring;
        self.samples.ab.push((ab_static_ns, ab_dynamic_ns));
        self.samples.ba.push((ba_dynamic_ns, ba_static_ns));
        self.consecutive_invalidations = 0;
        AttemptOutcome::Recorded {
            retained: self.samples.ab.len(),
        }
    }

    /// Records a pair the rig could not measure cleanly.
    ///
    /// Returns [`AttemptOutcome::ImmediateFailure`] once either the consecutive
    /// or the total budget is exhausted, because at that point the machine is
    /// the finding.
    pub fn record_invalidation(&mut self, reason: InvalidationReason) -> AttemptOutcome {
        if matches!(self.phase, ExperimentPhase::Failed | ExperimentPhase::Complete) {
            return AttemptOutcome::ImmediateFailure {
                reason: "experiment already concluded".to_owned(),
            };
        }
        self.consecutive_invalidations += 1;
        self.total_invalidations += 1;
        if self.consecutive_invalidations > MAX_CONSECUTIVE_INVALIDATIONS {
            self.phase = ExperimentPhase::Failed;
            return AttemptOutcome::ImmediateFailure {
                reason: format!(
                    "{} consecutive invalidations ({reason}); the rig is not measurable",
                    self.consecutive_invalidations
                ),
            };
        }
        if self.total_invalidations > MAX_TOTAL_INVALIDATIONS {
            self.phase = ExperimentPhase::Failed;
            return AttemptOutcome::ImmediateFailure {
                reason: format!(
                    "{} total invalidations ({reason}); the rig is not measurable",
                    self.total_invalidations
                ),
            };
        }
        AttemptOutcome::Invalidated {
            reason,
            consecutive: self.consecutive_invalidations,
        }
    }

    /// Records a defect in the product under measurement.
    ///
    /// This is terminal: a crash, a non-zero exit, or a protocol violation is a
    /// result about the build, and retrying until it disappears would hide it.
    pub fn record_product_error(&mut self, detail: impl Into<String>) -> AttemptOutcome {
        self.phase = ExperimentPhase::Failed;
        AttemptOutcome::ImmediateFailure {
            reason: detail.into(),
        }
    }

    /// Records a measured regression the bootstrap confirmed.
    pub fn record_confirmed_regression(&mut self, lower_bound: f64) -> AttemptOutcome {
        self.phase = ExperimentPhase::Complete;
        AttemptOutcome::ConfirmedRegression { lower_bound }
    }

    /// Whether the harness would run this experiment again.
    ///
    /// True only while the experiment is still collecting. A concluded
    /// experiment — whether it failed the gate or was abandoned — is never
    /// rerun, because rerunning a valid failure until it passes is exactly the
    /// laundering this harness exists to prevent.
    #[must_use]
    pub fn would_rerun(&self) -> bool {
        matches!(
            self.phase,
            ExperimentPhase::Warmup | ExperimentPhase::Measuring
        )
    }
}
