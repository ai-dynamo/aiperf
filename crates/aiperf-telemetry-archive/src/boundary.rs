// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Atomic source-cardinal boundary capture plans.
//!
//! Boundary membership is topology, not a timestamp inference. A plan is
//! validated in full before any source driver may observe it, and a registry
//! permanently rejects late subscribers or reused transition/group IDs.

use std::collections::BTreeSet;
use std::fmt::{self, Display, Formatter};

use serde::{Deserialize, Serialize};

/// Phase-side meaning of one forced source snapshot subscription.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundaryRole {
    /// Snapshot used as the baseline for a phase that is starting.
    PhaseStart,
    /// Snapshot used as the terminal observation for a phase that is ending.
    PhaseEnd,
}

/// Complete source-scoped join key shared by markers and attempt/loss rows.
#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BoundaryReference {
    /// Adjacent-phase transition that owns the reference.
    pub transition_id: String,
    /// Stable identity of this individual phase/source boundary.
    pub boundary_id: String,
    /// Phase receiving the snapshot.
    pub phase_id: String,
    /// Physical telemetry source that must satisfy the reference.
    pub source_id: String,
    /// Start/end interpretation within the receiving phase.
    pub role: BoundaryRole,
    /// Source-local coalescing group when multiple subscribers share a fetch.
    pub coalescing_group_id: Option<String>,
}

impl BoundaryReference {
    /// Returns the exact attempt-or-loss join key.
    #[must_use]
    pub fn key(&self) -> BoundaryReferenceKey<'_> {
        BoundaryReferenceKey {
            transition_id: &self.transition_id,
            source_id: &self.source_id,
            boundary_id: &self.boundary_id,
        }
    }
}

/// Borrowed exact join key used to prove attempt-or-loss cardinality.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct BoundaryReferenceKey<'a> {
    /// Transition identity.
    pub transition_id: &'a str,
    /// Physical source identity.
    pub source_id: &'a str,
    /// Boundary identity within that source/transition.
    pub boundary_id: &'a str,
}

/// One physical source command inside an atomic transition plan.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SourceBoundarySnapshotCommand {
    /// Physical source receiving this command.
    pub source_id: String,
    /// Group shared by every subscriber when the physical fetch is coalesced.
    pub coalescing_group_id: Option<String>,
    /// Non-empty phase subscriptions satisfied by the same attempt or loss.
    pub subscribers: Vec<BoundaryReference>,
    /// Absolute injected-Clock deadline for this source capture.
    pub absolute_deadline_ns: i64,
}

/// Complete plan that must contain exactly one command per expected source.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BoundaryCapturePlan {
    /// Stable identity allocated once for the adjacent-phase transition.
    pub transition_id: String,
    /// Source-cardinal commands, routed only after atomic registration.
    pub commands: Vec<SourceBoundarySnapshotCommand>,
}

/// Immutable proof that a complete plan was registered atomically.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SealedBoundaryCapturePlan(BoundaryCapturePlan);

impl SealedBoundaryCapturePlan {
    /// Borrows the immutable registered plan.
    #[must_use]
    pub const fn plan(&self) -> &BoundaryCapturePlan {
        &self.0
    }

    /// Consumes the proof and returns the immutable plan value for routing.
    #[must_use]
    pub fn into_plan(self) -> BoundaryCapturePlan {
        self.0
    }
}

/// Run-owned registry enforcing transition and group uniqueness.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BoundaryPlanRegistry {
    expected_sources: BTreeSet<String>,
    sealed_transitions: BTreeSet<String>,
    used_coalescing_groups: BTreeSet<String>,
}

impl BoundaryPlanRegistry {
    /// Creates a registry for the exact physical source set prepared by a run.
    pub fn new<I, S>(expected_sources: I) -> Result<Self, BoundaryPlanError>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut sources = BTreeSet::new();
        for source in expected_sources {
            let source = source.into();
            validate_identifier("source_id", &source)?;
            if !sources.insert(source.clone()) {
                return Err(BoundaryPlanError::DuplicateExpectedSource(source));
            }
        }
        if sources.is_empty() {
            return Err(BoundaryPlanError::NoExpectedSources);
        }
        Ok(Self {
            expected_sources: sources,
            sealed_transitions: BTreeSet::new(),
            used_coalescing_groups: BTreeSet::new(),
        })
    }

    /// Validates and registers a complete plan as one atomic state change.
    ///
    /// No transition or group identity is consumed when validation fails.
    pub fn seal(
        &mut self,
        plan: BoundaryCapturePlan,
    ) -> Result<SealedBoundaryCapturePlan, BoundaryPlanError> {
        validate_identifier("transition_id", &plan.transition_id)?;
        if self.sealed_transitions.contains(&plan.transition_id) {
            return Err(BoundaryPlanError::TransitionAlreadySealed(
                plan.transition_id,
            ));
        }
        if plan.commands.len() != self.expected_sources.len() {
            return Err(BoundaryPlanError::SourceCardinality {
                expected: self.expected_sources.len(),
                actual: plan.commands.len(),
            });
        }

        let mut command_sources = BTreeSet::new();
        let mut new_groups = BTreeSet::new();
        let mut references = BTreeSet::new();
        for command in &plan.commands {
            validate_identifier("command.source_id", &command.source_id)?;
            if !self.expected_sources.contains(&command.source_id) {
                return Err(BoundaryPlanError::UnexpectedSource(
                    command.source_id.clone(),
                ));
            }
            if !command_sources.insert(command.source_id.clone()) {
                return Err(BoundaryPlanError::DuplicateSourceCommand(
                    command.source_id.clone(),
                ));
            }
            if command.subscribers.is_empty() {
                return Err(BoundaryPlanError::EmptySubscribers(
                    command.source_id.clone(),
                ));
            }

            validate_group_shape(command)?;
            if let Some(group) = &command.coalescing_group_id {
                validate_identifier("coalescing_group_id", group)?;
                if self.used_coalescing_groups.contains(group) || !new_groups.insert(group.clone())
                {
                    return Err(BoundaryPlanError::CoalescingGroupReused(group.clone()));
                }
            }

            for reference in &command.subscribers {
                validate_reference(reference)?;
                if reference.transition_id != plan.transition_id {
                    return Err(BoundaryPlanError::ReferenceTransitionMismatch {
                        expected: plan.transition_id.clone(),
                        actual: reference.transition_id.clone(),
                    });
                }
                if reference.source_id != command.source_id {
                    return Err(BoundaryPlanError::ReferenceSourceMismatch {
                        expected: command.source_id.clone(),
                        actual: reference.source_id.clone(),
                    });
                }
                let key = (
                    reference.transition_id.clone(),
                    reference.source_id.clone(),
                    reference.boundary_id.clone(),
                );
                if !references.insert(key) {
                    return Err(BoundaryPlanError::DuplicateReference {
                        transition_id: reference.transition_id.clone(),
                        source_id: reference.source_id.clone(),
                        boundary_id: reference.boundary_id.clone(),
                    });
                }
            }
        }

        if command_sources != self.expected_sources {
            let missing = self
                .expected_sources
                .difference(&command_sources)
                .cloned()
                .collect();
            return Err(BoundaryPlanError::MissingSources(missing));
        }

        self.sealed_transitions.insert(plan.transition_id.clone());
        self.used_coalescing_groups.extend(new_groups);
        Ok(SealedBoundaryCapturePlan(plan))
    }

    /// Returns expected sources in deterministic identity order.
    pub fn expected_sources(&self) -> impl ExactSizeIterator<Item = &str> {
        self.expected_sources.iter().map(String::as_str)
    }
}

fn validate_group_shape(command: &SourceBoundarySnapshotCommand) -> Result<(), BoundaryPlanError> {
    match command.subscribers.len() {
        0 => unreachable!("empty subscribers are rejected before group validation"),
        1 => {
            if command.coalescing_group_id.is_some()
                || command.subscribers[0].coalescing_group_id.is_some()
            {
                return Err(BoundaryPlanError::SingleSubscriberHasGroup(
                    command.source_id.clone(),
                ));
            }
        }
        _ => {
            let expected = command.coalescing_group_id.as_ref().ok_or_else(|| {
                BoundaryPlanError::CoalescedCommandMissingGroup(command.source_id.clone())
            })?;
            for reference in &command.subscribers {
                if reference.coalescing_group_id.as_ref() != Some(expected) {
                    return Err(BoundaryPlanError::ReferenceGroupMismatch {
                        source_id: command.source_id.clone(),
                        expected: expected.clone(),
                        actual: reference.coalescing_group_id.clone(),
                    });
                }
            }
        }
    }
    Ok(())
}

fn validate_reference(reference: &BoundaryReference) -> Result<(), BoundaryPlanError> {
    for (field, value) in [
        ("reference.transition_id", reference.transition_id.as_str()),
        ("reference.boundary_id", reference.boundary_id.as_str()),
        ("reference.phase_id", reference.phase_id.as_str()),
        ("reference.source_id", reference.source_id.as_str()),
    ] {
        validate_identifier(field, value)?;
    }
    if let Some(group) = &reference.coalescing_group_id {
        validate_identifier("reference.coalescing_group_id", group)?;
    }
    Ok(())
}

fn validate_identifier(field: &'static str, value: &str) -> Result<(), BoundaryPlanError> {
    if value.is_empty() || value.trim() != value || value.chars().any(char::is_control) {
        return Err(BoundaryPlanError::InvalidIdentifier {
            field,
            value: value.to_owned(),
        });
    }
    Ok(())
}

/// Rejected source topology or illegal boundary-plan mutation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BoundaryPlanError {
    /// A run cannot seal a source-cardinal plan without physical sources.
    NoExpectedSources,
    /// The prepared source inventory itself contained a duplicate.
    DuplicateExpectedSource(String),
    /// A stable identity was empty, padded, or contained a control character.
    InvalidIdentifier {
        /// Field carrying the invalid identifier.
        field: &'static str,
        /// Redaction-safe authored value.
        value: String,
    },
    /// A sealed transition cannot accept a second plan or late subscriber.
    TransitionAlreadySealed(String),
    /// The plan command count differs from the prepared physical source count.
    SourceCardinality {
        /// Prepared source count.
        expected: usize,
        /// Authored command count.
        actual: usize,
    },
    /// A command referred to a source outside the prepared inventory.
    UnexpectedSource(String),
    /// More than one command targeted the same source in one transition.
    DuplicateSourceCommand(String),
    /// Exact expected sources were absent after validating all commands.
    MissingSources(Vec<String>),
    /// Every physical command must have at least one subscriber.
    EmptySubscribers(String),
    /// A single subscriber is never represented as a coalesced group.
    SingleSubscriberHasGroup(String),
    /// Multiple subscribers require one source-local group identity.
    CoalescedCommandMissingGroup(String),
    /// A group identity was reused in this plan or an earlier transition.
    CoalescingGroupReused(String),
    /// A subscriber did not copy its owning command's group exactly.
    ReferenceGroupMismatch {
        /// Physical source receiving the command.
        source_id: String,
        /// Group required by the command.
        expected: String,
        /// Group carried by the inconsistent subscriber.
        actual: Option<String>,
    },
    /// A subscriber carried a transition different from its plan.
    ReferenceTransitionMismatch {
        /// Plan transition.
        expected: String,
        /// Subscriber transition.
        actual: String,
    },
    /// A subscriber carried a source different from its command.
    ReferenceSourceMismatch {
        /// Command source.
        expected: String,
        /// Subscriber source.
        actual: String,
    },
    /// The exact attempt-or-loss join key occurred more than once.
    DuplicateReference {
        /// Transition identity.
        transition_id: String,
        /// Source identity.
        source_id: String,
        /// Boundary identity.
        boundary_id: String,
    },
}

impl Display for BoundaryPlanError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoExpectedSources => {
                formatter.write_str("boundary plan registry requires at least one source")
            }
            Self::DuplicateExpectedSource(source) => {
                write!(formatter, "duplicate expected boundary source {source:?}")
            }
            Self::InvalidIdentifier { field, value } => {
                write!(formatter, "{field} has invalid identifier {value:?}")
            }
            Self::TransitionAlreadySealed(transition) => {
                write!(
                    formatter,
                    "boundary transition {transition:?} is already sealed"
                )
            }
            Self::SourceCardinality { expected, actual } => write!(
                formatter,
                "boundary plan has {actual} source commands; expected exactly {expected}"
            ),
            Self::UnexpectedSource(source) => {
                write!(
                    formatter,
                    "boundary command references unknown source {source:?}"
                )
            }
            Self::DuplicateSourceCommand(source) => {
                write!(
                    formatter,
                    "duplicate boundary command for source {source:?}"
                )
            }
            Self::MissingSources(sources) => {
                write!(formatter, "boundary plan is missing sources {sources:?}")
            }
            Self::EmptySubscribers(source) => {
                write!(
                    formatter,
                    "boundary command for source {source:?} has no subscribers"
                )
            }
            Self::SingleSubscriberHasGroup(source) => write!(
                formatter,
                "single-subscriber boundary command for source {source:?} must not have a coalescing group"
            ),
            Self::CoalescedCommandMissingGroup(source) => write!(
                formatter,
                "coalesced boundary command for source {source:?} requires a group"
            ),
            Self::CoalescingGroupReused(group) => {
                write!(formatter, "boundary coalescing group {group:?} was reused")
            }
            Self::ReferenceGroupMismatch {
                source_id,
                expected,
                actual,
            } => write!(
                formatter,
                "boundary subscriber for source {source_id:?} has group {actual:?}; expected {expected:?}"
            ),
            Self::ReferenceTransitionMismatch { expected, actual } => write!(
                formatter,
                "boundary subscriber transition {actual:?} does not match plan {expected:?}"
            ),
            Self::ReferenceSourceMismatch { expected, actual } => write!(
                formatter,
                "boundary subscriber source {actual:?} does not match command {expected:?}"
            ),
            Self::DuplicateReference {
                transition_id,
                source_id,
                boundary_id,
            } => write!(
                formatter,
                "duplicate boundary reference ({transition_id:?}, {source_id:?}, {boundary_id:?})"
            ),
        }
    }
}

impl std::error::Error for BoundaryPlanError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(
        transition: &str,
        source: &str,
        boundary: &str,
        phase: &str,
        role: BoundaryRole,
        group: Option<&str>,
    ) -> BoundaryReference {
        BoundaryReference {
            transition_id: transition.to_owned(),
            boundary_id: boundary.to_owned(),
            phase_id: phase.to_owned(),
            source_id: source.to_owned(),
            role,
            coalescing_group_id: group.map(str::to_owned),
        }
    }

    fn single_command(transition: &str, source: &str) -> SourceBoundarySnapshotCommand {
        SourceBoundarySnapshotCommand {
            source_id: source.to_owned(),
            coalescing_group_id: None,
            subscribers: vec![reference(
                transition,
                source,
                &format!("{source}-end"),
                "phase-a",
                BoundaryRole::PhaseEnd,
                None,
            )],
            absolute_deadline_ns: 100,
        }
    }

    #[test]
    fn exact_source_cardinality_seals_before_routing() {
        let mut registry = BoundaryPlanRegistry::new(["node-b", "node-a"]).unwrap();
        let plan = BoundaryCapturePlan {
            transition_id: "transition-1".to_owned(),
            commands: vec![
                single_command("transition-1", "node-a"),
                single_command("transition-1", "node-b"),
            ],
        };

        let sealed = registry.seal(plan.clone()).unwrap();

        assert_eq!(sealed.plan(), &plan);
        assert_eq!(
            registry.expected_sources().collect::<Vec<_>>(),
            vec!["node-a", "node-b"]
        );
        assert!(matches!(
            registry.seal(plan),
            Err(BoundaryPlanError::TransitionAlreadySealed(_))
        ));
    }

    #[test]
    fn adjacent_end_start_subscribers_share_only_one_source_local_group() {
        let command = |source: &str, group: &str| SourceBoundarySnapshotCommand {
            source_id: source.to_owned(),
            coalescing_group_id: Some(group.to_owned()),
            subscribers: vec![
                reference(
                    "transition-1",
                    source,
                    &format!("{source}-end"),
                    "phase-a",
                    BoundaryRole::PhaseEnd,
                    Some(group),
                ),
                reference(
                    "transition-1",
                    source,
                    &format!("{source}-start"),
                    "phase-b",
                    BoundaryRole::PhaseStart,
                    Some(group),
                ),
            ],
            absolute_deadline_ns: 100,
        };
        let mut registry = BoundaryPlanRegistry::new(["node-a", "node-b"]).unwrap();
        registry
            .seal(BoundaryCapturePlan {
                transition_id: "transition-1".to_owned(),
                commands: vec![command("node-a", "group-a"), command("node-b", "group-b")],
            })
            .unwrap();
    }

    #[test]
    fn groups_cannot_cross_sources_or_be_reused_later() {
        let grouped = |transition: &str, source: &str, group: &str| SourceBoundarySnapshotCommand {
            source_id: source.to_owned(),
            coalescing_group_id: Some(group.to_owned()),
            subscribers: vec![
                reference(
                    transition,
                    source,
                    &format!("{source}-a"),
                    "phase-a",
                    BoundaryRole::PhaseEnd,
                    Some(group),
                ),
                reference(
                    transition,
                    source,
                    &format!("{source}-b"),
                    "phase-b",
                    BoundaryRole::PhaseStart,
                    Some(group),
                ),
            ],
            absolute_deadline_ns: 100,
        };
        let mut registry = BoundaryPlanRegistry::new(["node-a", "node-b"]).unwrap();
        let error = registry
            .seal(BoundaryCapturePlan {
                transition_id: "transition-bad".to_owned(),
                commands: vec![
                    grouped("transition-bad", "node-a", "shared"),
                    grouped("transition-bad", "node-b", "shared"),
                ],
            })
            .unwrap_err();
        assert_eq!(
            error,
            BoundaryPlanError::CoalescingGroupReused("shared".to_owned())
        );

        registry
            .seal(BoundaryCapturePlan {
                transition_id: "transition-good".to_owned(),
                commands: vec![
                    grouped("transition-good", "node-a", "group-a"),
                    grouped("transition-good", "node-b", "group-b"),
                ],
            })
            .unwrap();
        let error = registry
            .seal(BoundaryCapturePlan {
                transition_id: "transition-later".to_owned(),
                commands: vec![
                    grouped("transition-later", "node-a", "group-a"),
                    grouped("transition-later", "node-b", "group-c"),
                ],
            })
            .unwrap_err();
        assert_eq!(
            error,
            BoundaryPlanError::CoalescingGroupReused("group-a".to_owned())
        );
    }

    #[test]
    fn failed_registration_is_atomic_and_consumes_no_identity() {
        let mut registry = BoundaryPlanRegistry::new(["node-a", "node-b"]).unwrap();
        let bad = BoundaryCapturePlan {
            transition_id: "transition-1".to_owned(),
            commands: vec![single_command("transition-1", "node-a")],
        };
        assert!(matches!(
            registry.seal(bad),
            Err(BoundaryPlanError::SourceCardinality { .. })
        ));

        registry
            .seal(BoundaryCapturePlan {
                transition_id: "transition-1".to_owned(),
                commands: vec![
                    single_command("transition-1", "node-a"),
                    single_command("transition-1", "node-b"),
                ],
            })
            .unwrap();
    }

    #[test]
    fn inconsistent_embedded_identity_and_duplicate_join_keys_fail() {
        let mut registry = BoundaryPlanRegistry::new(["node-a"]).unwrap();
        let mut command = single_command("transition-1", "node-a");
        command.subscribers[0].source_id = "node-b".to_owned();
        assert!(matches!(
            registry.seal(BoundaryCapturePlan {
                transition_id: "transition-1".to_owned(),
                commands: vec![command],
            }),
            Err(BoundaryPlanError::ReferenceSourceMismatch { .. })
        ));

        let mut registry = BoundaryPlanRegistry::new(["node-a"]).unwrap();
        let duplicated = reference(
            "transition-2",
            "node-a",
            "boundary-a",
            "phase-a",
            BoundaryRole::PhaseEnd,
            Some("group-a"),
        );
        let mut second = duplicated.clone();
        second.phase_id = "phase-b".to_owned();
        second.role = BoundaryRole::PhaseStart;
        let error = registry
            .seal(BoundaryCapturePlan {
                transition_id: "transition-2".to_owned(),
                commands: vec![SourceBoundarySnapshotCommand {
                    source_id: "node-a".to_owned(),
                    coalescing_group_id: Some("group-a".to_owned()),
                    subscribers: vec![duplicated, second],
                    absolute_deadline_ns: 100,
                }],
            })
            .unwrap_err();
        assert!(matches!(
            error,
            BoundaryPlanError::DuplicateReference { .. }
        ));
    }

    #[test]
    fn empty_and_duplicate_prepared_source_sets_fail_closed() {
        assert_eq!(
            BoundaryPlanRegistry::new(Vec::<String>::new()),
            Err(BoundaryPlanError::NoExpectedSources)
        );
        assert_eq!(
            BoundaryPlanRegistry::new(["node-a", "node-a"]),
            Err(BoundaryPlanError::DuplicateExpectedSource(
                "node-a".to_owned()
            ))
        );
    }
}
