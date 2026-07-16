// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Absolute-timestamp trace replay workload.
//!
//! This validates and stable-sorts first-turn timestamps, resolves the schedule
//! zero, schedules every first turn up front, then selects absolute
//! `timestamp_ms`, relative `delay_ms`, or immediate dispatch (in that
//! precedence order) for each continuation. The workload intentionally ignores
//! stop bounds: the trace is the run plan.

use std::cell::RefCell;
use std::rc::Rc;

use anyhow::{Result, bail};
use async_trait::async_trait;
use uuid::Uuid;

use crate::multiturn::{ConversationSource, TurnToSend};
use crate::scheduled::{ScheduledRuntime, Workload};
use crate::scheduler::LocalTaskScheduler;

/// Fixed-schedule anchoring configuration.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FixedScheduleConfig {
    /// Slide the trace so its earliest first turn lands at run start.
    pub auto_offset_timestamps: bool,
    /// Explicit trace timestamp treated as run start when auto-offset is off.
    pub start_offset_ms: Option<f64>,
}

/// One first-turn entry in the sorted replay schedule.
#[derive(Clone, Debug)]
pub struct FixedScheduleEntry {
    /// Trace timestamp in milliseconds.
    pub timestamp_ms: f64,
    /// First turn with a freshly minted runtime correlation id.
    pub turn: TurnToSend,
}

/// Validated fixed replay plan.
#[derive(Clone, Debug)]
pub struct FixedSchedule {
    /// Timestamp subtracted from every absolute trace timestamp.
    pub schedule_zero_ms: f64,
    /// Stable timestamp-sorted first turns.
    pub entries: Vec<FixedScheduleEntry>,
}

/// Source seam that lowers conversation metadata into a fixed replay plan.
pub trait FixedScheduleSource {
    /// Validate and build a fresh plan. Every call mints new correlation ids.
    fn build_schedule(&self, conversations: &dyn ConversationSource) -> Result<FixedSchedule>;

    /// Convert a trace timestamp to an absolute clock target.
    fn timestamp_to_ns(
        &self,
        started_ns: i64,
        schedule_zero_ms: f64,
        timestamp_ms: f64,
    ) -> Result<i64>;
}

/// Dataset metadata implementation of [`FixedScheduleSource`].
pub struct DatasetFixedScheduleSource {
    config: FixedScheduleConfig,
}

impl DatasetFixedScheduleSource {
    /// Create a schedule source with explicit anchoring behavior.
    pub fn new(config: FixedScheduleConfig) -> Result<Self> {
        if config
            .start_offset_ms
            .is_some_and(|value| !value.is_finite() || value < 0.0)
        {
            bail!("fixed-schedule start offset must be finite and non-negative");
        }
        Ok(Self { config })
    }
}

impl FixedScheduleSource for DatasetFixedScheduleSource {
    fn build_schedule(&self, conversations: &dyn ConversationSource) -> Result<FixedSchedule> {
        let mut entries = Vec::new();
        for conversation in conversations.conversations() {
            if conversation.turns.is_empty() {
                continue;
            }
            let Some(timestamp_ms) = conversation.turns[0].timestamp_ms else {
                bail!(
                    "first turn of {} missing timestamp_ms",
                    conversation.conversation_id
                );
            };
            if !timestamp_ms.is_finite() {
                bail!(
                    "first turn of {} has non-finite timestamp_ms",
                    conversation.conversation_id
                );
            }
            let session = conversations
                .session_for(&conversation.conversation_id, Uuid::new_v4().to_string())?;
            entries.push(FixedScheduleEntry {
                timestamp_ms,
                turn: session.build_first_turn(None)?,
            });
        }
        if entries.is_empty() {
            bail!("no conversations with valid first-turn timestamps found");
        }

        // `sort_by` is stable: equal-timestamp conversations retain dataset
        // insertion order, matching the Clock's deterministic registration tie.
        entries.sort_by(|left, right| left.timestamp_ms.total_cmp(&right.timestamp_ms));
        let schedule_zero_ms = if self.config.auto_offset_timestamps {
            entries[0].timestamp_ms
        } else {
            self.config.start_offset_ms.unwrap_or(0.0)
        };
        Ok(FixedSchedule {
            schedule_zero_ms,
            entries,
        })
    }

    fn timestamp_to_ns(
        &self,
        started_ns: i64,
        schedule_zero_ms: f64,
        timestamp_ms: f64,
    ) -> Result<i64> {
        let offset_ns = milliseconds_to_ns(timestamp_ms - schedule_zero_ms)?;
        Ok(started_ns.saturating_add(offset_ns))
    }
}

/// Convert a finite millisecond interval to integer nanoseconds with ties-to-even
/// rounding, preserving deterministic `SimClock` behavior.
pub fn milliseconds_to_ns(milliseconds: f64) -> Result<i64> {
    if !milliseconds.is_finite() {
        bail!("timestamp interval must be finite");
    }
    let nanoseconds = milliseconds * 1_000_000.0;
    if nanoseconds < i64::MIN as f64 || nanoseconds >= i64::MAX as f64 {
        bail!("timestamp interval is outside the i64 nanosecond range");
    }
    Ok(nanoseconds.round_ties_even() as i64)
}

/// Fixed-schedule [`Workload`] over a conversation source and schedule source.
pub struct FixedScheduleWorkload {
    conversations: Rc<RefCell<Box<dyn ConversationSource>>>,
    schedule_source: Rc<dyn FixedScheduleSource>,
    schedule: FixedSchedule,
}

impl FixedScheduleWorkload {
    /// Build and validate the full first-turn schedule before run start.
    pub fn new(
        conversations: Box<dyn ConversationSource>,
        schedule_source: Rc<dyn FixedScheduleSource>,
    ) -> Result<Self> {
        let schedule = schedule_source.build_schedule(conversations.as_ref())?;
        Ok(Self {
            conversations: Rc::new(RefCell::new(conversations)),
            schedule_source,
            schedule,
        })
    }

    /// Validated schedule used by this workload.
    pub fn schedule(&self) -> &FixedSchedule {
        &self.schedule
    }
}

#[async_trait(?Send)]
impl Workload for FixedScheduleWorkload {
    fn name(&self) -> &'static str {
        "fixed_schedule"
    }

    fn has_credit_timestamps(&self) -> bool {
        false
    }

    async fn execute(&self, runtime: Rc<ScheduledRuntime>) -> Result<()> {
        // Warm the dispatch path before anchoring so the first authored request
        // is not delayed relative to its scheduled time by one-time transport
        // setup (connection, body materialization, tokenizer/JIT). This is the
        // synchronization barrier that lets issuance start flowing on-grid; the
        // warmup round-trip is discarded and never recorded.
        if let Some(entry) = self.schedule.entries.first() {
            if let Err(error) = runtime.prewarm(entry.turn.clone()).await {
                tracing::debug!(error = %error, "fixed-schedule prewarm round-trip failed");
            }
        }
        // Anchor the replay to the moment issuance actually begins, not the run
        // origin (`start_ns`) captured before per-phase dataset/schedule setup.
        // Add a small start lead so the earliest authored target sits just in
        // the future when the scheduler begins draining: every turn is scheduled
        // up front on this task, and that O(n) scheduling pass runs before the
        // scheduler can fire anything, so without the lead the first target is
        // already overdue and fires as-soon-as-possible (a few ms late relative
        // to its grid position). The lead moves the whole grid a few ms later
        // (negligible over a run) and lets every send land on the authored grid.
        // Combined with the connection prewarm above, issuance starts warm and
        // on-schedule — the Rust-native equivalent of the Python engine's
        // ZMQ "workers ready, go" barrier.
        const SCHEDULE_START_LEAD_NS: i64 = 25_000_000;
        let anchor_ns = runtime.now_ns().saturating_add(SCHEDULE_START_LEAD_NS);
        for entry in &self.schedule.entries {
            let target_ns = self.schedule_source.timestamp_to_ns(
                anchor_ns,
                self.schedule.schedule_zero_ms,
                entry.timestamp_ms,
            )?;
            schedule_fixed_turn(
                runtime.clone(),
                self.conversations.clone(),
                self.schedule_source.clone(),
                self.schedule.schedule_zero_ms,
                entry.turn.clone(),
                target_ns,
                anchor_ns,
            );
        }
        Ok(())
    }
}

fn schedule_fixed_turn(
    runtime: Rc<ScheduledRuntime>,
    conversations: Rc<RefCell<Box<dyn ConversationSource>>>,
    schedule_source: Rc<dyn FixedScheduleSource>,
    schedule_zero_ms: f64,
    turn: TurnToSend,
    target_ns: i64,
    anchor_ns: i64,
) {
    let scheduler = runtime.scheduler();
    scheduler.schedule_at_ns(
        target_ns,
        Box::pin(async move {
            let runtime_for_completion = runtime.clone();
            let conversations_for_completion = conversations.clone();
            let schedule_source_for_completion = schedule_source.clone();
            runtime.issue_turn(
                turn,
                target_ns,
                None,
                Box::new(move |credit, outcome| {
                    Box::pin(async move {
                        if credit.is_final_turn() {
                            return;
                        }

                        let (next_metadata, next_turn) = {
                            let source = conversations_for_completion.borrow();
                            let metadata = match source.next_turn_metadata(&credit) {
                                Ok(metadata) => metadata.clone(),
                                Err(error) => {
                                    tracing::warn!(
                                        error = %error,
                                        conversation_id = %credit.turn.conversation_id,
                                        "fixed-schedule continuation metadata failed"
                                    );
                                    return;
                                }
                            };
                            let next = match source.next_turn(&credit, outcome.to_turn_response()) {
                                Ok(Some(turn)) => turn,
                                Ok(None) => return,
                                Err(error) => {
                                    tracing::warn!(
                                        error = %error,
                                        conversation_id = %credit.turn.conversation_id,
                                        "fixed-schedule continuation materialization failed"
                                    );
                                    return;
                                }
                            };
                            (metadata, next)
                        };

                        let next_target = if let Some(timestamp_ms) = next_metadata.timestamp_ms {
                            match schedule_source_for_completion.timestamp_to_ns(
                                anchor_ns,
                                schedule_zero_ms,
                                timestamp_ms,
                            ) {
                                Ok(target) => target,
                                Err(error) => {
                                    tracing::warn!(error = %error, "invalid fixed timestamp");
                                    return;
                                }
                            }
                        } else if let Some(delay_ms) = next_metadata.delay_ms {
                            match milliseconds_to_ns(delay_ms) {
                                Ok(delay_ns) => outcome.end_ns.saturating_add(delay_ns),
                                Err(error) => {
                                    tracing::warn!(error = %error, "invalid fixed delay");
                                    return;
                                }
                            }
                        } else {
                            outcome.end_ns
                        };

                        schedule_fixed_turn(
                            runtime_for_completion,
                            conversations_for_completion,
                            schedule_source_for_completion,
                            schedule_zero_ms,
                            next_turn,
                            next_target,
                            anchor_ns,
                        );
                    })
                }),
            );
        }),
    );
}

#[cfg(test)]
mod tests {
    use crate::test_util::{synthetic_prepared_source, timestamped_prepared_source};

    use super::*;

    #[tokio::test]
    async fn setup_sorts_stably_and_resolves_all_zero_modes() {
        let auto = DatasetFixedScheduleSource::new(FixedScheduleConfig {
            auto_offset_timestamps: true,
            start_offset_ms: None,
        })
        .unwrap();
        let source = timestamped_prepared_source(
            &[("late", 1200.0), ("first", 1000.0), ("tie", 1000.0)],
            "m",
        )
        .await;
        let schedule = auto.build_schedule(source.as_ref()).unwrap();
        assert_eq!(schedule.schedule_zero_ms, 1000.0);
        assert_eq!(
            schedule
                .entries
                .iter()
                .map(|entry| entry.turn.conversation_id.as_str())
                .collect::<Vec<_>>(),
            vec!["first", "tie", "late"]
        );

        let manual = DatasetFixedScheduleSource::new(FixedScheduleConfig {
            auto_offset_timestamps: false,
            start_offset_ms: Some(500.0),
        })
        .unwrap();
        assert_eq!(
            manual
                .build_schedule(source.as_ref())
                .unwrap()
                .schedule_zero_ms,
            500.0
        );
        let zero = DatasetFixedScheduleSource::new(FixedScheduleConfig::default()).unwrap();
        assert_eq!(
            zero.build_schedule(source.as_ref())
                .unwrap()
                .schedule_zero_ms,
            0.0
        );
    }

    #[tokio::test]
    async fn setup_rejects_missing_first_timestamp_and_empty_dataset() {
        let source = synthetic_prepared_source(2, 2, 1, None, "m").await;
        let fixed = DatasetFixedScheduleSource::new(FixedScheduleConfig::default()).unwrap();
        assert!(
            fixed
                .build_schedule(source.as_ref())
                .unwrap_err()
                .to_string()
                .contains("missing timestamp_ms")
        );

        let empty = EmptyConversationSource;
        assert!(
            fixed
                .build_schedule(&empty)
                .unwrap_err()
                .to_string()
                .contains("no conversations")
        );
    }

    /// A source with no conversations, exercising the empty-dataset guard
    /// without the removed synthetic loader.
    struct EmptyConversationSource;

    impl ConversationSource for EmptyConversationSource {
        fn conversations(&self) -> &[crate::multiturn::ConversationMetadata] {
            &[]
        }

        fn next(
            &mut self,
            _x_correlation_id: Option<String>,
        ) -> anyhow::Result<crate::multiturn::SampledSession> {
            anyhow::bail!("empty conversation source")
        }

        fn session_for(
            &self,
            _conversation_id: &str,
            _x_correlation_id: String,
        ) -> anyhow::Result<crate::multiturn::SampledSession> {
            anyhow::bail!("empty conversation source")
        }
    }

    #[test]
    fn timestamp_conversion_handles_past_and_fractional_values() {
        let fixed = DatasetFixedScheduleSource::new(FixedScheduleConfig::default()).unwrap();
        assert_eq!(fixed.timestamp_to_ns(1_000, 10.0, 9.5).unwrap(), -499_000);
        assert_eq!(milliseconds_to_ns(0.000_001_5).unwrap(), 2);
        assert_eq!(milliseconds_to_ns(0.000_002_5).unwrap(), 2);
    }

    #[test]
    fn manual_offset_rejects_negative_or_non_finite_values() {
        for value in [-1.0, f64::NAN, f64::INFINITY] {
            assert!(
                DatasetFixedScheduleSource::new(FixedScheduleConfig {
                    auto_offset_timestamps: false,
                    start_offset_ms: Some(value),
                })
                .is_err()
            );
        }
    }
}
