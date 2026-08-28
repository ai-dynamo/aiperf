// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "streaming")]

//! Product-level coverage for the executable streaming shadow-replay plane.
//!
//! Two artifacts are exercised end to end: the frozen action inventory the
//! reliability reporter accepts as gap-closure proof, and the `shadow_replay`
//! workload's capability agreement.

use std::collections::BTreeMap;

use aiperf_runtime::{
    engine::streaming_execution::{
        SHADOW_REPLAY_WORKLOAD_DESCRIPTOR, ShadowReplaySelection, SynthesisAuthority,
        ensure_reliability_policy_agreement, ensure_single_profiling_phase,
        ensure_supported_composition,
    },
    streaming::{
        action::{ActionInventoryLedger, FrozenActionInventoryView},
        checkpoint::StreamRunIdentity,
        identity::{ContentDigest, GlobalSequence, LogicalReplayRunId},
        reliability::{
            BudgetOwnedStreamingIssueReporter, IssueSequenceUpdate, PreparedStreamingIssuePolicy,
            StreamingIssueReporter,
        },
    },
};

fn run_identity() -> StreamRunIdentity {
    StreamRunIdentity::new(LogicalReplayRunId::from_bytes([0x51; 32]))
}

fn digest(byte: u8) -> ContentDigest {
    ContentDigest::from_bytes([byte; 32])
}

fn budget() -> aiperf_runtime::streaming::budget::StreamingResourceBudget {
    aiperf_runtime::streaming::budget::StreamingResourceBudget::new(
        aiperf_runtime::streaming::budget::BudgetLimits {
            max_items: 512,
            max_bytes: 1 << 20,
        },
    )
    .expect("valid budget limits")
}

/// Author one phase through serde so wire defaults are what the test sees.
fn phase(name: &str, kind: Option<&str>) -> aiperf_runtime::engine::protocol::PhaseSpec {
    let mut authored = serde_json::json!({
        "type": "concurrency",
        "name": name,
        "exclude_from_results": false,
        "requests": 1,
        "concurrency": 1,
    });
    if let Some(kind) = kind {
        authored["kind"] = serde_json::Value::String(kind.to_owned());
    }
    serde_json::from_value(authored).expect("valid authored phase")
}

/// The amendment's named gate: a gap proof may not outrun its terminals.
#[tokio::test(flavor = "current_thread")]
async fn frozen_action_inventory_view_prepares_gap_only_after_every_terminal() {
    let run = run_identity();
    let mut inventory = ActionInventoryLedger::new(run);
    let mut reporter = BudgetOwnedStreamingIssueReporter::new(
        run,
        PreparedStreamingIssuePolicy::new([]).expect("valid empty policy"),
        budget(),
    )
    .expect("valid reporter");

    inventory
        .record_accepted(GlobalSequence::new(0))
        .expect("accept 0");
    inventory
        .record_accepted(GlobalSequence::new(1))
        .expect("accept 1");
    inventory
        .record_terminal(GlobalSequence::new(0), digest(0xA0))
        .expect("terminal 0");

    // Sequence 1 is accepted and outstanding: freezing through it must fail, so
    // the reporter is never even offered an inventory it could act on.
    assert!(
        inventory.freeze_through(GlobalSequence::new(1)).is_err(),
        "freezing past an outstanding action must be refused"
    );

    inventory
        .record_terminal(GlobalSequence::new(1), digest(0xA1))
        .expect("terminal 1");
    let frozen = inventory
        .freeze_through(GlobalSequence::new(1))
        .expect("dense freeze after every terminal");
    assert_eq!(frozen.through(), GlobalSequence::new(1));
    assert!(frozen.contains_terminal(GlobalSequence::new(1), digest(0xA1)));

    let closure = reporter
        .prepare_no_more_actions_before(&frozen, GlobalSequence::new(1))
        .expect("prepare gap closure from a dense inventory");
    reporter
        .report(IssueSequenceUpdate::CheckedNoMoreActionsBefore(closure))
        .await
        .expect("record the checked gap closure");
}

/// Freezing may not name a sequence the host never accepted.
#[test]
fn frozen_inventory_refuses_a_frontier_beyond_accepted_work() {
    let mut inventory = ActionInventoryLedger::new(run_identity());
    inventory
        .record_terminal(GlobalSequence::new(0), digest(0xB0))
        .expect("terminal 0");
    assert!(inventory.freeze_through(GlobalSequence::new(0)).is_ok());
    assert!(
        inventory.freeze_through(GlobalSequence::new(1)).is_err(),
        "a proof may not describe work the run never issued"
    );
}

/// A membership digest that was never recorded is not provable.
#[test]
fn frozen_inventory_does_not_vouch_for_an_unrecorded_membership() {
    let mut inventory = ActionInventoryLedger::new(run_identity());
    inventory
        .record_terminal(GlobalSequence::new(0), digest(0xC0))
        .expect("terminal 0");
    let frozen = inventory
        .freeze_through(GlobalSequence::new(0))
        .expect("freeze");
    assert!(frozen.contains_terminal(GlobalSequence::new(0), digest(0xC0)));
    assert!(!frozen.contains_terminal(GlobalSequence::new(0), digest(0xC1)));
}

#[test]
fn warmup_or_second_profiling_phase_is_refused_during_validation() {
    assert!(ensure_single_profiling_phase(&[phase("profiling", None)]).is_ok());
    assert!(
        ensure_single_profiling_phase(&[phase("warmup", Some("warmup")), phase("profiling", None)])
            .is_err(),
        "a streaming source cannot be replayed across a phase handoff"
    );
    assert!(
        ensure_single_profiling_phase(&[phase("a", None), phase("b", None)]).is_err(),
        "generation one runs exactly one profiling phase"
    );
}

#[test]
fn dynamo_composition_is_refused_during_capability_agreement() {
    assert!(ensure_supported_composition("local", "reference_jsonl").is_ok());
    assert!(ensure_supported_composition("dynamo", "reference_jsonl").is_err());
    assert!(ensure_supported_composition("local", "streaming_dynamo").is_err());
}

#[test]
fn pipeline_rejects_reliability_policy_digest_mismatch() {
    assert!(ensure_reliability_policy_agreement(&digest(1), &digest(1)).is_ok());
    assert!(
        ensure_reliability_policy_agreement(&digest(1), &digest(2)).is_err(),
        "no adapter or workload default may replace the agreed policy"
    );
}

#[test]
fn unregistered_selection_fails_closed_against_the_compiled_inventory() {
    let registered: std::collections::BTreeSet<String> = ["local", "reference_jsonl", "conversation"]
        .into_iter()
        .map(str::to_owned)
        .collect();
    let supported = ShadowReplaySelection {
        source: "local".to_owned(),
        format: "reference_jsonl".to_owned(),
        session_program: "conversation".to_owned(),
    };
    assert!(supported.ensure_registered(&registered).is_ok());

    let unsupported = ShadowReplaySelection {
        session_program: "not_registered".to_owned(),
        ..supported
    };
    assert!(unsupported.ensure_registered(&registered).is_err());
}

#[test]
fn resume_with_mismatched_synthesis_authority_is_refused() {
    let authored = SynthesisAuthority::Unbound;
    let restored = SynthesisAuthority::Bound {
        session_program_digest: digest(9),
    };
    assert!(authored.accept_restored(&authored).is_ok());
    assert!(
        authored.accept_restored(&restored).is_err(),
        "restored state describing a different content lineage must be refused"
    );
}

#[test]
fn shadow_replay_workload_is_registered_under_its_stable_id() {
    assert_eq!(SHADOW_REPLAY_WORKLOAD_DESCRIPTOR.id, "shadow_replay");
    let mut registry = aiperf_runtime::extensions::AIPerfRegistry::empty_or_base();
    aiperf_runtime::engine::streaming_execution::register_streaming_workloads(&mut registry)
        .expect("register the streaming workload");
    let ids: BTreeMap<&str, &str> = registry
        .workload_descriptors()
        .iter()
        .map(|descriptor| (descriptor.id, descriptor.description))
        .collect();
    assert!(
        ids.contains_key("shadow_replay"),
        "shadow_replay must be selectable by id after registration"
    );
    // Registration is transactional, so a second registration is refused.
    assert!(
        aiperf_runtime::engine::streaming_execution::register_streaming_workloads(&mut registry)
            .is_err()
    );
}
