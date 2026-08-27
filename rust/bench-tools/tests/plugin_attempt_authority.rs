// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Same-process attempt and invalidation authority for plugin parity.

use aiperf_bench_tools::plugin_stats::{
    AttemptDisposition, ControlledAttemptDecision, ControlledMeasurementEvaluator,
    InfrastructureEvent, MemberTerminalOutcome, PairAttemptDecision, RawMemberTerminalRecord,
    RawPairTerminalRecord, Variant, checked_in_case_plans,
};

fn planned_pair() -> (String, [Variant; 2]) {
    let evaluator = ControlledMeasurementEvaluator::new().expect("frozen authority validates");
    let planned = evaluator.pair_schedule()[0].clone();
    (planned.pair_id, planned.member_order)
}

fn terminal_pair(outcomes: [MemberTerminalOutcome; 2]) -> RawPairTerminalRecord {
    let (pair_id, member_order) = planned_pair();
    RawPairTerminalRecord {
        scenario: "http_non_streaming_c1".to_owned(),
        pair_id,
        member_order,
        members: member_order
            .into_iter()
            .zip(outcomes)
            .map(|(variant, outcome)| RawMemberTerminalRecord {
                variant,
                outcome,
                samples: Vec::new(),
                terminal_evidence_index: None,
            })
            .collect(),
        asserted_reason: Some("forged infrastructure label".to_owned()),
        asserted_disposition: Some(AttemptDisposition::InfrastructureInvalid),
    }
}

fn completed() -> MemberTerminalOutcome {
    MemberTerminalOutcome::Completed
}

fn terminal_pair_for_variant(
    variant: Variant,
    outcome: MemberTerminalOutcome,
) -> RawPairTerminalRecord {
    let (_, member_order) = planned_pair();
    let mut outcomes = [completed(), completed()];
    let index = member_order
        .iter()
        .position(|member| *member == variant)
        .expect("variant is present in every pair");
    outcomes[index] = outcome;
    terminal_pair(outcomes)
}

fn exhaust_replacements(evaluator: &mut ControlledMeasurementEvaluator) {
    for replacement in 1..=5 {
        assert!(matches!(
            evaluator
                .record_pair(terminal_pair([
                    MemberTerminalOutcome::Infrastructure(
                        InfrastructureEvent::MockServerDeathUnrelatedToMember,
                    ),
                    completed(),
                ]))
                .expect("approved infrastructure event is classified"),
            PairAttemptDecision::ReplaceWholePair {
                replacement_ordinal,
                ..
            } if replacement_ordinal == replacement
        ));
    }
    assert_eq!(
        evaluator
            .record_pair(terminal_pair([
                MemberTerminalOutcome::Infrastructure(InfrastructureEvent::HostReboot),
                completed(),
            ]))
            .expect("the cap terminates the attempt"),
        PairAttemptDecision::AttemptInvalid
    );
}

#[test]
fn forged_infrastructure_labels_cannot_replace_product_failures() {
    let product_outcomes = [
        MemberTerminalOutcome::Crash("dynamic process exited".to_owned()),
        MemberTerminalOutcome::Timeout("member deadline expired".to_owned()),
        MemberTerminalOutcome::IncompleteBudget {
            expected: 1_000,
            completed: 999,
        },
        MemberTerminalOutcome::MalformedOutput("invalid JSONL".to_owned()),
    ];
    for product_outcome in product_outcomes {
        let mut evaluator =
            ControlledMeasurementEvaluator::new().expect("frozen authority validates");
        assert_eq!(evaluator.begin_attempt().expect("attempt starts"), 1);
        assert_eq!(
            evaluator
                .record_pair(terminal_pair_for_variant(Variant::Dynamic, product_outcome,))
                .expect("raw product outcome is classified"),
            PairAttemptDecision::ExperimentFailed
        );
        assert_eq!(evaluator.history().len(), 1);
        assert_eq!(
            evaluator.history()[0].decision,
            ControlledAttemptDecision::ValidFailure
        );
        assert!(evaluator.begin_attempt().is_err());
    }
}

#[test]
fn both_member_product_error_is_an_immediate_authoritative_failure() {
    let mut evaluator = ControlledMeasurementEvaluator::new().expect("authority validates");
    evaluator.begin_attempt().expect("attempt starts");
    assert_eq!(
        evaluator
            .record_pair(terminal_pair([
                MemberTerminalOutcome::ProductError("static error".to_owned()),
                MemberTerminalOutcome::ProductError("dynamic error".to_owned()),
            ]))
            .expect("raw pair is classified"),
        PairAttemptDecision::ExperimentFailed
    );
    assert_eq!(evaluator.raw_pair_history().len(), 1);
    assert_eq!(
        evaluator.history()[0].decision,
        ControlledAttemptDecision::ValidFailure
    );
}

#[test]
fn frozen_classifier_alone_replaces_the_whole_pair_in_the_same_order() {
    let plan = checked_in_case_plans()
        .expect("inventory validates")
        .into_iter()
        .find(|case| case.scenario == "http_non_streaming_c1")
        .expect("scenario exists");
    assert!(plan.invalidation_classifier.contains("affinity_loss"));
    let mut evaluator = ControlledMeasurementEvaluator::new().expect("authority validates");
    evaluator.begin_attempt().expect("attempt starts");
    let (_, member_order) = planned_pair();
    assert_eq!(
        evaluator
            .record_pair(terminal_pair([
                MemberTerminalOutcome::Infrastructure(InfrastructureEvent::AffinityLoss),
                completed(),
            ]))
            .expect("classifier-listed event is valid infrastructure"),
        PairAttemptDecision::ReplaceWholePair {
            member_order,
            replacement_ordinal: 1,
        }
    );
    let retained = evaluator
        .raw_pair_history()
        .last()
        .expect("raw attempt is retained");
    assert_eq!(
        retained.raw.asserted_reason.as_deref(),
        Some("forged infrastructure label")
    );
    assert_eq!(retained.derived_reason, "affinity_loss");
}

#[test]
fn invalid_then_valid_failure_history_cannot_be_extended() {
    let mut evaluator = ControlledMeasurementEvaluator::new().expect("authority validates");
    evaluator.begin_attempt().expect("first attempt starts");
    exhaust_replacements(&mut evaluator);
    assert_eq!(evaluator.begin_attempt().expect("second attempt starts"), 2);
    assert_eq!(
        evaluator
            .record_pair(terminal_pair([
                completed(),
                MemberTerminalOutcome::Crash("dynamic crash".to_owned()),
            ]))
            .expect("product failure is classified"),
        PairAttemptDecision::ExperimentFailed
    );
    assert!(evaluator.begin_attempt().is_err());
    assert_eq!(
        evaluator
            .history()
            .iter()
            .map(|attempt| attempt.decision)
            .collect::<Vec<_>>(),
        vec![
            ControlledAttemptDecision::Invalid,
            ControlledAttemptDecision::ValidFailure,
        ]
    );
}

#[test]
fn three_invalid_attempts_block_a_fourth_attempt() {
    let mut evaluator = ControlledMeasurementEvaluator::new().expect("authority validates");
    for ordinal in 1..=3 {
        assert_eq!(evaluator.begin_attempt().expect("attempt starts"), ordinal);
        exhaust_replacements(&mut evaluator);
    }
    assert!(evaluator.begin_attempt().is_err());
    assert_eq!(
        evaluator
            .history()
            .iter()
            .map(|attempt| attempt.decision)
            .collect::<Vec<_>>(),
        vec![
            ControlledAttemptDecision::Invalid,
            ControlledAttemptDecision::Invalid,
            ControlledAttemptDecision::Invalid,
        ]
    );
    assert_eq!(evaluator.raw_pair_history().len(), 18);
}
