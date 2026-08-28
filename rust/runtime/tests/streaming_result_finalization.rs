// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Deterministic final/aborted generation, compaction, and derived-sink
//! finalization across restart.

#![cfg(feature = "streaming")]

use aiperf_runtime::streaming::{
    checkpoint::{CheckpointTerminalReason, CommittedCheckpointGeneration},
    identity::ContentDigest,
    reliability::StreamingIssueComponentId,
    results::{
        ResultPlaneError, SinkFinalizationFailureCode, StreamingResultCompactor,
        compactor::{GenerationResultCompactor, commit_aborted_generation, retain_unsafe_abort},
        sink_status::{
            DerivedSinkRetrySupervisor, DerivedSinkStatus, DerivedSinkStatusStore,
            DerivedStatusSubstrate, DurableSinkOutputProbe, DurableSinkOutputWriter,
        },
    },
};

#[path = "support/streaming_checkpoint.rs"]
mod support;

use support::{
    committed_final_generation, component, export_budget, export_policy, latest_generation,
    open_leased, page_budget, prepared_export_persistence, report_budget, run_id,
    sink_attempt_budget, staged_abort_transaction, streaming_backend,
};

/// Run one current-thread `LocalSet` future; the finalization plane is `!Send`.
fn local<T>(future: impl Future<Output = T>) -> T {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("current-thread runtime");
    tokio::task::LocalSet::new().block_on(&runtime, future)
}

fn refusal(code: SinkFinalizationFailureCode) -> ResultPlaneError {
    ResultPlaneError::SinkFinalization { code }
}

// ── Compaction and abort ─────────────────────────────────────────────────────

#[tokio::test(flavor = "current_thread")]
async fn report_lease_releases_only_after_authoritative_report_commit() {
    let backend = streaming_backend();
    let run = run_id(1);
    let committed = committed_final_generation(&backend, run).await;
    let budget = report_budget();
    let compactor = GenerationResultCompactor::new(run, page_budget(4), budget.clone());

    let prepared = compactor
        .compact(open_leased(&backend, run, &committed).await)
        .await
        .expect("compact committed generation");

    // The lease is retained by the prepared report, not by compaction, so the
    // charge is still outstanding while the report is uncommitted.
    assert!(budget.snapshot().used_bytes > 0);
    assert_eq!(budget.snapshot().used_items, 1);

    prepared
        .report_commit
        .commit()
        .expect("acknowledge authoritative report");
    assert_eq!(budget.snapshot().used_bytes, 0);
    assert_eq!(budget.snapshot().used_items, 0);
}

#[tokio::test(flavor = "current_thread")]
async fn unsafe_abort_preserves_last_partial_without_fabricating_terminal_root() {
    let backend = streaming_backend();
    let run = run_id(2);
    let committed = committed_final_generation(&backend, run).await;
    let head_before = latest_generation(&backend, run)
        .await
        .expect("committed head");

    let outcome = retain_unsafe_abort(None);
    assert!(outcome.aborted_generation.is_none());
    assert!(outcome.retained_partial.is_none());

    // No generation was published, so the authoritative head is byte-identical
    // and no terminal root exists that no barrier produced.
    let head_after = latest_generation(&backend, run)
        .await
        .expect("committed head");
    assert_eq!(head_before, head_after);
    assert_eq!(head_after, committed.generation());
}

#[tokio::test(flavor = "current_thread")]
async fn safe_abort_commits_complete_aborted_generation() {
    let backend = streaming_backend();
    let run = run_id(3);
    let committed = committed_final_generation(&backend, run).await;

    let (transaction, metadata) = staged_abort_transaction(&backend, run, &committed).await;
    let aborted = commit_aborted_generation(transaction, metadata)
        .await
        .expect("commit aborted generation");

    assert!(aborted.is_final());
    assert_eq!(
        aborted.terminal_reason(),
        Some(CheckpointTerminalReason::Aborted)
    );
    assert_eq!(
        latest_generation(&backend, run)
            .await
            .expect("committed head"),
        aborted.generation()
    );
}

#[tokio::test(flavor = "current_thread")]
async fn compaction_order_is_stable_across_page_sizes() {
    let backend = streaming_backend();
    let run = run_id(4);
    let committed = committed_final_generation(&backend, run).await;

    let mut digests = Vec::new();
    for page in [1usize, 2, 8] {
        let compactor = GenerationResultCompactor::new(run, page_budget(page), report_budget());
        let prepared = compactor
            .compact(open_leased(&backend, run, &committed).await)
            .await
            .expect("compact committed generation");
        digests.push(prepared.report_digest);
    }

    assert_eq!(digests[0], digests[1]);
    assert_eq!(digests[1], digests[2]);
}

#[tokio::test(flavor = "current_thread")]
async fn compaction_failure_retains_reconstructable_generation() {
    let backend = streaming_backend();
    let run = run_id(5);
    let committed = committed_final_generation(&backend, run).await;

    // A foreign-run compactor refuses the descriptors it reads. The committed
    // generation is untouched: it is neither rewritten nor invalidated.
    let hostile = GenerationResultCompactor::new(run_id(9), page_budget(4), report_budget());
    assert!(
        hostile
            .compact(open_leased(&backend, run, &committed).await)
            .await
            .is_err()
    );

    let compactor = GenerationResultCompactor::new(run, page_budget(4), report_budget());
    let prepared = compactor
        .compact(open_leased(&backend, run, &committed).await)
        .await
        .expect("recompact the retained generation");
    assert_eq!(
        latest_generation(&backend, run)
            .await
            .expect("committed head"),
        committed.generation()
    );
    prepared.report_commit.commit().expect("acknowledge report");
}

// ── Status compare-and-swap and restart ──────────────────────────────────────

#[tokio::test(flavor = "current_thread")]
async fn crash_before_initial_status_is_found_by_generation_sink_reconciliation() {
    let backend = streaming_backend();
    let run = run_id(6);
    let committed = committed_final_generation(&backend, run).await;
    let sink = component("native_report");
    let substrate = DerivedStatusSubstrate::new();

    // No in-memory owner survived, so nothing is retained at all.
    let store = DerivedSinkStatusStore::open(run, substrate.clone(), sink_attempt_budget());
    assert_eq!(store.load(&committed, &sink).await.expect("load"), None);

    let status = store
        .reconcile_initial(&committed, &sink)
        .await
        .expect("reconcile initial status");
    assert_eq!(
        status,
        DerivedSinkStatus::PendingAttempt { next_ordinal: 0 }
    );

    drop(store);
    let reopened = DerivedSinkStatusStore::open(run, substrate, sink_attempt_budget());
    assert_eq!(
        reopened.load(&committed, &sink).await.expect("load"),
        Some(DerivedSinkStatus::PendingAttempt { next_ordinal: 0 })
    );
}

#[tokio::test(flavor = "current_thread")]
async fn crash_or_cancellation_before_receipt_status_cas_reuses_exact_ordinal() {
    let backend = streaming_backend();
    let run = run_id(7);
    let committed = committed_final_generation(&backend, run).await;
    let sink = component("native_report");
    let substrate = DerivedStatusSubstrate::new();
    let store = DerivedSinkStatusStore::open(run, substrate.clone(), sink_attempt_budget());
    store
        .reconcile_initial(&committed, &sink)
        .await
        .expect("initial status");

    let token = store
        .open_attempt(&committed, &sink)
        .await
        .expect("admit first attempt");
    assert_eq!(token.ordinal(), 0);
    let writes_before = substrate.write_count();
    drop(token);

    // Admitting an attempt writes nothing durable, so a replacement process
    // reuses the same dense ordinal rather than burning one.
    assert_eq!(substrate.write_count(), writes_before);
    drop(store);
    let reopened = DerivedSinkStatusStore::open(run, substrate, sink_attempt_budget());
    assert_eq!(
        reopened
            .open_attempt(&committed, &sink)
            .await
            .expect("readmit attempt")
            .ordinal(),
        0
    );
}

#[tokio::test(flavor = "current_thread")]
async fn crash_or_cancellation_after_receipt_status_cas_reopens_exact_pending_status() {
    let backend = streaming_backend();
    let run = run_id(8);
    let committed = committed_final_generation(&backend, run).await;
    let sink = component("native_report");
    let substrate = DerivedStatusSubstrate::new();
    let store = DerivedSinkStatusStore::open(run, substrate.clone(), sink_attempt_budget());
    store
        .reconcile_initial(&committed, &sink)
        .await
        .expect("initial status");

    let budget = export_budget();
    let persistence = prepared_export_persistence(run, &committed, &sink, 0, &budget).await;
    let status = store
        .commit_retry(&committed, &sink, &persistence)
        .await
        .expect("commit retry");
    assert_eq!(
        status,
        DerivedSinkStatus::PendingRetry {
            last_ordinal: 0,
            counter_before: 0,
        }
    );

    drop(persistence);
    drop(store);
    let reopened = DerivedSinkStatusStore::open(run, substrate, sink_attempt_budget());
    assert_eq!(
        reopened.load(&committed, &sink).await.expect("load"),
        Some(DerivedSinkStatus::PendingRetry {
            last_ordinal: 0,
            counter_before: 0,
        })
    );
    assert_eq!(
        reopened
            .open_attempt(&committed, &sink)
            .await
            .expect("next attempt")
            .ordinal(),
        1
    );
}

#[tokio::test(flavor = "current_thread")]
async fn durable_output_before_complete_cas_recovers_complete() {
    let backend = streaming_backend();
    let run = run_id(9);
    let committed = committed_final_generation(&backend, run).await;
    let sink = component("native_report");
    let substrate = DerivedStatusSubstrate::new();
    let store = DerivedSinkStatusStore::open(run, substrate.clone(), sink_attempt_budget());
    store
        .reconcile_initial(&committed, &sink)
        .await
        .expect("initial status");

    let token = store
        .open_attempt(&committed, &sink)
        .await
        .expect("admit attempt");
    let writer = DurableSinkOutputWriter::new(&committed, sink.clone(), substrate.clone())
        .expect("bind writer");
    let proof = writer
        .write(token, b"authoritative-report")
        .await
        .expect("durable write");
    let expected_digest = proof.output_digest();

    // The process dies between the durable output and the completing swap.
    drop(proof);
    drop(writer);
    drop(store);

    let reopened = DerivedSinkStatusStore::open(run, substrate.clone(), sink_attempt_budget());
    assert_eq!(
        reopened.load(&committed, &sink).await.expect("load"),
        Some(DerivedSinkStatus::PendingAttempt { next_ordinal: 0 })
    );
    let probe =
        DurableSinkOutputProbe::new(&committed, sink.clone(), substrate).expect("bind probe");
    let recovered = probe
        .probe()
        .await
        .expect("probe")
        .expect("durable output survives");
    assert_eq!(recovered.output_digest(), expected_digest);
    assert_eq!(
        reopened
            .commit_complete(&committed, recovered)
            .await
            .expect("complete swap"),
        DerivedSinkStatus::Complete {
            output_digest: expected_digest,
            output_length: b"authoritative-report".len() as u64,
        }
    );
}

#[tokio::test(flavor = "current_thread")]
async fn reopen_rejects_tampered_or_unreachable_export_receipt() {
    let backend = streaming_backend();
    let run = run_id(10);
    let committed = committed_final_generation(&backend, run).await;
    let sink = component("native_report");
    let substrate = DerivedStatusSubstrate::new();
    let store = DerivedSinkStatusStore::open(run, substrate.clone(), sink_attempt_budget());
    store
        .reconcile_initial(&committed, &sink)
        .await
        .expect("initial status");
    let budget = export_budget();
    let persistence = prepared_export_persistence(run, &committed, &sink, 0, &budget).await;
    store
        .commit_retry(&committed, &sink, &persistence)
        .await
        .expect("commit retry");
    let intact = persistence.encoded_bytes().to_vec();
    drop(persistence);

    let policy = export_policy();
    let mut tampered = intact.clone();
    let last = tampered.len() - 1;
    tampered[last] ^= 0xff;
    substrate.overwrite_encoded_receipt(&committed.generation(), &sink, Some(tampered));
    assert_eq!(
        store
            .reopen_receipt(
                &committed,
                &sink,
                &policy,
                &export_budget(),
                &export_budget()
            )
            .await
            .expect_err("tampered receipt refused"),
        refusal(SinkFinalizationFailureCode::TamperedReceipt)
    );

    substrate.overwrite_encoded_receipt(&committed.generation(), &sink, None);
    assert_eq!(
        store
            .reopen_receipt(
                &committed,
                &sink,
                &policy,
                &export_budget(),
                &export_budget()
            )
            .await
            .expect_err("unreachable receipt refused"),
        refusal(SinkFinalizationFailureCode::MissingReceipt)
    );
}

#[tokio::test(flavor = "current_thread")]
async fn receipt_attempt_or_issue_mismatch_refuses_before_store_io() {
    let backend = streaming_backend();
    let run = run_id(11);
    let committed = committed_final_generation(&backend, run).await;
    let sink = component("native_report");
    let substrate = DerivedStatusSubstrate::new();
    let store = DerivedSinkStatusStore::open(run, substrate.clone(), sink_attempt_budget());
    store
        .reconcile_initial(&committed, &sink)
        .await
        .expect("initial status");

    let budget = export_budget();
    // The status expects ordinal 0; a receipt authored at ordinal 2 is not the
    // dense successor.
    let persistence = prepared_export_persistence(run, &committed, &sink, 2, &budget).await;
    let writes_before = substrate.write_count();
    assert_eq!(
        store
            .commit_retry(&committed, &sink, &persistence)
            .await
            .expect_err("non-dense ordinal refused"),
        refusal(SinkFinalizationFailureCode::OrdinalMismatch)
    );
    assert_eq!(substrate.write_count(), writes_before);
    assert_eq!(
        store.load(&committed, &sink).await.expect("load"),
        Some(DerivedSinkStatus::PendingAttempt { next_ordinal: 0 })
    );
}

#[tokio::test(flavor = "current_thread")]
async fn illegal_sink_transition_and_terminal_successor_are_unnameable() {
    let backend = streaming_backend();
    let run = run_id(12);
    let committed = committed_final_generation(&backend, run).await;
    let sink = component("native_report");
    let substrate = DerivedStatusSubstrate::new();
    let store = DerivedSinkStatusStore::open(run, substrate.clone(), sink_attempt_budget());
    store
        .reconcile_initial(&committed, &sink)
        .await
        .expect("initial status");

    let token = store
        .open_attempt(&committed, &sink)
        .await
        .expect("admit attempt");
    let writer = DurableSinkOutputWriter::new(&committed, sink.clone(), substrate.clone())
        .expect("bind writer");
    let proof = writer.write(token, b"report").await.expect("durable write");
    store
        .commit_complete(&committed, proof)
        .await
        .expect("complete swap");

    // `Complete` is terminal: it has no retry successor, no second completion,
    // and no further admitted attempt.
    let budget = export_budget();
    let persistence = prepared_export_persistence(run, &committed, &sink, 0, &budget).await;
    assert_eq!(
        store
            .commit_retry(&committed, &sink, &persistence)
            .await
            .expect_err("complete has no retry successor"),
        refusal(SinkFinalizationFailureCode::IllegalTransition)
    );
    assert_eq!(
        store
            .open_attempt(&committed, &sink)
            .await
            .err()
            .expect("complete admits no attempt"),
        refusal(SinkFinalizationFailureCode::IllegalTransition)
    );
    let probe = DurableSinkOutputProbe::new(&committed, sink, substrate).expect("bind probe");
    let replayed = probe.probe().await.expect("probe").expect("durable output");
    assert_eq!(
        store
            .commit_complete(&committed, replayed)
            .await
            .expect_err("complete has no terminal successor"),
        refusal(SinkFinalizationFailureCode::IllegalTransition)
    );
}

#[tokio::test(flavor = "current_thread")]
async fn retry_ordinal_overflow_refuses_before_store_io() {
    let backend = streaming_backend();
    let run = run_id(13);
    let committed = committed_final_generation(&backend, run).await;
    let sink = component("native_report");
    let substrate = DerivedStatusSubstrate::new();
    let store = DerivedSinkStatusStore::open(run, substrate.clone(), sink_attempt_budget());
    substrate.force_status(
        &committed.generation(),
        &sink,
        DerivedSinkStatus::PendingRetry {
            last_ordinal: u32::MAX,
            counter_before: u64::from(u32::MAX),
        },
    );

    let writes_before = substrate.write_count();
    assert_eq!(
        store
            .open_attempt(&committed, &sink)
            .await
            .expect_err("exhausted ordinal space refused"),
        refusal(SinkFinalizationFailureCode::OrdinalOverflow)
    );
    assert_eq!(substrate.write_count(), writes_before);
}

#[tokio::test(flavor = "current_thread")]
async fn reopened_status_and_receipt_retain_exact_encoded_and_parsed_charges() {
    let backend = streaming_backend();
    let run = run_id(14);
    let committed = committed_final_generation(&backend, run).await;
    let sink = component("native_report");
    let substrate = DerivedStatusSubstrate::new();
    let store = DerivedSinkStatusStore::open(run, substrate.clone(), sink_attempt_budget());
    store
        .reconcile_initial(&committed, &sink)
        .await
        .expect("initial status");
    let budget = export_budget();
    let persistence = prepared_export_persistence(run, &committed, &sink, 0, &budget).await;
    let encoded_len = persistence.encoded_bytes().len();
    store
        .commit_retry(&committed, &sink, &persistence)
        .await
        .expect("commit retry");
    drop(persistence);

    let encoded = export_budget();
    let parsed = export_budget();
    let receipt = store
        .reopen_receipt(&committed, &sink, &export_policy(), &encoded, &parsed)
        .await
        .expect("reopen receipt");

    assert_eq!(receipt.encoded_charge_bytes(), encoded_len);
    assert_eq!(
        encoded.snapshot().used_bytes,
        receipt.encoded_charge_bytes()
    );
    assert_eq!(parsed.snapshot().used_bytes, receipt.parsed_charge_bytes());
    assert_eq!(encoded.snapshot().used_items, 1);
    assert_eq!(parsed.snapshot().used_items, 1);

    drop(receipt);
    assert_eq!(encoded.snapshot().used_bytes, 0);
    assert_eq!(parsed.snapshot().used_bytes, 0);
}

#[tokio::test(flavor = "current_thread")]
async fn reporter_prepares_exactly_charged_export_failure_from_retained_receipt() {
    let backend = streaming_backend();
    let run = run_id(15);
    let committed = committed_final_generation(&backend, run).await;
    let sink = component("native_report");
    let substrate = DerivedStatusSubstrate::new();
    let store = DerivedSinkStatusStore::open(run, substrate.clone(), sink_attempt_budget());
    store
        .reconcile_initial(&committed, &sink)
        .await
        .expect("initial status");
    let first_budget = export_budget();
    let first = prepared_export_persistence(run, &committed, &sink, 0, &first_budget).await;
    store
        .commit_retry(&committed, &sink, &first)
        .await
        .expect("commit first retry");
    drop(first);

    let status = store
        .reopen_verified_status(&committed, &sink)
        .await
        .expect("reopen verified status");
    let next_ordinal = status
        .last_attempt_ordinal()
        .checked_add(1)
        .expect("dense successor");

    let budget = export_budget();
    let second = prepared_export_persistence(run, &committed, &sink, next_ordinal, &budget).await;
    assert_eq!(second.attempt_ordinal(), 1);
    assert_eq!(second.counter_before(), 1);
    // The export budget carries the exact encoded and parsed charges and nothing
    // more: two items, and bytes equal to the encoded document plus its parse.
    assert_eq!(budget.snapshot().used_items, 2);
    assert_eq!(
        budget.snapshot().high_water_bytes,
        budget.snapshot().used_bytes
    );
    store
        .commit_retry(&committed, &sink, &second)
        .await
        .expect("commit second retry");
    drop(second);
    assert_eq!(budget.snapshot().used_items, 0);
}

#[tokio::test(flavor = "current_thread")]
async fn export_failure_consumes_into_persistence_without_reallocation_or_lease_split() {
    let backend = streaming_backend();
    let run = run_id(16);
    let committed = committed_final_generation(&backend, run).await;
    let sink = component("native_report");
    let budget = export_budget();

    let failure = support::prepared_export_failure(run, &committed, &sink, 0, &budget).await;
    let before = budget.snapshot();
    let issue_id = failure.issue_id();
    let reference = failure.receipt_reference().clone();

    let persistence = failure.into_persistence();
    let after = budget.snapshot();

    // The handoff is a move, not a copy: no new charge, no split lease, and the
    // decision and receipt facts are the identical ones the failure carried.
    assert_eq!(before.used_items, after.used_items);
    assert_eq!(before.used_bytes, after.used_bytes);
    assert_eq!(before.high_water_bytes, after.high_water_bytes);
    assert_eq!(persistence.issue_id(), issue_id);
    assert_eq!(persistence.receipt_reference(), &reference);
}

#[tokio::test(flavor = "current_thread")]
async fn status_store_persists_encoded_export_receipt_while_intact_owner_is_live() {
    let backend = streaming_backend();
    let run = run_id(17);
    let committed = committed_final_generation(&backend, run).await;
    let sink = component("native_report");
    let substrate = DerivedStatusSubstrate::new();
    let store = DerivedSinkStatusStore::open(run, substrate.clone(), sink_attempt_budget());
    store
        .reconcile_initial(&committed, &sink)
        .await
        .expect("initial status");

    let budget = export_budget();
    let persistence = prepared_export_persistence(run, &committed, &sink, 0, &budget).await;
    store
        .commit_retry(&committed, &sink, &persistence)
        .await
        .expect("commit retry");

    // The store borrowed the handoff, so the intact owner still holds its exact
    // encoded and parsed leases after the durable write.
    assert_eq!(budget.snapshot().used_items, 2);
    let encoded = persistence.encoded_bytes().to_vec();
    drop(persistence);
    assert_eq!(budget.snapshot().used_items, 0);

    let reopened = DerivedSinkStatusStore::open(run, substrate, sink_attempt_budget());
    let receipt = reopened
        .reopen_receipt(
            &committed,
            &sink,
            &export_policy(),
            &export_budget(),
            &export_budget(),
        )
        .await
        .expect("reopen persisted receipt");
    assert_eq!(receipt.encoded_charge_bytes(), encoded.len());
}

#[tokio::test(flavor = "current_thread")]
async fn reporter_rejects_foreign_run_generation_sink_or_ordinal() {
    let backend = streaming_backend();
    let run = run_id(18);
    let foreign_run = run_id(19);
    let committed = committed_final_generation(&backend, run).await;
    let foreign_committed = committed_final_generation(&backend, foreign_run).await;
    let generation = committed.generation();
    let foreign_generation = foreign_committed.generation();
    let sink = component("native_report");
    let other_sink = component("other_sink");
    let budget = export_budget();

    // Foreign run: the call names a run the reporter does not own.
    assert!(
        support::try_prepared_export_failure_for(
            run,
            run,
            &generation,
            &sink,
            0,
            foreign_run,
            &generation,
            &sink,
            0,
            &budget,
        )
        .await
        .is_err()
    );
    // Foreign generation: the issue and the call name different generations.
    assert!(
        support::try_prepared_export_failure_for(
            run,
            run,
            &generation,
            &sink,
            0,
            run,
            &foreign_generation,
            &sink,
            0,
            &budget,
        )
        .await
        .is_err()
    );
    // Foreign sink: the issue and the call name different derived sinks.
    assert!(
        support::try_prepared_export_failure_for(
            run,
            run,
            &generation,
            &sink,
            0,
            run,
            &generation,
            &other_sink,
            0,
            &budget,
        )
        .await
        .is_err()
    );
    // Foreign ordinal: the issue and the call name different dense ordinals.
    assert!(
        support::try_prepared_export_failure_for(
            run,
            run,
            &generation,
            &sink,
            0,
            run,
            &generation,
            &sink,
            1,
            &budget,
        )
        .await
        .is_err()
    );

    // The status owner applies the same density rule to a well-formed receipt
    // whose ordinal is not the dense successor of the retained status.
    let substrate = DerivedStatusSubstrate::new();
    let store = DerivedSinkStatusStore::open(run, substrate, sink_attempt_budget());
    store
        .reconcile_initial(&committed, &sink)
        .await
        .expect("initial status");
    let persistence = prepared_export_persistence(run, &committed, &sink, 3, &budget).await;
    assert_eq!(
        store
            .commit_retry(&committed, &sink, &persistence)
            .await
            .expect_err("foreign ordinal refused"),
        refusal(SinkFinalizationFailureCode::OrdinalMismatch)
    );
}

#[tokio::test(flavor = "current_thread")]
async fn durable_writer_and_probe_are_the_only_output_proof_minting_paths() {
    let backend = streaming_backend();
    let run = run_id(20);
    let committed = committed_final_generation(&backend, run).await;
    let sink = component("native_report");
    let substrate = DerivedStatusSubstrate::new();
    let store = DerivedSinkStatusStore::open(run, substrate.clone(), sink_attempt_budget());
    store
        .reconcile_initial(&committed, &sink)
        .await
        .expect("initial status");

    // Before any durable write the probe mints nothing: a proof cannot be
    // conjured from a status that names no output.
    let probe = DurableSinkOutputProbe::new(&committed, sink.clone(), substrate.clone())
        .expect("bind probe");
    assert!(probe.probe().await.expect("probe").is_none());

    let token = store
        .open_attempt(&committed, &sink)
        .await
        .expect("admit attempt");
    let writer = DurableSinkOutputWriter::new(&committed, sink, substrate).expect("bind writer");
    let written = writer.write(token, b"report").await.expect("durable write");
    let probed = probe.probe().await.expect("probe").expect("durable output");

    assert_eq!(written.output_digest(), probed.output_digest());
    assert_eq!(written.output_length(), probed.output_length());
    assert_eq!(
        written.output_digest(),
        ContentDigest::from_bytes(*blake3::hash(b"report").as_bytes())
    );
}

#[tokio::test(flavor = "current_thread")]
async fn unbudgeted_or_forged_export_tokens_are_unnameable() {
    let backend = streaming_backend();
    let run = run_id(21);
    let committed = committed_final_generation(&backend, run).await;
    let sink = component("native_report");
    let substrate = DerivedStatusSubstrate::new();

    // A store whose attempt budget cannot admit one token cannot mint one, and
    // the token type has no reachable constructor, so no forgery substitutes.
    let store =
        DerivedSinkStatusStore::open(run, substrate.clone(), support::exhausted_attempt_budget());
    store
        .reconcile_initial(&committed, &sink)
        .await
        .expect("initial status");
    assert_eq!(
        store
            .open_attempt(&committed, &sink)
            .await
            .expect_err("unbudgeted attempt refused"),
        refusal(SinkFinalizationFailureCode::Budget)
    );

    let funded = DerivedSinkStatusStore::open(run, substrate, sink_attempt_budget());
    let token = funded
        .open_attempt(&committed, &sink)
        .await
        .expect("budgeted attempt admitted");
    assert!(token.charged_bytes() > 0);
}

#[tokio::test(flavor = "current_thread")]
async fn post_final_restart_reopens_pending_from_generation_and_derived_store_without_issue_ledger()
{
    let backend = streaming_backend();
    let run = run_id(22);
    let committed = committed_final_generation(&backend, run).await;
    let sink = component("native_report");
    let substrate = DerivedStatusSubstrate::new();

    {
        let store = DerivedSinkStatusStore::open(run, substrate.clone(), sink_attempt_budget());
        store
            .reconcile_initial(&committed, &sink)
            .await
            .expect("initial status");
        let budget = export_budget();
        let persistence = prepared_export_persistence(run, &committed, &sink, 0, &budget).await;
        store
            .commit_retry(&committed, &sink, &persistence)
            .await
            .expect("commit retry");
        // Drop execution, the reporter that authored the receipt, and every
        // mutable ledger object before reopening.
        drop(persistence);
        drop(store);
        drop(budget);
    }

    // Reopen from the leased final generation plus a fresh derived status store
    // instance alone. No issue ledger participates.
    let reopened = DerivedSinkStatusStore::open(run, substrate, sink_attempt_budget());
    let status = reopened
        .reopen_verified_status(&committed, &sink)
        .await
        .expect("reopen verified status");
    assert_eq!(status.generation(), &committed.generation());
    assert_eq!(status.sink_id(), &sink);
    assert_eq!(status.last_attempt_ordinal(), 0);

    let receipt = reopened
        .reopen_receipt(
            &committed,
            &sink,
            &export_policy(),
            &export_budget(),
            &export_budget(),
        )
        .await
        .expect("ledger-free receipt reopen");
    assert!(receipt.parsed_charge_bytes() > 0);
}

#[tokio::test(flavor = "current_thread")]
async fn missing_or_tampered_embedded_receipt_or_reference_refuses_reopen() {
    let backend = streaming_backend();
    let run = run_id(23);
    let committed = committed_final_generation(&backend, run).await;
    let sink = component("native_report");
    let substrate = DerivedStatusSubstrate::new();
    let store = DerivedSinkStatusStore::open(run, substrate.clone(), sink_attempt_budget());
    store
        .reconcile_initial(&committed, &sink)
        .await
        .expect("initial status");
    let budget = export_budget();
    let persistence = prepared_export_persistence(run, &committed, &sink, 0, &budget).await;
    store
        .commit_retry(&committed, &sink, &persistence)
        .await
        .expect("commit retry");
    drop(persistence);

    // A status whose reference no longer names the embedded receipt refuses,
    // even though the encoded document is untouched.
    substrate.overwrite_embedded_receipt_digest(
        &committed.generation(),
        &sink,
        ContentDigest::from_bytes([0x5a; 32]),
    );
    assert_eq!(
        store
            .reopen_receipt(
                &committed,
                &sink,
                &export_policy(),
                &export_budget(),
                &export_budget()
            )
            .await
            .expect_err("tampered reference refused"),
        refusal(SinkFinalizationFailureCode::TamperedReceipt)
    );

    // A status with no reachable receipt at all refuses before any parse.
    substrate.overwrite_encoded_receipt(&committed.generation(), &sink, None);
    assert_eq!(
        store
            .reopen_receipt(
                &committed,
                &sink,
                &export_policy(),
                &export_budget(),
                &export_budget()
            )
            .await
            .expect_err("missing receipt refused"),
        refusal(SinkFinalizationFailureCode::MissingReceipt)
    );
}

#[tokio::test(flavor = "current_thread")]
async fn restart_reconstructs_exact_sink_ordinal_and_counter() {
    let backend = streaming_backend();
    let run = run_id(24);
    let committed = committed_final_generation(&backend, run).await;
    let sink = component("native_report");
    let substrate = DerivedStatusSubstrate::new();
    let store = DerivedSinkStatusStore::open(run, substrate.clone(), sink_attempt_budget());
    store
        .reconcile_initial(&committed, &sink)
        .await
        .expect("initial status");

    for ordinal in 0..2u32 {
        let budget = export_budget();
        let persistence =
            prepared_export_persistence(run, &committed, &sink, ordinal, &budget).await;
        store
            .commit_retry(&committed, &sink, &persistence)
            .await
            .expect("commit retry");
    }
    drop(store);

    let reopened = DerivedSinkStatusStore::open(run, substrate, sink_attempt_budget());
    let status = reopened
        .reopen_verified_status(&committed, &sink)
        .await
        .expect("reopen verified status");
    assert_eq!(status.last_attempt_ordinal(), 1);
    assert_eq!(status.counter_before(), 1);
    assert_eq!(
        reopened
            .open_attempt(&committed, &sink)
            .await
            .expect("next attempt")
            .ordinal(),
        2
    );
}

#[tokio::test(flavor = "current_thread")]
async fn first_attempt_exhausted_restart_uses_status_ordinal_zero_and_counter_zero() {
    let backend = streaming_backend();
    let run = run_id(25);
    let committed = committed_final_generation(&backend, run).await;
    let sink = component("native_report");
    let substrate = DerivedStatusSubstrate::new();
    let store = DerivedSinkStatusStore::open(run, substrate.clone(), sink_attempt_budget());
    store
        .reconcile_initial(&committed, &sink)
        .await
        .expect("initial status");
    let budget = export_budget();
    let persistence = prepared_export_persistence(run, &committed, &sink, 0, &budget).await;
    store
        .commit_retry(&committed, &sink, &persistence)
        .await
        .expect("commit retry");
    drop(persistence);
    drop(store);

    let reopened = DerivedSinkStatusStore::open(run, substrate, sink_attempt_budget());
    let status = reopened
        .reopen_verified_status(&committed, &sink)
        .await
        .expect("reopen verified status");
    assert_eq!(status.last_attempt_ordinal(), 0);
    assert_eq!(status.counter_before(), 0);

    let supervisor =
        DerivedSinkRetrySupervisor::new(reopened, 4.try_into().expect("nonzero page size"));
    let page = supervisor
        .pending_page(&committed, None)
        .await
        .expect("page pending sinks");
    assert_eq!(page.len(), 1);
    assert_eq!(page[0].0, sink);
    assert_eq!(
        page[0].1,
        DerivedSinkStatus::PendingRetry {
            last_ordinal: 0,
            counter_before: 0,
        }
    );
}

#[tokio::test(flavor = "current_thread")]
async fn multi_retry_exhausted_restart_uses_status_authored_last_ordinal_and_counter() {
    let backend = streaming_backend();
    let run = run_id(26);
    let committed = committed_final_generation(&backend, run).await;
    let sink = component("native_report");
    let substrate = DerivedStatusSubstrate::new();
    let store = DerivedSinkStatusStore::open(run, substrate.clone(), sink_attempt_budget());
    store
        .reconcile_initial(&committed, &sink)
        .await
        .expect("initial status");

    for ordinal in 0..3u32 {
        let budget = export_budget();
        let persistence =
            prepared_export_persistence(run, &committed, &sink, ordinal, &budget).await;
        store
            .commit_retry(&committed, &sink, &persistence)
            .await
            .expect("commit retry");
    }
    drop(store);

    // The status, not any in-memory counter, authors the reopen point.
    let reopened = DerivedSinkStatusStore::open(run, substrate, sink_attempt_budget());
    let status = reopened
        .reopen_verified_status(&committed, &sink)
        .await
        .expect("reopen verified status");
    assert_eq!(status.last_attempt_ordinal(), 2);
    assert_eq!(status.counter_before(), 2);

    let supervisor =
        DerivedSinkRetrySupervisor::new(reopened, 1.try_into().expect("nonzero page size"));
    let page = supervisor
        .pending_page(&committed, None)
        .await
        .expect("page pending sinks");
    assert_eq!(page.len(), 1);
    assert_eq!(
        page[0].1,
        DerivedSinkStatus::PendingRetry {
            last_ordinal: 2,
            counter_before: 2,
        }
    );
    assert!(
        supervisor
            .pending_page(&committed, Some(&sink))
            .await
            .expect("page after the last sink")
            .is_empty()
    );
}

/// Keep the `local` helper referenced: the `#[tokio::test]` attribute above
/// provides the current-thread runtime for every case, and this asserts the
/// same shape is reachable for a caller that drives the plane manually.
#[test]
fn manual_current_thread_driver_reaches_the_same_status_plane() {
    local(async {
        let backend = streaming_backend();
        let run = run_id(27);
        let committed: CommittedCheckpointGeneration =
            committed_final_generation(&backend, run).await;
        let sink: StreamingIssueComponentId = component("native_report");
        let store =
            DerivedSinkStatusStore::open(run, DerivedStatusSubstrate::new(), sink_attempt_budget());
        assert_eq!(
            store
                .reconcile_initial(&committed, &sink)
                .await
                .expect("initial status"),
            DerivedSinkStatus::PendingAttempt { next_ordinal: 0 }
        );
    });
}
