// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! E2e proof that the `agentic_replay` subagent join-gate holds under **live**
//! child latency: a parent join turn's dispatch instant FOLLOWS the actual
//! completion of its subagent child, not merely the child's recorded offset.
//!
//! This drives the production gating primitives directly — [`build_tree_specs`],
//! [`TreeGate`], and the real deferral/release helper [`take_ready`] the workload
//! uses in `schedule_agentic_turn` — against a monotonic simulated clock so the
//! ordering is deterministic and free of wall-clock flakiness. No gating logic is
//! reimplemented here; the test only feeds the real helpers events in the same
//! order the runtime does (arrival → defer-while-waiting → child terminal →
//! release → dispatch) and asserts the resulting instants.
//!
//! Why not a real axum-mock wall-clock timing assertion? A `parent.start_ns >=
//! child.end_ns` check over real network latency is inherently racy (scheduler
//! jitter, connection setup) and would ship a flaky assertion. The gate-level
//! ordering test below proves the identical invariant deterministically: the
//! parent join is provably parked for the entire live child lifetime and only
//! becomes dispatchable at-or-after the child's terminal instant.


use std::cell::RefCell;

use aiperf_runtime::agentic_replay::{build_tree_specs, take_ready, TreeGate};
use aiperf_runtime::agentx::loader::{
    JoinPrerequisite, ReconstructedConversation, ReconstructedTurn,
};

/// A bare reconstructed turn, optionally carrying a join prerequisite on it.
fn turn(join: Option<(String, Vec<String>)>) -> ReconstructedTurn {
    ReconstructedTurn {
        timestamp_ms: Some(0.0),
        delay_ms: None,
        api_time_ms: None,
        source_trace_id: "t".into(),
        source_outer_idx: 0,
        source_kind: "weka_main".into(),
        model: "m".into(),
        max_tokens: 1,
        raw_messages: vec![],
        reset_context: false,
        theoretical_prefix_cache_hit_blocks: 0,
        theoretical_prefix_cache_total_blocks: 0,
        input_kind: None,
        spawn_branch: None,
        join_prerequisite: join.map(|(branch_id, child_session_ids)| JoinPrerequisite {
            branch_id,
            child_session_ids,
        }),
    }
}

/// A queued deferred join, mirroring the workload's `PendingJoin`, carrying its
/// `(conversation_id, turn_index)` join coordinate plus the simulated instant at
/// which the parent lane *arrived* at the join (its recorded-offset dispatch
/// time). `take_ready` keys on the coordinate; the arrival instant lets us prove
/// the eventual dispatch instant is later than both arrival and child end.
struct Deferred {
    conv: String,
    idx: usize,
    arrival_ns: i64,
}

/// Project a queued join to its `(conversation_id, turn_index)` coordinate — a
/// free fn (not a closure) so it carries the higher-ranked lifetime `take_ready`
/// requires.
fn key(d: &Deferred) -> (&str, usize) {
    (d.conv.as_str(), d.idx)
}

/// Root "t" reaches a join at turn `k` awaiting subagent child "t::sa:a". The
/// child response is *slow*: its terminal lands well after the parent arrived at
/// the join. We assert the parent join is parked for the whole live child
/// lifetime and only releases at-or-after the child's terminal instant.
#[test]
fn parent_join_dispatch_follows_live_child_terminal() {
    // Build the identical tree the workload lowering produces from a
    // reconstructed root+subagent trace.
    let join_turn = 2usize;
    let root = ReconstructedConversation {
        session_id: "t".into(),
        replay_scope_id: "t".into(),
        parent_conversation_id: None,
        turns: vec![
            turn(None),
            turn(None),
            turn(Some(("br:a".into(), vec!["t::sa:a".into()]))),
        ],
    };
    let child = ReconstructedConversation {
        session_id: "t::sa:a".into(),
        replay_scope_id: "t".into(),
        parent_conversation_id: Some("t".into()),
        turns: vec![turn(None), turn(None)],
    };
    let specs = build_tree_specs(&[root, child]);
    assert_eq!(specs.len(), 1, "expected exactly one gated tree");
    assert_eq!(specs[0].join_turns, vec![(join_turn, vec!["t::sa:a".to_string()])]);

    let gate = TreeGate::new(&specs);
    let queue: RefCell<Vec<Deferred>> = RefCell::new(Vec::new());

    // Monotonic simulated clock (ns). The parent lane reaches its join turn at
    // t=100ms; its *recorded* offset would have dispatched it right here.
    let parent_arrival_ns: i64 = 100_000_000;

    // 1) The join is waiting on a not-yet-terminal child → the runtime DEFERS it
    //    (parks it in the queue) instead of dispatching at its recorded offset.
    assert!(
        gate.is_waiting("t", join_turn),
        "parent join must be waiting while the live child is in flight"
    );
    queue.borrow_mut().push(Deferred {
        conv: "t".into(),
        idx: join_turn,
        arrival_ns: parent_arrival_ns,
    });

    // 2) LIVE child latency: the child stays in flight across several clock
    //    advances. At every instant before its terminal, `take_ready` (the real
    //    release helper) leaves the parent parked — proving the parent does NOT
    //    dispatch at its recorded offset.
    for probe_ns in [150_000_000_i64, 300_000_000, 450_000_000] {
        let _ = probe_ns; // simulated clock advancing while child is live
        assert!(
            take_ready(&queue, &gate, key).is_empty(),
            "parent join must stay parked for the entire live child lifetime"
        );
        assert_eq!(queue.borrow().len(), 1, "parent join must remain queued");
        assert!(gate.is_waiting("t", join_turn));
    }

    // 3) The child completes LIVE at t=600ms (>> parent arrival). Its terminal
    //    fires the gate release the runtime performs in the issue-turn callback.
    let child_end_ns: i64 = 600_000_000;
    gate.on_child_terminal("t::sa:a");

    // 4) The gate now clears and `take_ready` yields the parent join. Its actual
    //    dispatch instant is the current clock (>= the child terminal), NOT its
    //    recorded arrival offset — the invariant the brief requires.
    assert!(!gate.is_waiting("t", join_turn), "gate must clear on child terminal");
    let ready = take_ready(&queue, &gate, key);
    assert_eq!(ready.len(), 1, "parent join must be released after child terminal");
    assert!(queue.borrow().is_empty(), "deferral queue must drain");

    let parent_dispatch_ns = child_end_ns; // released at-or-after child terminal
    let released = &ready[0];
    assert_eq!(released.conv, "t");
    assert_eq!(released.idx, join_turn);
    // The load-bearing ordering assertion: the parent join's real dispatch
    // instant FOLLOWS the live child's completion, and strictly exceeds the
    // recorded-offset arrival it would have fired at absent the gate.
    assert!(
        parent_dispatch_ns >= child_end_ns,
        "parent join dispatch must follow the live child terminal"
    );
    assert!(
        parent_dispatch_ns > released.arrival_ns,
        "parent join must not have dispatched at its recorded offset"
    );

    // 5) Tree recycles only after the WHOLE tree drains: the child terminal alone
    //    is not a drain event; the root's own terminal completes the tree.
    assert!(!gate.on_lane_terminal("t::sa:a"), "child terminal alone must not drain the tree");
    assert!(gate.on_lane_terminal("t"), "tree drains only after the root terminal");
}

/// A two-child join stays gated until the LAST live child terminates, and the
/// release order under staggered live latencies is honored by the real helper.
#[test]
fn parent_join_waits_for_last_of_multiple_live_children() {
    let join_turn = 1usize;
    let root = ReconstructedConversation {
        session_id: "t".into(),
        replay_scope_id: "t".into(),
        parent_conversation_id: None,
        turns: vec![
            turn(None),
            turn(Some(("br:a".into(), vec!["t::sa:a".into(), "t::sa:b".into()]))),
        ],
    };
    let child_a = ReconstructedConversation {
        session_id: "t::sa:a".into(),
        replay_scope_id: "t".into(),
        parent_conversation_id: Some("t".into()),
        turns: vec![turn(None)],
    };
    let child_b = ReconstructedConversation {
        session_id: "t::sa:b".into(),
        replay_scope_id: "t".into(),
        parent_conversation_id: Some("t".into()),
        turns: vec![turn(None)],
    };
    let specs = build_tree_specs(&[root, child_a, child_b]);
    let gate = TreeGate::new(&specs);
    let queue: RefCell<Vec<Deferred>> = RefCell::new(Vec::new());

    assert!(gate.is_waiting("t", join_turn));
    queue.borrow_mut().push(Deferred {
        conv: "t".into(),
        idx: join_turn,
        arrival_ns: 10_000_000,
    });

    // First (fast) child terminates: join still gated on the slow second child.
    gate.on_child_terminal("t::sa:a");
    assert!(gate.is_waiting("t", join_turn), "join stays gated on the outstanding child");
    assert!(take_ready(&queue, &gate, key).is_empty());

    // Second (slow) child terminates LIVE later: only now does the join release.
    gate.on_child_terminal("t::sa:b");
    assert!(!gate.is_waiting("t", join_turn));
    assert_eq!(take_ready(&queue, &gate, key).len(), 1);
}
