// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Deterministic SimClock-driven replay executor.
//!
//! Unlike the pure `replay_schedule`/`dispatch_ordered_records` computation, this
//! actually *runs* the dispatch under the runtime's virtual [`SimClock`]: each
//! scheduled turn sleeps on the clock until its dispatch instant and then emits
//! its export record, and the virtual driver ([`drive_sim`]) fast-forwards the
//! clock through the parked deadlines in order. The result is the export records
//! in true dispatch order, each stamped with the virtual-clock instant it fired
//! — an actual clock-driven execution (SimClock, not the full online
//! engine/transport) producing execution-order raw byte-exact timing + content.

use std::cell::RefCell;
use std::rc::Rc;

use serde_json::Value;

use crate::agentx::export::raw_export_records;
use crate::agentx::loader::ReconstructedConversation;
use crate::agentx::trajectory_source::{
    profiling_dispatch_delays_ms, replay_schedule, ReplayPhase,
};
use crate::clock::clock::Clock;
use crate::clock::sim_clock::SimClock;
use crate::graph::runtime::drive_sim;

const NS_PER_MS: i64 = 1_000_000;

/// Run `dispatches` (each `(dispatch_ns, record)`) under a fresh [`SimClock`]:
/// each record's future sleeps until `dispatch_ns` then fires. Returns the
/// records in the order they actually fired, each paired with the virtual-clock
/// instant (ns) it fired at.
pub fn execute_replay_sim(dispatches: Vec<(i64, Value)>) -> Vec<(i64, Value)> {
    let clock = Rc::new(SimClock::new());
    let recorded: Rc<RefCell<Vec<(i64, usize, Value)>>> = Rc::new(RefCell::new(Vec::new()));

    let clock_body = clock.clone();
    let recorded_body = recorded.clone();
    drive_sim(clock.clone(), move |_handle| async move {
        let futs = dispatches.into_iter().enumerate().map(|(i, (ns, rec))| {
            let c = clock_body.clone();
            let r = recorded_body.clone();
            async move {
                // Sleep on the virtual clock until this turn's dispatch instant.
                Clock::sleep(c.clone(), ns.max(0)).await;
                r.borrow_mut().push((c.now_ns(), i, rec));
            }
        });
        futures::future::join_all(futs).await;
    });

    let mut out = Rc::try_unwrap(recorded)
        .expect("sole owner after drive_sim")
        .into_inner();
    // Fire order is deadline order; break exact ties by original index.
    out.sort_by_key(|(ns, i, _)| (*ns, *i));
    out.into_iter().map(|(ns, _, rec)| (ns, rec)).collect()
}

/// Execute one conversation's agentic replay under the SimClock at `t_star_ms`:
/// warmup turn(s) fire at t=0, profiling turns fire at their phase-start-anchored
/// delay (`profiling_dispatch_delays_ms`, honoring `burst` / `cap_ms`), history
/// turns are omitted. Returns `(dispatch_ns, export_record)` in true fired order.
pub fn run_replay_sim(
    conv: &ReconstructedConversation,
    t_star_ms: f64,
    burst: bool,
    cap_ms: Option<f64>,
) -> Vec<(i64, Value)> {
    let ts: Vec<Option<f64>> = conv.turns.iter().map(|t| t.timestamp_ms).collect();
    let schedule = replay_schedule(&ts, t_star_ms);
    let records = raw_export_records(conv, Some(t_star_ms));

    // Anchor the profiling turns' offsets across the lane.
    let prof_offsets: Vec<f64> = schedule
        .iter()
        .filter(|st| st.phase == ReplayPhase::Profiling)
        .map(|st| st.offset_ms.unwrap_or(0.0))
        .collect();
    let prof_delays = profiling_dispatch_delays_ms(&prof_offsets, burst, cap_ms);

    let mut dispatches: Vec<(i64, Value)> = Vec::new();
    let mut prof_i = 0usize;
    for (k, st) in schedule.iter().enumerate() {
        match st.phase {
            ReplayPhase::Warmup => dispatches.push((0, records[k].clone())),
            ReplayPhase::Profiling => {
                let delay_ms = prof_delays[prof_i];
                prof_i += 1;
                dispatches.push(((delay_ms * NS_PER_MS as f64) as i64, records[k].clone()));
            }
            ReplayPhase::History => {}
        }
    }
    execute_replay_sim(dispatches)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agentx::loader::{ReconstructedConversation, ReconstructedTurn};
    use crate::agentx::synth::ChatMessage;

    fn turn(ts: f64, outer: i64) -> ReconstructedTurn {
        ReconstructedTurn {
            timestamp_ms: Some(ts),
            delay_ms: None,
            api_time_ms: None,
            source_trace_id: "t".into(),
            source_outer_idx: outer,
            source_kind: "weka_main".into(),
            model: "m".into(),
            max_tokens: 4,
            raw_messages: vec![ChatMessage::plain("user", format!("turn{outer}"))],
            reset_context: false,
            theoretical_prefix_cache_hit_blocks: 0,
            theoretical_prefix_cache_total_blocks: 1,
            input_kind: None,
        }
    }

    #[test]
    fn execute_orders_by_virtual_clock() {
        let d = vec![
            (300 * NS_PER_MS, serde_json::json!({"id": "c"})),
            (50 * NS_PER_MS, serde_json::json!({"id": "a"})),
            (200 * NS_PER_MS, serde_json::json!({"id": "b"})),
        ];
        let out = execute_replay_sim(d);
        let order: Vec<&str> = out.iter().map(|(_, r)| r["id"].as_str().unwrap()).collect();
        assert_eq!(order, vec!["a", "b", "c"]);
        // Stamped with the virtual instants they fired at.
        assert_eq!(out[0].0, 50 * NS_PER_MS);
        assert_eq!(out[2].0, 300 * NS_PER_MS);
    }

    #[test]
    fn run_replay_fires_warmup_then_profiling_under_simclock() {
        let conv = ReconstructedConversation {
            session_id: "t".into(),
            replay_scope_id: "t".into(),
            parent_conversation_id: None,
            turns: vec![turn(0.0, 0), turn(500.0, 1), turn(300.0, 2), turn(900.0, 3)],
        };
        // t*=250: warmup turn0 (fires @0), profiling turn2(off 50), turn1(250), turn3(650).
        let fired = run_replay_sim(&conv, 250.0, false, None);
        let outers: Vec<i64> = fired
            .iter()
            .map(|(_, r)| r["source_outer_idx"].as_i64().unwrap())
            .collect();
        assert_eq!(outers, vec![0, 2, 1, 3]);
        // Virtual fire instants (ns): 0, 50ms, 250ms, 650ms.
        assert_eq!(fired[0].0, 0);
        assert_eq!(fired[1].0, 50 * NS_PER_MS);
        assert_eq!(fired[2].0, 250 * NS_PER_MS);
        assert_eq!(fired[3].0, 650 * NS_PER_MS);
        // Content survives the run.
        assert_eq!(fired[1].1["raw_messages"][0]["content"], serde_json::json!("turn2"));
    }
}
