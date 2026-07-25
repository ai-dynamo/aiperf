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

/// The complete clock-driven legacy replay: reconstruct a trace, then execute
/// each of its conversations (root + subagent/flat children) under the SimClock
/// at `t_star_ms`, returning every conversation's fired export records
/// (`session_id` → `(dispatch_ns, record)` in fired order). This is the full
/// legacy path — trace bytes → reconstruction → clock-driven dispatch → export
/// — the deterministic end-to-end run.
pub fn execute_legacy_replay<S>(
    trace_id: &str,
    trace: &crate::agentx::trace::WekaTrace,
    synth: &mut S,
    model_map: &std::collections::HashMap<String, String>,
    cfg: &crate::agentx::config::WekaConfig,
    opts: &crate::agentx::loader::MainReconstructOptions,
    t_star_ms: f64,
    burst: bool,
    cap_ms: Option<f64>,
) -> Result<Vec<(String, Vec<(i64, Value)>)>, crate::agentx::synth::PrefixTooTruncated>
where
    S: crate::agentx::synth::TokenSynth,
{
    let convs = crate::agentx::loader::convert_trace_to_conversations(
        trace_id, trace, synth, model_map, cfg, opts,
    )?;
    Ok(convs
        .iter()
        .map(|c| (c.session_id.clone(), run_replay_sim(c, t_star_ms, burst, cap_ms)))
        .collect())
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

    #[test]
    fn full_legacy_replay_from_trace_bytes_under_simclock() {
        use crate::agentx::config::WekaConfig;
        use crate::agentx::loader::MainReconstructOptions;
        use crate::agentx::synth::TokenSynth;
        use crate::agentx::trace::{HashIdScope, WekaNormalRequest, WekaRequest, WekaTrace};
        struct Stub;
        impl TokenSynth for Stub {
            fn decode_block_tokens(&mut self, h: &[i64]) -> Vec<u32> {
                h.iter().flat_map(|&x| (0..4).map(move |i| x as u32 * 1000 + i)).collect()
            }
            fn sample_partial_tail_tokens(&mut self, n: usize, _s: &str) -> Vec<u32> {
                (0..n as u32).map(|i| 900_000 + i).collect()
            }
            fn decode_tokens_to_text(&self, t: &[u32]) -> String {
                t.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" ")
            }
        }
        let norm = |t: f64, hs: &[i64]| {
            WekaRequest::Normal(WekaNormalRequest {
                t,
                model: "m".into(),
                input_length: hs.len() as i64 * 4,
                output_length: 4,
                hash_ids: hs.to_vec(),
                input_types: vec![],
                output_types: vec![],
                stop: String::new(),
                api_time: Some(0.1),
                think_time: None,
            })
        };
        let trace = WekaTrace {
            id: "t".into(),
            models: vec!["m".into()],
            block_size: 4,
            hash_id_scope: HashIdScope::Local,
            tool_tokens: 0,
            system_tokens: 0,
            requests: vec![norm(0.0, &[1, 2]), norm(1.0, &[1, 2, 3]), norm(2.0, &[1, 2, 3, 4])],
            totals: None,
        };
        let mut synth = Stub;
        // Full path: trace -> reconstruct -> SimClock dispatch -> export.
        // t*=500ms: turn0 warmup@0, turn1(t=1000->off500), turn2(t=2000->off1500).
        let out = execute_legacy_replay(
            "t",
            &trace,
            &mut synth,
            &std::collections::HashMap::new(),
            &WekaConfig { split_flattened_agents: false, ..WekaConfig::default() },
            &MainReconstructOptions::default(),
            500.0,
            false,
            None,
        )
        .unwrap();
        assert_eq!(out.len(), 1); // root conversation
        let (sid, fired) = &out[0];
        assert_eq!(sid, "t");
        assert_eq!(fired.len(), 3);
        // Fired under the clock at 0, 500ms, 1500ms.
        assert_eq!(fired[0].0, 0);
        assert_eq!(fired[1].0, 500 * NS_PER_MS);
        assert_eq!(fired[2].0, 1500 * NS_PER_MS);
        assert_eq!(fired[0].1["phase"], serde_json::json!("warmup"));
        assert_eq!(fired[1].1["phase"], serde_json::json!("profiling"));
        // Real reconstructed content present on the fired records.
        assert!(!fired[1].1["raw_messages"].as_array().unwrap().is_empty());
    }
}
