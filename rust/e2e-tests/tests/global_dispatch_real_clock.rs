// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! RealClock spot-check: `runtime.dispatch: global` against a live
//! `aiperf-mock-server` process, deterministic TTFT/ITL (zero jitter),
//! asserting raw per-record ISL/OSL/latency/status output AND that the
//! shared global-admission gate holds the *true aggregate* concurrency cap
//! across `workers=4` OS-thread sub-cells — the CLAUDE.md-mandated
//! end-to-end shape for generated-token timing and concurrency-admission
//! changes.
//!
//! This is a real subprocess `aiperf` binary against a real (in-process,
//! but real-socket) `aiperf-mock-server`, driven entirely by `RealClock`:
//! `aiperf profile` against a live HTTP target never selects `SimClock` —
//! that clock only exists behind `transport.type: dynosim_offline` /
//! `dynosim_online` (see `rust/runtime/src/engine/execute.rs` and
//! `rust/cli/src/yaml.rs`, which parse `dynosim_offline`/`dynosim_online` as
//! a distinct `Transport` variant). No such flag or config key appears here,
//! so this run is provably on the real wall clock.
//!
//! `runtime.workers` (thread-per-core sub-cell count) and `runtime.dispatch`
//! are YAML-only config-surface fields (see `rust/cli/src/yaml.rs`
//! `RuntimeSection` and `rust/cli/src/load.rs::Inputs::runtime_workers`) —
//! there is no `--workers` CLI flag, so this test authors a config file
//! rather than passing flags (confirmed by reading `rust/cli/src/flags.rs`,
//! which only exposes `--workers-max`/`--max-workers`, a distinct
//! worker-*process* cap for the Python engine, and `--dispatch`, added in
//! Task 3 / commit 467eedbca and fixed for flag-only (no-YAML) invocation
//! in commit 55b0f9374).

mod common;
use common::*;

const TTFT_MS: f64 = 40.0;
const ITL_MS: f64 = 5.0;
const OSL: usize = 8;
const ISL: usize = 32;
const WORKERS: u32 = 4;
// `3` does NOT evenly divide `WORKERS = 4`, mirroring
// `global_dispatch_enforces_true_aggregate_concurrency_cap_sharded_does_not`
// in `rust/runtime/src/engine/workers_characterization.rs`: `Sharded`
// mode's per-thread `owned_positions(cap, t, workers).max(1)` floor rounds
// one thread's `0` share up to `1`, so `Sharded`'s aggregate cap across all
// 4 threads is `4`, not the authored `3` — verified against a scratch copy
// of this test with `dispatch: sharded` forced, which observed peak
// concurrency of 4 at these parameters (vs. `Global`'s genuinely bounded
// peak <= 3). An evenly divisible cap (the previous `8`/`4` pairing) gives
// `Sharded` a per-thread share of exactly `2` with no `.max(1)` floor
// triggered, so `Sharded` and `Global` produce an IDENTICAL peak of 8 —
// zero discriminating power between the two dispatch modes.
const CONCURRENCY: u32 = 3;
const REQUEST_COUNT: u64 = 24;

#[tokio::test]
async fn global_dispatch_real_clock_concurrency_matches_expected_records() {
    if cfg!(target_os = "macos") {
        return;
    }

    let h = AIPerfHarness::new_with(tuned_mock_config(TTFT_MS, ITL_MS)).await;

    let files = tempfile::TempDir::new().unwrap();
    let cfg_body = format!(
        "schemaVersion: \"2.0\"\n\
         randomSeed: 20260719\n\
         \n\
         benchmark:\n\
        \x20 model: gpt-4\n\
        \x20 endpoint:\n\
        \x20   url: {url}/v1/chat/completions\n\
        \x20   type: chat\n\
        \x20   streaming: true\n\
        \x20 dataset:\n\
        \x20   prompts:\n\
        \x20     isl: {ISL}\n\
        \x20     osl: {OSL}\n\
        \x20 profiling:\n\
        \x20   type: concurrency\n\
        \x20   requests: {REQUEST_COUNT}\n\
        \x20   concurrency: {CONCURRENCY}\n\
        \x20 artifacts:\n\
        \x20   raw: true\n\
        \x20   records:\n\
        \x20     - jsonl\n\
         \n\
         runtime:\n\
        \x20 workers: {WORKERS}\n\
        \x20 dispatch: global\n",
        url = h.mock.url,
    );
    let cfg = files.path().join("global_dispatch_real_clock.yaml");
    std::fs::write(&cfg, cfg_body).unwrap();

    let r = h.run(&format!("--config {} --ui simple", cfg.display()));
    assert!(
        r.success(),
        "global-dispatch RealClock run failed (exit {}):\nstdout:\n{}\nstderr:\n{}",
        r.exit_code,
        r.stdout,
        r.stderr
    );

    let records = r.artifacts.raw_records();
    assert_eq!(
        records.len(),
        REQUEST_COUNT as usize,
        "expected exactly {REQUEST_COUNT} raw records, got {}: full run stdout:\n{}",
        records.len(),
        r.stdout
    );

    if timing_fast_forwarded(&records, TTFT_MS) {
        return;
    }

    assert_raw_records_timing_and_data(
        &records,
        &TunedExpectations::new(TTFT_MS, ITL_MS, OSL)
            .model("gpt-4")
            // 4 worker threads keeping 3 requests in flight globally (vs. a
            // lone concurrency-2 run in `tuned_scheduled_single_turn_raw_timing`)
            // adds real scheduler/OS contention to first-token queue wait. That
            // error is one-sided for the same reason ITL's is (below): the TTFT
            // window contains the mock's first-token sleep, which only resolves
            // to scheduler wakeup granularity and so always overshoots, observed
            // to +15.7 ms under a full-suite run. The band is therefore sized by
            // what it must DETECT rather than by the last sample: a dropped
            // first-token delay (TTFT -> ~0 ms) or a doubled one (-> ~80 ms) each
            // sit a full 40 ms from tuned, so 25 ms rejects both with margin while
            // absorbing contention.
            //
            // ITL is NOT knife-edge here. The mock's per-token pacing is a 5 ms
            // sleep, and a 5 ms timer only resolves to host scheduler wakeup
            // granularity: every token gap rounds UP, never down, so the error
            // is one-sided and accumulates into the mean. Measured over repeated
            // runs of this binary at `--test-threads=4`, the mean ITL lands
            // between 5.0 ms and 7.93 ms — an overshoot of up to +2.93 ms, which
            // a 2 ms band rejects roughly one run in three. 4 ms covers the
            // observed spread while still catching a genuine pacing regression
            // (a doubled or dropped inter-token sleep moves the mean by >= 5 ms).
            .tol_ms(25.0, 4.0),
    );

    let peak = peak_wall_clock_concurrency(&records);
    println!(
        "global_dispatch_real_clock: {} records, peak observed wall-clock concurrency = {peak} \
         (cap = {CONCURRENCY}, workers = {WORKERS})",
        records.len()
    );
    assert!(
        peak <= CONCURRENCY as usize,
        "Global dispatch must hold the TRUE aggregate concurrency cap ({CONCURRENCY}) across all \
         {WORKERS} worker threads; observed peak {peak} exceeds it — this is exactly the \
         over-subscription `Sharded` mode's static per-thread `owned_positions(...).max(1)` floor \
         can transiently produce"
    );
}

/// Peak concurrency via a sweep-line over each record's
/// `[request_start_ns, request_end_ns)` wall-clock interval — the same
/// authoritative bounds `assert_raw_records_timing_and_data` uses for
/// `request_latency` (see `raw_jsonl.rs`'s module doc: these are wall-clock,
/// NOT the `perf_ns` token timeline, so intervals from different records are
/// directly comparable).
///
/// Builds `+1` events at each `request_start_ns` and `-1` events at each
/// `request_end_ns`, sorts by timestamp (ends before starts on a tie, so a
/// request that completes at exactly the instant another starts does not
/// spuriously inflate the count), and returns the maximum running sum.
fn peak_wall_clock_concurrency(records: &[serde_json::Value]) -> usize {
    #[derive(Eq, PartialEq)]
    struct Event {
        ns: i64,
        delta: i64,
    }
    let mut events: Vec<Event> = Vec::with_capacity(records.len() * 2);
    for record in records {
        let metadata = record.get("metadata");
        let start = metadata
            .and_then(|m| m.get("request_start_ns"))
            .and_then(serde_json::Value::as_i64)
            .expect("record missing metadata.request_start_ns");
        let end = metadata
            .and_then(|m| m.get("request_end_ns"))
            .and_then(serde_json::Value::as_i64)
            .expect("record missing metadata.request_end_ns");
        assert!(
            end >= start,
            "record_end_ns {end} < request_start_ns {start}"
        );
        events.push(Event {
            ns: start,
            delta: 1,
        });
        events.push(Event { ns: end, delta: -1 });
    }
    // Process end-events before start-events at the same timestamp so a
    // request's release does not overlap the next request's admission when
    // both land on the identical nanosecond.
    events.sort_by(|a, b| a.ns.cmp(&b.ns).then(a.delta.cmp(&b.delta)));

    let mut running: i64 = 0;
    let mut peak: i64 = 0;
    for event in events {
        running += event.delta;
        peak = peak.max(running);
    }
    peak as usize
}
