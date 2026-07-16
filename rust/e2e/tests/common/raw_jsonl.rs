// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reusable tuned-mock `profile_export_raw.jsonl` TIMING + DATA verification.
//!
//! This is the operational form of the "feature-complete" bar: given a mock
//! server tuned to fixed, jitter-free per-token latencies (`--ttft T`,
//! `--itl I`, `--ttft-jitter-cv 0 --itl-jitter-cv 0`, analytic mode, fixed
//! synthetic ISL/OSL), every raw record's on-the-wire token timing must
//! reproduce the tuned model within a tight transport tolerance — proving the
//! whole `Python -> aiperf runner -> transport -> record` path measures and
//! persists per-request timing faithfully (single-process, cellular fold+ship+
//! merge, multi-turn, or graph).
//!
//! # The raw-record schema this parses
//!
//! Each line of `profile_export_raw.jsonl` (see
//! `aiperf_runtime::engine::records::RawRecordRow`) carries:
//!   * `metadata.request_start_ns` / `request_end_ns` — wall-clock request
//!     bounds; their difference is the authoritative `request_latency`.
//!   * `start_perf_ns` — the request's start on the perf-counter timeline that
//!     `responses[].perf_ns` share (a DIFFERENT epoch from `request_start_ns`,
//!     so never subtract one from the other).
//!   * `responses[]` — one entry per received SSE frame, each
//!     `{ perf_ns, packets: [{ name: "data", value: "<chunk>" }] }`. The chunk
//!     `value` is either a JSON `chat.completion.chunk` or the literal `[DONE]`.
//!
//! # The critical OSL rule (earned in blood)
//!
//! OSL is the count of *generated-token* chunks — a data chunk whose
//! `choices[0].delta.content` is present (non-null). It EXCLUDES:
//!   * the terminal `[DONE]` sentinel,
//!   * the `stream_options.include_usage` usage chunk (empty `choices`), which
//!     arrives ~0 ms after the last token, and
//!   * `reasoning_content` chunks (reasoning models stream those separately).
//! Counting the trailing usage/`[DONE]` chunk both inflates OSL by one AND
//! dilutes the measured ITL (its ~0 ms gap drags the mean down), which is why
//! callers pin a NON-reasoning model (`gpt-4`) so `content` chunks == the
//! requested output cap exactly.
//!
//! Timing derived per record (all in milliseconds):
//!   * TTFT   = (first content chunk `perf_ns` - `start_perf_ns`)
//!   * ITL    = mean gap between consecutive content chunks' `perf_ns`
//!   * latency = (`request_end_ns` - `request_start_ns`)
//!
//! Reference (ttft=100, itl=10, osl=8): TTFT ~101.1 ms, ITL ~9.99 ms,
//! latency ~171 ms, OSL 8.

use serde_json::Value;

use super::MockServerConfig;

/// A mock tuned for deterministic, jitter-free per-token latency and fixed
/// sequence lengths — the target every timing assertion is checked against.
///
/// Analytic mode (scheduler off, the default) is used so per-request latency is
/// `ttft + (osl - 1) * itl` independent of concurrency, and `no_tokenizer` keeps
/// the mock fast to start without a HF download. Both jitter CVs are pinned to 0
/// so the model is exact, not distributional.
pub fn tuned_mock_config(ttft_ms: f64, itl_ms: f64) -> MockServerConfig {
    let mut cfg = MockServerConfig::default();
    cfg.no_tokenizer = true;
    cfg.ttft = ttft_ms;
    cfg.itl = itl_ms;
    cfg.ttft_jitter_cv = 0.0;
    cfg.itl_jitter_cv = 0.0;
    // Analytic closed-form TTFT/ITL, not the batched scheduler.
    cfg.scheduler_enabled = false;
    // --fast would zero the latencies; make sure it is off.
    cfg.fast = false;
    cfg
}

/// The tuned values each raw record's timing + data must reproduce.
#[derive(Debug, Clone)]
pub struct TunedExpectations {
    /// Configured mock TTFT (ms) — the first content token should arrive this
    /// long after request start, within `tol_ms`.
    pub ttft_ms: f64,
    /// Configured mock ITL (ms) — the mean inter-content-token gap, within
    /// `tol_ms`.
    pub itl_ms: f64,
    /// Requested output cap == the exact number of content chunks per record.
    pub osl: usize,
    /// Expected model string on each response chunk, when set.
    pub model: Option<String>,
    /// Expected HTTP status on each record (defaults to 200).
    pub status: u16,
    /// One-sided TTFT tolerance (ms). TTFT folds in connection setup + first-token
    /// queue wait, which inflates under CPU contention (a full parallel e2e suite,
    /// or a fork/multi-cell run); in isolation the overhead is ~1-2 ms. Kept tight,
    /// not a wide band — pass a small value for an isolated run.
    pub ttft_tol_ms: f64,
    /// One-sided ITL tolerance (ms). ITL is steady-state per-token pacing (the mock
    /// sleeps `itl` between tokens regardless of CPU pressure) and averages over the
    /// stream, so it stays knife-edge tight (~0.1 ms isolated, ~1 ms under load).
    pub itl_tol_ms: f64,
}

impl TunedExpectations {
    /// Tuned expectations for `(ttft_ms, itl_ms, osl)` with tight default
    /// tolerances and an HTTP-200 success expectation. Defaults leave TTFT enough
    /// headroom to be non-flaky in the shared parallel e2e suite while ITL stays
    /// knife-edge; tighten both for an isolated run.
    pub fn new(ttft_ms: f64, itl_ms: f64, osl: usize) -> Self {
        Self {
            ttft_ms,
            itl_ms,
            osl,
            model: None,
            status: 200,
            // Tight: the manual isolated reference measured ~1 ms of transport
            // overhead on TTFT and ~0.03 ms on ITL. These leave a little slack for
            // loopback/connection jitter without becoming a wide band. (Callers
            // that cannot isolate the run should widen TTFT via `tol_ms`.)
            ttft_tol_ms: 6.0,
            itl_tol_ms: 2.0,
        }
    }

    /// Pin the model each response chunk must carry.
    pub fn model(mut self, model: &str) -> Self {
        self.model = Some(model.to_string());
        self
    }

    /// Override both tolerances (ms): TTFT and ITL.
    pub fn tol_ms(mut self, ttft_tol_ms: f64, itl_tol_ms: f64) -> Self {
        self.ttft_tol_ms = ttft_tol_ms;
        self.itl_tol_ms = itl_tol_ms;
        self
    }
}

/// The timing + data extracted from one raw record.
#[derive(Debug, Clone)]
pub struct RawRecordTiming {
    /// Count of content-bearing (generated-token) chunks == OSL.
    pub osl: usize,
    /// (first content `perf_ns` - `start_perf_ns`) in ms, if any content chunk.
    pub ttft_ms: Option<f64>,
    /// Mean gap between consecutive content chunks in ms, if >= 2 chunks.
    pub itl_ms: Option<f64>,
    /// (`request_end_ns` - `request_start_ns`) in ms.
    pub latency_ms: f64,
    /// HTTP status, if present.
    pub status: Option<u16>,
    /// Model string observed on the first content chunk, if present.
    pub model: Option<String>,
}

/// Iterate a raw record's SSE data chunks, yielding `(perf_ns, parsed_json)` for
/// every non-`[DONE]` data packet.
fn data_chunks(record: &Value) -> Vec<(i64, Value)> {
    let mut out = Vec::new();
    let Some(responses) = record.get("responses").and_then(Value::as_array) else {
        return out;
    };
    for resp in responses {
        let perf_ns = resp.get("perf_ns").and_then(Value::as_i64).unwrap_or(0);
        let Some(packets) = resp.get("packets").and_then(Value::as_array) else {
            continue;
        };
        for packet in packets {
            if packet.get("name").and_then(Value::as_str) != Some("data") {
                continue;
            }
            let Some(raw) = packet.get("value").and_then(Value::as_str) else {
                continue;
            };
            let trimmed = raw.trim();
            if trimmed == "[DONE]" {
                continue;
            }
            if let Ok(obj) = serde_json::from_str::<Value>(trimmed) {
                out.push((perf_ns, obj));
            }
        }
    }
    out
}

/// True when an SSE `chat.completion.chunk` carries a generated *content* token
/// (`choices[0].delta.content` present and non-null). This is the exact
/// predicate that defines OSL — it is false for the usage chunk (empty
/// `choices`) and for `reasoning_content`-only chunks.
fn is_content_chunk(obj: &Value) -> bool {
    obj.get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("delta"))
        .and_then(|delta| delta.get("content"))
        .map(|content| !content.is_null())
        .unwrap_or(false)
}

/// Extract the tuned-relevant timing + data from a single raw record.
pub fn extract_timing(record: &Value) -> RawRecordTiming {
    let start_perf_ns = record.get("start_perf_ns").and_then(Value::as_i64);
    let metadata = record.get("metadata");
    let request_start_ns = metadata
        .and_then(|m| m.get("request_start_ns"))
        .and_then(Value::as_i64)
        .unwrap_or(0);
    let request_end_ns = metadata
        .and_then(|m| m.get("request_end_ns"))
        .and_then(Value::as_i64)
        .unwrap_or(0);

    let chunks = data_chunks(record);
    let content_perf: Vec<i64> = chunks
        .iter()
        .filter(|(_, obj)| is_content_chunk(obj))
        .map(|(perf_ns, _)| *perf_ns)
        .collect();

    let model = chunks
        .iter()
        .find(|(_, obj)| is_content_chunk(obj))
        .and_then(|(_, obj)| obj.get("model").and_then(Value::as_str))
        .map(str::to_string);

    let ttft_ms = match (start_perf_ns, content_perf.first()) {
        (Some(start), Some(&first)) => Some((first - start) as f64 / 1e6),
        _ => None,
    };

    let itl_ms = if content_perf.len() >= 2 {
        let gaps: Vec<f64> = content_perf
            .windows(2)
            .map(|pair| (pair[1] - pair[0]) as f64 / 1e6)
            .collect();
        Some(gaps.iter().sum::<f64>() / gaps.len() as f64)
    } else {
        None
    };

    RawRecordTiming {
        osl: content_perf.len(),
        ttft_ms,
        itl_ms,
        latency_ms: (request_end_ns - request_start_ns) as f64 / 1e6,
        status: record
            .get("status")
            .and_then(Value::as_u64)
            .map(|s| s as u16),
        model,
    }
}

/// Detect a timer-virtualizing sandbox that fast-forwarded the mock's `timerfd`
/// sleeps to ~0 ms, collapsing the tuned latencies.
///
/// Returns `true` (and prints a clear `SKIP:` line) when the first record's
/// measured TTFT is drastically below the tuned value (`< tuned / 4`), which is
/// only possible if the timer was virtualized — a real slow OR fast run still
/// pays the tuned first-token sleep, so this never masks a genuine timing
/// regression. Timing tests call this first and `return` early when it is true,
/// turning the otherwise-confusing "TTFT 0.05ms not within 6ms of 100ms" hard
/// failure into an explicit skip.
pub fn timing_fast_forwarded(records: &[Value], tuned_ttft_ms: f64) -> bool {
    let Some(first) = records.first() else {
        return false;
    };
    let Some(ttft) = extract_timing(first).ttft_ms else {
        return false;
    };
    if ttft < tuned_ttft_ms / 4.0 {
        eprintln!(
            "SKIP: timing e2e requires an un-sandboxed / real-timer environment \
             (mock timerfd sleeps were fast-forwarded: first-record TTFT {ttft:.3}ms \
             is far below tuned {tuned_ttft_ms}ms)"
        );
        return true;
    }
    false
}

/// Assert every raw record's on-the-wire TIMING (TTFT / ITL / request_latency)
/// and DATA (OSL / model / status) reproduces the tuned mock within
/// `expected.tol_ms`.
///
/// Panics with a per-record diagnostic on the first violation. `records` must be
/// non-empty — an empty slice means the run produced no `profile_export_raw.jsonl`
/// (wrong `--export-level`, or the run failed), which is itself a failure.
pub fn assert_raw_records_timing_and_data(records: &[Value], expected: &TunedExpectations) {
    assert!(
        !records.is_empty(),
        "no raw records to verify — did the run pass `--export-level raw` and succeed?"
    );

    let ttft_tol = expected.ttft_tol_ms;
    let itl_tol = expected.itl_tol_ms;
    // request_latency = TTFT + (osl-1) ITLs, so its rigorous error bound is the
    // sum of the per-component tolerances — TTFT contention plus one ITL-tol per
    // gap. (In practice ITL error is a small systematic bias, not random, so the
    // real drift is far under this bound.)
    let latency_tol = ttft_tol + (expected.osl.saturating_sub(1)) as f64 * itl_tol;
    let expected_latency =
        expected.ttft_ms + (expected.osl.saturating_sub(1)) as f64 * expected.itl_ms;

    for (index, record) in records.iter().enumerate() {
        let timing = extract_timing(record);

        // DATA: HTTP status.
        assert_eq!(
            timing.status,
            Some(expected.status),
            "record {index}: status {:?} != expected {}\n{record}",
            timing.status,
            expected.status
        );

        // DATA: OSL == the requested output cap (content chunks only).
        assert_eq!(
            timing.osl, expected.osl,
            "record {index}: OSL (content chunks) {} != expected {} \
             (did a usage/[DONE] chunk leak in, or a reasoning model split the stream?)",
            timing.osl, expected.osl
        );

        // DATA: model, when pinned.
        if let Some(want_model) = &expected.model {
            assert_eq!(
                timing.model.as_deref(),
                Some(want_model.as_str()),
                "record {index}: model {:?} != expected {want_model:?}",
                timing.model
            );
        }

        // TIMING: TTFT ~= configured ttft within tol.
        let ttft = timing
            .ttft_ms
            .unwrap_or_else(|| panic!("record {index}: no content chunk, cannot measure TTFT"));
        assert!(
            (ttft - expected.ttft_ms).abs() <= ttft_tol,
            "record {index}: TTFT {ttft:.2}ms is not within {ttft_tol}ms of tuned {}ms",
            expected.ttft_ms
        );

        // TIMING: mean ITL ~= configured itl within itl_tol (only when >= 2 tokens).
        if expected.osl >= 2 {
            let itl = timing.itl_ms.unwrap_or_else(|| {
                panic!("record {index}: OSL {} but no ITL computed", timing.osl)
            });
            assert!(
                (itl - expected.itl_ms).abs() <= itl_tol,
                "record {index}: mean ITL {itl:.3}ms is not within {itl_tol}ms of tuned {}ms",
                expected.itl_ms
            );
        }

        // TIMING: request_latency ~= ttft + (osl-1)*itl within the accumulated tol.
        assert!(
            (timing.latency_ms - expected_latency).abs() <= latency_tol,
            "record {index}: request_latency {:.2}ms is not within {latency_tol}ms of \
             tuned {expected_latency:.2}ms (ttft + (osl-1)*itl)",
            timing.latency_ms
        );
    }
}

/// Assert every raw record's TIMING against the tuned `(ttft_ms, itl_ms)` using
/// **each record's own OSL** — for paths where the output length is not a fixed
/// constant (authored-payload multi-turn `inputs_json`, or a `dag_jsonl` graph
/// whose nodes carry a `max_tokens` the mock streams a variable count for,
/// because aiperf sends no exact-output control on those bodies).
///
/// Per record: TTFT ~= `ttft_ms`, mean ITL ~= `itl_ms` (when OSL >= 2), and
/// `request_latency` ~= `ttft_ms + (osl - 1) * itl_ms` where `osl` is the
/// record's measured content-chunk count. Each record must be a streamed HTTP
/// 200 with at least one content token.
/// `ttft_tol_ms` and `itl_tol_ms` are split because they degrade differently
/// under load: ITL is steady-state per-token pacing (the mock sleeps `itl`
/// between tokens regardless of CPU pressure, and averaging over the stream
/// smooths read jitter), so it stays knife-edge tight; TTFT folds in connection
/// setup + first-token queue wait, which inflates under CPU contention (many
/// concurrent subprocess-heavy e2e tests, or a fork/multi-cell graph run). Give
/// TTFT the headroom it needs for the path, keep ITL tight.
pub fn assert_raw_records_timing_self_consistent(
    records: &[Value],
    ttft_ms: f64,
    itl_ms: f64,
    ttft_tol_ms: f64,
    itl_tol_ms: f64,
) {
    assert_raw_records_timing_self_consistent_model(
        records,
        ttft_ms,
        itl_ms,
        ttft_tol_ms,
        itl_tol_ms,
        None,
    )
}

/// Like [`assert_raw_records_timing_self_consistent`] but additionally pins the
/// `model` on every record's content chunks when `model` is `Some` — the DATA
/// check the fixed-OSL variant carries. Callers whose payloads pin a model
/// (e.g. the multi-turn `inputs_json` fixtures streaming `gpt-4`) should assert
/// it here too, so a model-routing regression cannot hide behind timing-only
/// coverage.
pub fn assert_raw_records_timing_self_consistent_model(
    records: &[Value],
    ttft_ms: f64,
    itl_ms: f64,
    ttft_tol_ms: f64,
    itl_tol_ms: f64,
    model: Option<&str>,
) {
    assert!(
        !records.is_empty(),
        "no raw records to verify — did the run pass `--export-level raw` and succeed?"
    );

    for (index, record) in records.iter().enumerate() {
        let timing = extract_timing(record);

        assert_eq!(
            timing.status,
            Some(200),
            "record {index}: status {:?} != 200\n{record}",
            timing.status
        );
        assert!(
            timing.osl >= 1,
            "record {index}: no content (generated-token) chunks in the stream"
        );

        // DATA: model, when pinned — parity with the fixed-OSL variant.
        if let Some(want_model) = model {
            assert_eq!(
                timing.model.as_deref(),
                Some(want_model),
                "record {index}: model {:?} != expected {want_model:?}",
                timing.model
            );
        }

        let ttft = timing
            .ttft_ms
            .unwrap_or_else(|| panic!("record {index}: no content chunk, cannot measure TTFT"));
        assert!(
            (ttft - ttft_ms).abs() <= ttft_tol_ms,
            "record {index}: TTFT {ttft:.2}ms is not within {ttft_tol_ms}ms of tuned {ttft_ms}ms"
        );

        if timing.osl >= 2 {
            let itl = timing.itl_ms.unwrap_or_else(|| {
                panic!("record {index}: OSL {} but no ITL computed", timing.osl)
            });
            assert!(
                (itl - itl_ms).abs() <= itl_tol_ms,
                "record {index}: mean ITL {itl:.3}ms is not within {itl_tol_ms}ms of tuned {itl_ms}ms"
            );
        }

        // Latency inherits the TTFT contention plus one ITL-tol per gap.
        let expected_latency = ttft_ms + (timing.osl.saturating_sub(1)) as f64 * itl_ms;
        let latency_tol = ttft_tol_ms + (timing.osl.saturating_sub(1)) as f64 * itl_tol_ms.max(0.5);
        assert!(
            (timing.latency_ms - expected_latency).abs() <= latency_tol,
            "record {index}: request_latency {:.2}ms is not within {latency_tol}ms of \
             self-consistent {expected_latency:.2}ms (ttft + (osl-1)*itl, osl={})",
            timing.latency_ms,
            timing.osl
        );
    }
}

/// Sandbox-safe unit coverage for the classification/timing core
/// (`is_content_chunk` / `data_chunks` / `extract_timing`).
///
/// These are PURE PARSING tests over synthetic `serde_json` records — no mock
/// server, no network, no `timerfd` sleeps — so they run identically in a
/// normal sandbox and a timer-virtualizing CI. They pin the earned-in-blood OSL
/// gotcha (§"The critical OSL rule") independent of the wall-clock e2e tests,
/// which cannot run under a timer-fast-forwarding sandbox: the terminal usage
/// chunk (empty `choices`), the `[DONE]` sentinel, and `reasoning_content`-only
/// chunks must NOT count toward OSL, and ITL must be computed from
/// content-chunk gaps only.
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// One second in nanoseconds, used to keep synthetic perf timelines readable.
    const MS: i64 = 1_000_000;

    /// A generated-content SSE `chat.completion.chunk` (`choices[0].delta.content`
    /// present and non-null) — the ONLY chunk kind that counts toward OSL.
    fn content_chunk(model: &str, token: &str) -> serde_json::Value {
        json!({
            "id": "chatcmpl-x",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": model,
            "choices": [{
                "index": 0,
                "finish_reason": null,
                "delta": {"content": token},
            }],
        })
    }

    /// A reasoning-only chunk: `reasoning_content` present, `content` explicitly
    /// null. Reasoning models stream these separately; they must be excluded.
    fn reasoning_chunk(model: &str, token: &str) -> serde_json::Value {
        json!({
            "id": "chatcmpl-x",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": model,
            "choices": [{
                "index": 0,
                "finish_reason": null,
                "delta": {"content": null, "reasoning_content": token},
            }],
        })
    }

    /// The terminal `stream_options.include_usage` chunk: empty `choices`, a
    /// trailing `usage` block, arriving ~0 ms after the last token.
    fn usage_chunk(model: &str) -> serde_json::Value {
        json!({
            "id": "chatcmpl-x",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": model,
            "choices": [],
            "usage": {"prompt_tokens": 64, "completion_tokens": 1, "total_tokens": 65},
        })
    }

    /// Wrap an already-serialized SSE payload string into one `responses[]` entry
    /// at the given perf timestamp. `raw` is placed verbatim in the `data` packet
    /// value — pass a JSON chunk string or the literal `"[DONE]"`.
    fn response_entry(perf_ns: i64, raw: &str) -> serde_json::Value {
        json!({
            "perf_ns": perf_ns,
            "packets": [{"name": "data", "value": raw}],
        })
    }

    /// One `responses[]` entry carrying a JSON chunk value at `perf_ns`.
    fn chunk_entry(perf_ns: i64, chunk: &serde_json::Value) -> serde_json::Value {
        response_entry(perf_ns, &chunk.to_string())
    }

    /// Assemble a synthetic raw record from `start_perf_ns`, the request wall-clock
    /// bounds, and a set of `responses[]` entries.
    fn record(
        start_perf_ns: i64,
        request_start_ns: i64,
        request_end_ns: i64,
        status: u16,
        responses: Vec<serde_json::Value>,
    ) -> serde_json::Value {
        json!({
            "start_perf_ns": start_perf_ns,
            "status": status,
            "metadata": {
                "request_start_ns": request_start_ns,
                "request_end_ns": request_end_ns,
            },
            "responses": responses,
        })
    }

    /// The canonical gotcha case: a stream of one content chunk, one reasoning-only
    /// chunk, one usage-only chunk, and the `[DONE]` sentinel must yield OSL == 1
    /// (ONLY the content chunk), a TTFT/latency derived from the content chunk's
    /// perf timestamp, and no ITL (single token).
    #[test]
    fn osl_excludes_reasoning_usage_and_done() {
        let start = MS;
        // Reasoning streams before the content token; usage + [DONE] trail it. The
        // reasoning/usage/[DONE] perf timestamps are chosen so that if any leaked
        // into the content set, TTFT or ITL would visibly shift.
        let responses = vec![
            chunk_entry(start + 40 * MS, &reasoning_chunk("gpt-4", "hmm")),
            chunk_entry(start + 100 * MS, &content_chunk("gpt-4", "hello")),
            chunk_entry(start + 100 * MS, &usage_chunk("gpt-4")),
            response_entry(start + 100 * MS, "[DONE]"),
        ];
        let rec = record(start, 0, 171 * MS, 200, responses);

        let timing = extract_timing(&rec);
        assert_eq!(timing.osl, 1, "only the content chunk counts toward OSL");
        assert_eq!(timing.status, Some(200));
        assert_eq!(timing.model.as_deref(), Some("gpt-4"));
        assert_eq!(
            timing.ttft_ms,
            Some(100.0),
            "TTFT from content chunk perf_ns"
        );
        assert_eq!(timing.itl_ms, None, "osl==1 has no inter-token gap");
        assert_eq!(timing.latency_ms, 171.0);
    }

    /// A record with no reasoning/usage/[DONE] at all still counts its lone content
    /// chunk and yields no ITL — the osl==1 edge case in isolation.
    #[test]
    fn single_content_chunk_has_no_itl() {
        let start = 5 * MS;
        let responses = vec![chunk_entry(start + 100 * MS, &content_chunk("gpt-4", "a"))];
        let rec = record(start, 0, 105 * MS, 200, responses);

        let timing = extract_timing(&rec);
        assert_eq!(timing.osl, 1);
        assert_eq!(timing.ttft_ms, Some(100.0));
        assert_eq!(timing.itl_ms, None);
        assert_eq!(timing.latency_ms, 105.0);
    }

    /// Multi-content stream: OSL counts every content chunk, ITL is the mean gap of
    /// consecutive CONTENT chunks only — interleaved reasoning/usage/[DONE] entries
    /// (including one whose perf_ns falls between two content chunks) must not
    /// perturb the ITL mean.
    #[test]
    fn multi_content_itl_from_content_gaps_only() {
        let start = 2 * MS;
        // Content chunks at +100, +112, +120 -> gaps 12 and 8 -> mean ITL 10.
        // A usage chunk lands at +116 (between two content chunks); if it were
        // (wrongly) counted, the gap sequence would change and the mean would drift.
        let responses = vec![
            chunk_entry(start + 100 * MS, &content_chunk("gpt-4", "t0")),
            chunk_entry(start + 112 * MS, &content_chunk("gpt-4", "t1")),
            chunk_entry(start + 116 * MS, &usage_chunk("gpt-4")),
            chunk_entry(start + 120 * MS, &content_chunk("gpt-4", "t2")),
            response_entry(start + 120 * MS, "[DONE]"),
        ];
        let rec = record(start, 0, 130 * MS, 200, responses);

        let timing = extract_timing(&rec);
        assert_eq!(timing.osl, 3, "three content chunks");
        assert_eq!(timing.ttft_ms, Some(100.0));
        let itl = timing.itl_ms.expect("osl>=2 yields an ITL");
        assert!(
            (itl - 10.0).abs() < 1e-9,
            "ITL should be the mean of content-only gaps (12, 8) = 10, got {itl}"
        );
        assert_eq!(timing.latency_ms, 130.0);
    }

    /// A leading reasoning burst before any content: OSL and TTFT anchor on the
    /// FIRST content chunk, not the earlier reasoning tokens.
    #[test]
    fn reasoning_prefix_does_not_shift_ttft() {
        let start = 0;
        let responses = vec![
            chunk_entry(10 * MS, &reasoning_chunk("gpt-4", "r0")),
            chunk_entry(20 * MS, &reasoning_chunk("gpt-4", "r1")),
            chunk_entry(100 * MS, &content_chunk("gpt-4", "c0")),
            chunk_entry(110 * MS, &content_chunk("gpt-4", "c1")),
        ];
        let rec = record(start, 0, 110 * MS, 200, responses);

        let timing = extract_timing(&rec);
        assert_eq!(timing.osl, 2, "only content chunks, reasoning excluded");
        assert_eq!(timing.ttft_ms, Some(100.0), "TTFT anchors on first content");
        assert_eq!(timing.itl_ms, Some(10.0));
    }

    /// An empty (or content-free) response set yields OSL 0 and no TTFT/ITL — the
    /// helper must not panic, so callers can surface "no content chunk" cleanly.
    #[test]
    fn empty_and_content_free_records() {
        let empty = record(0, 0, 50 * MS, 200, vec![]);
        let t = extract_timing(&empty);
        assert_eq!(t.osl, 0);
        assert_eq!(t.ttft_ms, None);
        assert_eq!(t.itl_ms, None);
        assert_eq!(t.model, None);
        assert_eq!(t.latency_ms, 50.0);

        // Only reasoning + usage + [DONE], never a content token.
        let no_content = record(
            0,
            0,
            50 * MS,
            200,
            vec![
                chunk_entry(10 * MS, &reasoning_chunk("gpt-4", "r")),
                chunk_entry(20 * MS, &usage_chunk("gpt-4")),
                response_entry(20 * MS, "[DONE]"),
            ],
        );
        let t = extract_timing(&no_content);
        assert_eq!(t.osl, 0, "reasoning/usage/[DONE] never count as content");
        assert_eq!(t.ttft_ms, None);
        assert_eq!(t.itl_ms, None);
    }

    /// `is_content_chunk` directly: true only for a present, non-null
    /// `choices[0].delta.content`; false for null content, empty choices, and a
    /// missing delta.
    #[test]
    fn is_content_chunk_predicate() {
        assert!(is_content_chunk(&content_chunk("m", "x")));
        assert!(!is_content_chunk(&reasoning_chunk("m", "x")));
        assert!(!is_content_chunk(&usage_chunk("m")));
        assert!(!is_content_chunk(&json!({"choices": [{"index": 0}]})));
        assert!(!is_content_chunk(&json!({})));
    }

    /// `data_chunks` skips the `[DONE]` sentinel and non-`data` packets but keeps
    /// every parseable JSON `data` packet with its perf timestamp.
    #[test]
    fn data_chunks_skips_done_and_nondata() {
        let rec = json!({
            "responses": [
                {"perf_ns": 10 * MS, "packets": [{"name": "data", "value": content_chunk("m", "a").to_string()}]},
                {"perf_ns": 20 * MS, "packets": [{"name": "event", "value": "ping"}]},
                {"perf_ns": 30 * MS, "packets": [{"name": "data", "value": "[DONE]"}]},
            ],
        });
        let chunks = data_chunks(&rec);
        assert_eq!(
            chunks.len(),
            1,
            "only the parseable JSON data packet survives"
        );
        assert_eq!(chunks[0].0, 10 * MS);
    }
}
