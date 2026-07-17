// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared integration-test helpers.
//!
//! The product `run_*` one-shot HTTP builders and the synthetic/JSONL
//! conversation loaders were removed from the crate; Python composes real runs
//! through the runner. These helpers rebuild equivalent synthetic and
//! trace-timestamped conversation sources over the live native dataset +
//! prepared-endpoint path, and re-compose the scheduled/fixed-schedule/
//! user-centric online runtimes from public building blocks, so the timing,
//! scheduling, and transport coverage the deleted helpers exercised is
//! preserved.

#![allow(dead_code)]

use std::rc::Rc;

use aiperf_runtime::ancillary::{AncillaryTimingConfig, parse_base_urls, url_selector};
use aiperf_runtime::clock::{Clock, RealClock};
use aiperf_runtime::dataset::{
    ComposeConfig, DatasetSource, LoadConfig, LoaderRegistry, TiktokenTokenizer,
};
use aiperf_runtime::endpoints::{
    EndpointId, EndpointRegistry, PreparedEndpointTable, RawEndpointConfig,
};
use aiperf_runtime::fixed_schedule::{
    DatasetFixedScheduleSource, FixedScheduleConfig, FixedScheduleWorkload,
};
use aiperf_runtime::multiturn::{
    ConversationSource, NativeDatasetConversationSource, PreparedEndpointReference,
};
use aiperf_runtime::rng::RngRoot;
use aiperf_runtime::scheduled::{
    ScheduledAncillaryPolicies, ScheduledRunReport, SingleTurnDatasetWorkload, TurnDispatcher,
    TurnRecordProcessor, Workload, run_scheduled_workload_with_ancillary,
    run_scheduled_workload_with_processors,
};
use aiperf_runtime::timing::{Phase, StopConfig};
use aiperf_runtime::transport::http::TransportSink;
use aiperf_runtime::user_centric::{UserCentricConfig, UserCentricWorkload};

/// Build a conversation source over the live native dataset + prepared chat
/// endpoint path from an inline `multi_turn` conversation array.
pub async fn prepared_source_from_conversations(
    conversations: serde_json::Value,
    model: &str,
    default_output_tokens: usize,
) -> Box<dyn ConversationSource> {
    let dataset = LoaderRegistry::with_builtin_formats()
        .unwrap()
        .build_dataset(
            Some("multi_turn"),
            &LoadConfig::new(DatasetSource::Inline(conversations)),
            &ComposeConfig::new(model, RngRoot::new(Some(1))),
            &TiktokenTokenizer::builtin(),
        )
        .await
        .unwrap();
    let registry = EndpointRegistry::builtin().unwrap();
    let endpoint_id = EndpointId::new("chat").unwrap();
    let endpoint = registry
        .prepare(
            &endpoint_id,
            RawEndpointConfig {
                streaming: true,
                use_server_token_count: true,
                ..RawEndpointConfig::default()
            },
        )
        .unwrap();
    let mut table = PreparedEndpointTable::new();
    let key = table.push(endpoint).unwrap();
    Box::new(
        NativeDatasetConversationSource::sequential_with_prepared_endpoint(
            dataset,
            model,
            default_output_tokens,
            Rc::new(table),
            PreparedEndpointReference { key, endpoint_id },
        )
        .unwrap(),
    )
}

/// Synthetic multi-turn conversation source.
pub async fn synthetic_prepared_source(
    turns: usize,
    input_tokens: usize,
    output_tokens: usize,
    think_time_ms: Option<u64>,
    model: &str,
) -> Box<dyn ConversationSource> {
    let mut turn_objs = Vec::new();
    for index in 0..turns.max(1) {
        let mut turn = serde_json::json!({
            "text": format!("turn {index}: {}", vec!["lorem"; input_tokens].join(" ")),
            "input_length": input_tokens,
            "output_length": output_tokens,
        });
        if index > 0 {
            turn["delay"] = serde_json::json!(think_time_ms.unwrap_or(0));
        }
        turn_objs.push(turn);
    }
    prepared_source_from_conversations(
        serde_json::json!([{"session_id":"synthetic","turns": turn_objs}]),
        model,
        output_tokens,
    )
    .await
}

/// Trace-timestamped single-turn conversation source: `(session_id, timestamp_ms)`.
pub async fn timestamped_prepared_source(
    entries: &[(&str, f64)],
    model: &str,
    default_output_tokens: usize,
) -> Box<dyn ConversationSource> {
    let convs = entries
        .iter()
        .map(|(id, timestamp)| {
            serde_json::json!({
                "session_id": id,
                "turns": [{
                    "text": "hello",
                    "timestamp": timestamp,
                    "input_length": 2,
                    "output_length": default_output_tokens,
                }],
            })
        })
        .collect::<Vec<_>>();
    prepared_source_from_conversations(serde_json::json!(convs), model, default_output_tokens).await
}

/// A prepared-endpoint table with the builtin streaming `chat` endpoint at key
/// 0 — the endpoint `synthetic_prepared_source`/`prepared_source_from_conversations`
/// bind. Attach it to the dispatching `TransportSink` so prepared turns resolve.
pub fn chat_dispatch_table() -> Rc<PreparedEndpointTable> {
    let endpoint = EndpointRegistry::builtin()
        .unwrap()
        .prepare(
            &EndpointId::new("chat").unwrap(),
            RawEndpointConfig {
                streaming: true,
                use_server_token_count: true,
                ..RawEndpointConfig::default()
            },
        )
        .unwrap();
    let mut table = PreparedEndpointTable::new();
    table.push(endpoint).unwrap();
    Rc::new(table)
}

fn scheduled_policies(seed: u64, base_urls: &[String]) -> ScheduledAncillaryPolicies {
    ScheduledAncillaryPolicies {
        cancellation_policy: AncillaryTimingConfig::default()
            .cancellation_policy(seed)
            .unwrap(),
        url_selector: url_selector(base_urls).unwrap(),
        phase: Phase::Profiling,
    }
}

/// Re-composed equivalent of the removed `run_fixed_schedule_online`.
pub async fn run_fixed_schedule_online(
    base_url: String,
    model: String,
    conversations: Box<dyn ConversationSource>,
    config: FixedScheduleConfig,
    http2: bool,
) -> anyhow::Result<ScheduledRunReport> {
    run_fixed_schedule_online_with_ancillary(
        base_url,
        model,
        conversations,
        config,
        http2,
        AncillaryTimingConfig::default(),
        0,
    )
    .await
}

/// Re-composed equivalent of the removed `run_fixed_schedule_online_with_ancillary`.
#[allow(clippy::too_many_arguments)]
pub async fn run_fixed_schedule_online_with_ancillary(
    base_url: String,
    model: String,
    conversations: Box<dyn ConversationSource>,
    config: FixedScheduleConfig,
    http2: bool,
    ancillary: AncillaryTimingConfig,
    seed: u64,
) -> anyhow::Result<ScheduledRunReport> {
    ancillary.validate()?;
    let base_urls = parse_base_urls(&base_url)?;
    let schedule_source = Rc::new(DatasetFixedScheduleSource::new(config)?);
    let workload: Rc<dyn Workload> =
        Rc::new(FixedScheduleWorkload::new(conversations, schedule_source)?);
    let clock: Rc<dyn Clock> = RealClock::new();
    let start_ns = clock.now_ns();
    let dispatcher: Rc<dyn TurnDispatcher> = Rc::new(
        TransportSink::new_multi(clock.clone(), start_ns, &base_urls, model, http2)?
            .with_wire_response_capture(false)
            .with_prepared_endpoints(chat_dispatch_table()),
    );
    run_scheduled_workload_with_ancillary(
        workload,
        clock,
        start_ns,
        dispatcher,
        StopConfig::default(),
        false,
        scheduled_policies(seed, &base_urls),
    )
    .await
}

/// Re-composed equivalent of the removed `run_user_centric_online`.
pub async fn run_user_centric_online(
    base_url: String,
    model: String,
    conversations: Box<dyn ConversationSource>,
    config: UserCentricConfig,
    stop: StopConfig,
    http2: bool,
) -> anyhow::Result<ScheduledRunReport> {
    let base_urls = parse_base_urls(&base_url)?;
    let clock: Rc<dyn Clock> = RealClock::new();
    let start_ns = clock.now_ns();
    let dispatcher: Rc<dyn TurnDispatcher> = Rc::new(
        TransportSink::new_multi(clock.clone(), start_ns, &base_urls, model, http2)?
            .with_wire_response_capture(false)
            .with_prepared_endpoints(chat_dispatch_table()),
    );
    let workload: Rc<dyn Workload> = Rc::new(UserCentricWorkload::new(config, conversations)?);
    run_scheduled_workload_with_ancillary(
        workload,
        clock,
        start_ns,
        dispatcher,
        stop,
        true,
        scheduled_policies(0, &base_urls),
    )
    .await
}

/// Re-composed equivalent of the removed `run_single_turn_dataset_online`.
///
/// `prepared_endpoints` MUST be the same table the source's prepared bindings
/// were built over, so the dispatcher resolves each turn's dense endpoint key to
/// the same configured endpoint (dialect, response field, polling policy).
#[allow(clippy::too_many_arguments)]
pub async fn run_single_turn_dataset_online(
    base_url: String,
    model: String,
    conversations: Box<dyn ConversationSource>,
    concurrency: usize,
    http2: bool,
    record_processors: Vec<Rc<dyn TurnRecordProcessor>>,
    prepared_endpoints: Rc<PreparedEndpointTable>,
) -> anyhow::Result<ScheduledRunReport> {
    let base_urls = parse_base_urls(&base_url)?;
    let clock: Rc<dyn Clock> = RealClock::new();
    let start_ns = clock.now_ns();
    let dispatcher: Rc<dyn TurnDispatcher> = Rc::new(
        TransportSink::new_multi(clock.clone(), start_ns, &base_urls, model, http2)?
            .with_wire_response_capture(false)
            .with_prepared_endpoints(prepared_endpoints),
    );
    let workload: Rc<dyn Workload> =
        Rc::new(SingleTurnDatasetWorkload::new(conversations, concurrency)?);
    run_scheduled_workload_with_processors(
        workload,
        clock,
        start_ns,
        dispatcher,
        StopConfig::default(),
        false,
        ScheduledAncillaryPolicies::default(),
        record_processors,
    )
    .await
}
