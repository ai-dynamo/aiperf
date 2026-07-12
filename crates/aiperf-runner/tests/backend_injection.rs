// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Proof that a complete run can replace native HTTP placement without
//! changing the coordinator's workload, phases, dispatcher, metrics, or report.

use std::cell::Cell;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use aiperf::http::{HttpTurnDispatchResult, HttpTurnExecutionBackend, PreparedHttpTurn};
use aiperf::multiturn::TurnToSend;
use aiperf::scheduled::{ModelResponseMetadata, TurnDispatchOutcome};
use aiperf_metrics::{HttpTrace, InferenceDimensions};
use aiperf_runner::{
    HttpExecutionBackendConfig, HttpExecutionBackendFactory, RunRequest,
    execute_run_with_backend_factory,
};
use aiperf_transport::models::RequestRecord;
use anyhow::{Result, ensure};
use async_trait::async_trait;
use loadgen_core::collector::ReplayTerminalStatus;
use loadgen_core::sink::{ObservedUsage, RequestObserver};

struct FakeRemoteFactory {
    calls: Arc<AtomicUsize>,
    dimension_calls: Arc<AtomicUsize>,
}

impl HttpExecutionBackendFactory for FakeRemoteFactory {
    fn build(
        &self,
        config: HttpExecutionBackendConfig,
    ) -> Result<Rc<dyn HttpTurnExecutionBackend>> {
        ensure!(config.workers == 3, "resolved worker count reached factory");
        Ok(Rc::new(FakeRemoteBackend {
            calls: self.calls.clone(),
            dimension_calls: self.dimension_calls.clone(),
            run_origin_ns: Cell::new(None),
            model: config.model,
        }))
    }
}

struct FakeRemoteBackend {
    calls: Arc<AtomicUsize>,
    dimension_calls: Arc<AtomicUsize>,
    run_origin_ns: Cell<Option<i64>>,
    model: String,
}

#[async_trait(?Send)]
impl HttpTurnExecutionBackend for FakeRemoteBackend {
    fn set_run_origin(&self, start_ns: i64) -> Result<()> {
        ensure!(self.run_origin_ns.replace(Some(start_ns)).is_none());
        Ok(())
    }

    fn inference_dimensions(&self, turn: &TurnToSend) -> InferenceDimensions {
        self.dimension_calls.fetch_add(1, Ordering::SeqCst);
        InferenceDimensions {
            endpoint_url: Some("zmq://remote-worker".into()),
            model: turn
                .effective_model
                .clone()
                .or_else(|| Some(self.model.clone())),
        }
    }

    async fn execute_turn(
        &self,
        turn: PreparedHttpTurn,
        observer: &dyn RequestObserver,
        on_first_token: &dyn Fn(i64),
    ) -> Result<HttpTurnDispatchResult> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let start_ns = self
            .run_origin_ns
            .get()
            .expect("run origin is configured before dispatch");
        let uuid = turn.request.uuid;
        observer.on_admit(uuid, 0.0, 0);
        on_first_token(100_000);
        observer.on_token(uuid, 0.1);
        observer.on_usage(
            uuid,
            ObservedUsage {
                prompt_tokens: Some(turn.request.input_length),
                completion_tokens: Some(1),
                ..ObservedUsage::default()
            },
        );
        observer.on_terminal(uuid, ReplayTerminalStatus::Completed);

        let request_payload = turn.request.request_body_bytes.clone().unwrap_or_default();
        Ok(HttpTurnDispatchResult {
            outcome: TurnDispatchOutcome {
                start_ns,
                end_ns: start_ns + 200_000,
                terminal: ReplayTerminalStatus::Completed,
                response_text: "remote response".into(),
                model_response: ModelResponseMetadata::default(),
                prompt_tokens: Some(turn.request.input_length as u64),
                completion_tokens: Some(1),
                http: HttpTrace::default(),
            },
            request_payload: request_payload.clone(),
            record: RequestRecord {
                start_ns,
                request_body: request_payload,
                request_headers: turn.request.headers,
                end_ns: Some(start_ns + 200_000),
                status: Some(200),
                ..RequestRecord::default()
            },
        })
    }
}

#[test]
fn complete_run_uses_injected_remote_backend_without_native_transport() {
    let artifacts = tempfile::tempdir().unwrap();
    let request: RunRequest = serde_json::from_value(serde_json::json!({
        "protocol_version": 1,
        "run": {
            "benchmark_id": "remote-backend-proof",
            "random_seed": 17,
            "workers": 3,
            "artifact_dir": artifacts.path(),
            "models": {
                "strategy": "round_robin",
                "items": [{"name": "remote-model"}]
            },
            "endpoint": {
                "urls": ["http://must-not-be-contacted.invalid"],
                "type": "chat",
                "streaming": true,
                "use_server_token_count": true
            },
            "dataset": {
                "type": "synthetic",
                "entries": 3,
                "prompts": {
                    "isl": {"value": 4.0},
                    "osl": {"value": 1.0},
                    "batch_size": 1
                },
                "turns": {"value": 1.0},
                "turn_delay_ms": {"value": 0.0},
                "turn_delay_ratio": 1.0
            },
            "phases": [{
                "type": "concurrency",
                "name": "profiling",
                "exclude_from_results": false,
                "requests": 3,
                "concurrency": 2
            }],
            "metrics": {},
            "artifacts": {}
        }
    }))
    .unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let dimension_calls = Arc::new(AtomicUsize::new(0));
    let terminal = execute_run_with_backend_factory(
        request,
        &FakeRemoteFactory {
            calls: calls.clone(),
            dimension_calls: dimension_calls.clone(),
        },
    )
    .unwrap();

    assert!(terminal.success);
    assert_eq!(calls.load(Ordering::SeqCst), 3);
    assert_eq!(dimension_calls.load(Ordering::SeqCst), 3);
    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(artifacts.path().join("native-v2.json")).unwrap())
            .unwrap();
    assert_eq!(
        report["metrics"]["request_count"]["series"][0]["stats"]["total"],
        3.0
    );
    assert_eq!(
        report["metrics"]["total_output_tokens"]["series"][0]["stats"]["value"],
        3.0
    );
}
