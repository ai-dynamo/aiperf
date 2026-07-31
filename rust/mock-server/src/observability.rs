// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! In-memory OTLP, MLflow, and W&B collector mocks.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use axum::Json;
use axum::body::Bytes;
use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::IntoResponse;
use parking_lot::Mutex;
use serde_json::{Value, json};

use crate::AppState;

/// Captured observability payloads exposed through [`crate::AppState`].
#[derive(Debug, Default)]
pub struct ObservabilityState {
    otlp_metrics: Mutex<Vec<Vec<u8>>>,
    mlflow_experiments: Mutex<BTreeMap<String, String>>,
    mlflow_requests: Mutex<Vec<MlflowRequest>>,
    mlflow_artifacts: Mutex<Vec<CapturedArtifact>>,
    wandb_runs: Mutex<Vec<CapturedWandbRun>>,
    next_experiment: AtomicU64,
    next_run: AtomicU64,
}

/// One captured MLflow JSON request.
#[derive(Clone, Debug)]
pub struct MlflowRequest {
    /// REST route suffix, such as `runs/log-batch`.
    pub route: String,
    /// Parsed request body.
    pub body: Value,
}

/// One captured MLflow artifact upload.
#[derive(Clone, Debug)]
pub struct CapturedArtifact {
    /// Artifact path below the MLflow proxy route.
    pub path: String,
    /// Uploaded bytes.
    pub body: Vec<u8>,
}

/// One AIPerf W&B datastore upload.
#[derive(Clone, Debug)]
pub struct CapturedWandbRun {
    /// W&B entity supplied by the exporter.
    pub entity: Option<String>,
    /// W&B project supplied by the exporter.
    pub project: String,
    /// W&B run id supplied by the exporter.
    pub run_id: String,
    /// Complete `.wandb` datastore bytes.
    pub body: Vec<u8>,
}

impl ObservabilityState {
    /// Snapshot raw OTLP/HTTP protobuf requests in arrival order.
    pub fn otlp_metrics(&self) -> Vec<Vec<u8>> {
        self.otlp_metrics.lock().clone()
    }

    /// Snapshot captured MLflow JSON calls.
    pub fn mlflow_requests(&self) -> Vec<MlflowRequest> {
        self.mlflow_requests.lock().clone()
    }

    /// Snapshot captured MLflow artifact uploads.
    pub fn mlflow_artifacts(&self) -> Vec<CapturedArtifact> {
        self.mlflow_artifacts.lock().clone()
    }

    /// Snapshot captured W&B datastore uploads.
    pub fn wandb_runs(&self) -> Vec<CapturedWandbRun> {
        self.wandb_runs.lock().clone()
    }
}

pub(crate) async fn receive_otlp(
    State(state): State<std::sync::Arc<AppState>>,
    body: Bytes,
) -> StatusCode {
    state.observability.otlp_metrics.lock().push(body.to_vec());
    StatusCode::OK
}

pub(crate) async fn mlflow_get_experiment(
    State(state): State<std::sync::Arc<AppState>>,
    Query(query): Query<BTreeMap<String, String>>,
) -> impl IntoResponse {
    let Some(name) = query.get("experiment_name") else {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error_code": "INVALID_PARAMETER_VALUE"})),
        );
    };
    match state.observability.mlflow_experiments.lock().get(name) {
        Some(id) => (
            StatusCode::OK,
            Json(json!({"experiment": {"experiment_id": id, "name": name}})),
        ),
        None => (
            StatusCode::NOT_FOUND,
            Json(json!({"error_code": "RESOURCE_DOES_NOT_EXIST"})),
        ),
    }
}

pub(crate) async fn mlflow_create_experiment(
    State(state): State<std::sync::Arc<AppState>>,
    Json(body): Json<Value>,
) -> impl IntoResponse {
    let Some(name) = body.get("name").and_then(Value::as_str) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error_code": "INVALID_PARAMETER_VALUE"})),
        );
    };
    let mut experiments = state.observability.mlflow_experiments.lock();
    if experiments.contains_key(name) {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error_code": "RESOURCE_ALREADY_EXISTS"})),
        );
    }
    let id = state
        .observability
        .next_experiment
        .fetch_add(1, Ordering::Relaxed)
        .to_string();
    experiments.insert(name.to_string(), id.clone());
    state
        .observability
        .mlflow_requests
        .lock()
        .push(MlflowRequest {
            route: "experiments/create".to_string(),
            body,
        });
    (StatusCode::OK, Json(json!({"experiment_id": id})))
}

pub(crate) async fn mlflow_create_run(
    State(state): State<std::sync::Arc<AppState>>,
    Json(body): Json<Value>,
) -> impl IntoResponse {
    let experiment_id = body
        .get("experiment_id")
        .and_then(Value::as_str)
        .unwrap_or("0");
    let run_id = format!(
        "mock-run-{}",
        state.observability.next_run.fetch_add(1, Ordering::Relaxed)
    );
    state
        .observability
        .mlflow_requests
        .lock()
        .push(MlflowRequest {
            route: "runs/create".to_string(),
            body: body.clone(),
        });
    let artifact_uri = format!("mlflow-artifacts:/{experiment_id}/{run_id}/artifacts");
    (
        StatusCode::OK,
        Json(json!({"run": {"info": {
            "run_id": run_id,
            "run_name": body.get("run_name").and_then(Value::as_str).unwrap_or(""),
            "artifact_uri": artifact_uri
        }}})),
    )
}

async fn capture_mlflow_json(
    state: std::sync::Arc<AppState>,
    route: &'static str,
    body: Value,
) -> impl IntoResponse {
    state
        .observability
        .mlflow_requests
        .lock()
        .push(MlflowRequest {
            route: route.to_string(),
            body,
        });
    (StatusCode::OK, Json(json!({})))
}

pub(crate) async fn mlflow_log_batch(
    State(state): State<std::sync::Arc<AppState>>,
    Json(body): Json<Value>,
) -> impl IntoResponse {
    capture_mlflow_json(state, "runs/log-batch", body).await
}

pub(crate) async fn mlflow_update_run(
    State(state): State<std::sync::Arc<AppState>>,
    Json(body): Json<Value>,
) -> impl IntoResponse {
    capture_mlflow_json(state, "runs/update", body).await
}

pub(crate) async fn mlflow_artifact(
    State(state): State<std::sync::Arc<AppState>>,
    Path(path): Path<String>,
    body: Bytes,
) -> StatusCode {
    state
        .observability
        .mlflow_artifacts
        .lock()
        .push(CapturedArtifact {
            path,
            body: body.to_vec(),
        });
    StatusCode::OK
}

pub(crate) async fn receive_wandb(
    State(state): State<std::sync::Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> StatusCode {
    let header = |name: &'static str| {
        headers
            .get(name)
            .and_then(|value| value.to_str().ok())
            .map(str::to_string)
    };
    let Some(project) = header("x-wandb-project") else {
        return StatusCode::BAD_REQUEST;
    };
    let Some(run_id) = header("x-wandb-run-id") else {
        return StatusCode::BAD_REQUEST;
    };
    state
        .observability
        .wandb_runs
        .lock()
        .push(CapturedWandbRun {
            entity: header("x-wandb-entity"),
            project,
            run_id,
            body: body.to_vec(),
        });
    StatusCode::OK
}
