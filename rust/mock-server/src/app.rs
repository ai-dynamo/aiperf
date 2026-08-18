// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Axum router construction.

use std::sync::Arc;

use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::routing::{get, post};

use crate::config::{MockServerConfig, WebSocketMode};
use crate::handlers;
use crate::observability;
pub use crate::state::AppState;

pub fn build_router(state: Arc<AppState>) -> Router {
    let mut router = Router::new()
        .route("/", get(handlers::root_info))
        .route("/health", get(handlers::health))
        .route("/v1/models", get(handlers::list_models))
        // GET is the OpenAI model-info / KServe v1 readiness route; POST on the
        // same path is the KServe v1 `:predict` inference verb (the model name
        // and `:predict` suffix arrive as one path segment).
        .route(
            "/v1/models/{id}",
            get(handlers::get_model).post(handlers::kserve_v1_predict),
        )
        .route("/v2/models/{model}/infer", post(handlers::kserve_v2_infer))
        .route(
            "/v2/models/{model}/ready",
            get(handlers::kserve_v2_model_ready),
        )
        .route("/v2/health/ready", get(handlers::kserve_v2_health_ready))
        .route("/v1/chat/completions", post(handlers::chat_completions))
        .route("/v1/messages", post(handlers::messages))
        .route("/v1/completions", post(handlers::text_completions))
        .route("/v1/embeddings", post(handlers::embeddings))
        .route("/v1/responses", post(handlers::responses))
        .route("/inference/v1/generate", post(handlers::vllm_generate))
        .route(
            "/endpoints/{endpoint_name}/invocations",
            post(handlers::sagemaker_invoke),
        )
        .route(
            "/endpoints/{endpoint_name}/invocations-response-stream",
            post(handlers::sagemaker_invoke_stream),
        )
        // KServe OpenAI-compatible `/openai/v1/*` aliases: the runner's KServe
        // chat/completions/embeddings factories default to these paths. They
        // dispatch to the identical OpenAI handlers above.
        .route(
            "/openai/v1/chat/completions",
            post(handlers::chat_completions),
        )
        .route("/openai/v1/completions", post(handlers::text_completions))
        .route("/openai/v1/embeddings", post(handlers::embeddings))
        .route("/openai/v1/models", get(handlers::list_models))
        .route("/v1/ranking", post(handlers::nim_ranking))
        .route("/rerank", post(handlers::hf_tei_rerank))
        .route("/v2/rerank", post(handlers::cohere_rerank))
        .route("/generate", post(handlers::tgi_generate))
        .route("/generate_stream", post(handlers::tgi_generate_stream))
        .route("/v1/images/generations", post(handlers::image_generation))
        .route("/v1/images/edits", post(handlers::image_edit))
        .route("/v1/image/infer", post(handlers::image_retrieval))
        // `image_retrieval` defaults to this alias.
        .route("/v1/infer", post(handlers::image_retrieval))
        .route("/v1/custom-multimodal", post(handlers::custom_multimodal))
        .route("/rag/api/prompt", post(handlers::solido_rag))
        .route("/reset_prefix_cache", post(handlers::reset_prefix_cache))
        .route("/start_profile", post(handlers::start_profile))
        .route("/stop_profile", post(handlers::stop_profile))
        .route("/accuracy", get(handlers::accuracy_status))
        .route("/metrics", get(handlers::aiperf_mock_metrics))
        .route("/vllm/metrics", get(handlers::vllm_metrics))
        .route("/sglang/metrics", get(handlers::sglang_metrics))
        .route("/trtllm/metrics", get(handlers::trtllm_metrics))
        .route(
            "/dynamo_frontend/metrics",
            get(handlers::dynamo_frontend_metrics),
        )
        .route(
            "/dynamo_component/prefill/metrics",
            get(handlers::dynamo_prefill_metrics),
        )
        .route(
            "/dynamo_component/decode/metrics",
            get(handlers::dynamo_decode_metrics),
        )
        // Explicit DCGM routes avoid conflicts with the other metrics paths.
        .route("/dcgm1/metrics", get(handlers::dcgm_metrics_1))
        .route("/dcgm2/metrics", get(handlers::dcgm_metrics_2))
        .route("/v1/metrics", post(observability::receive_otlp))
        .route(
            "/api/2.0/mlflow/experiments/get-by-name",
            get(observability::mlflow_get_experiment),
        )
        .route(
            "/api/2.0/mlflow/experiments/create",
            post(observability::mlflow_create_experiment),
        )
        .route(
            "/api/2.0/mlflow/runs/create",
            post(observability::mlflow_create_run),
        )
        .route(
            "/api/2.0/mlflow/runs/log-batch",
            post(observability::mlflow_log_batch),
        )
        .route(
            "/api/2.0/mlflow/runs/update",
            post(observability::mlflow_update_run),
        )
        .route(
            "/api/2.0/mlflow-artifacts/artifacts/{*path}",
            axum::routing::put(observability::mlflow_artifact),
        )
        .route("/api/wandb/runs", post(observability::receive_wandb))
        // Large prompts can exceed axum's 2 MiB default before reaching a handler.
        .layer(DefaultBodyLimit::disable());

    match state.config.websocket_mode {
        WebSocketMode::Disabled => {}
        WebSocketMode::TurnSerialized => {
            router = router.route(
                "/mock/websocket/turns",
                get(crate::websocket::turns_upgrade),
            );
        }
        WebSocketMode::Realtime => {
            router = router.route(
                "/mock/websocket/realtime",
                get(crate::websocket::realtime_upgrade),
            );
        }
        WebSocketMode::Both => {
            router = router
                .route(
                    "/mock/websocket/turns",
                    get(crate::websocket::turns_upgrade),
                )
                .route(
                    "/mock/websocket/realtime",
                    get(crate::websocket::realtime_upgrade),
                );
        }
    }
    if state.config.websocket_enabled() {
        router = router.route("/mock/websocket/captures", get(crate::websocket::captures));
    }
    router.with_state(state)
}

pub fn build_state(config: MockServerConfig) -> Arc<AppState> {
    AppState::build(config)
}
