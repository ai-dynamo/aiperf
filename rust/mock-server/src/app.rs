// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Axum router construction.

use std::sync::Arc;

use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::routing::{get, post};

use crate::config::MockServerConfig;
use crate::handlers;
pub use crate::state::AppState;

pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/", get(handlers::root_info))
        .route("/health", get(handlers::health))
        // OpenAI-compatible model listing.
        .route("/v1/models", get(handlers::list_models))
        .route("/v1/models/{id}", get(handlers::get_model))
        // LLM endpoints
        .route("/v1/chat/completions", post(handlers::chat_completions))
        .route("/v1/messages", post(handlers::messages))
        .route("/v1/completions", post(handlers::text_completions))
        .route("/v1/embeddings", post(handlers::embeddings))
        // Rerank / ranking endpoints
        .route("/v1/ranking", post(handlers::nim_ranking))
        .route("/rerank", post(handlers::hf_tei_rerank))
        .route("/v2/rerank", post(handlers::cohere_rerank))
        // TGI
        .route("/generate", post(handlers::tgi_generate))
        .route("/generate_stream", post(handlers::tgi_generate_stream))
        // Image endpoints
        .route("/v1/images/generations", post(handlers::image_generation))
        .route("/v1/images/edits", post(handlers::image_edit))
        .route("/v1/image/infer", post(handlers::image_retrieval))
        // Custom endpoints
        .route("/v1/custom-multimodal", post(handlers::custom_multimodal))
        .route("/rag/api/prompt", post(handlers::solido_rag))
        // Live accuracy tally for `--accuracy-dataset` runs.
        .route("/accuracy", get(handlers::accuracy_status))
        // Metrics
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
        // DCGM - two explicit routes (Python's FastAPI regex pattern isn't idiomatic in axum
        // and would conflict with the other `/xxx/metrics` routes above).
        .route("/dcgm1/metrics", get(handlers::dcgm_metrics_1))
        .route("/dcgm2/metrics", get(handlers::dcgm_metrics_2))
        // This is a load-test mock: accept arbitrarily large request bodies.
        // axum's Json/Bytes extractors otherwise cap bodies at a 2 MiB default,
        // which rejects large-ISL prompts (a 1M-token prompt is several MB of
        // JSON) with `413 Payload Too Large` before the handler ever runs.
        .layer(DefaultBodyLimit::disable())
        .with_state(state)
}

pub fn build_state(config: MockServerConfig) -> Arc<AppState> {
    AppState::build(config)
}
