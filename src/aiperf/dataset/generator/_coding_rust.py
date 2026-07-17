# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rust code-template generators (mixin for CodingContentGenerator).

Extracted from ``coding_content.py`` to keep that module under the
ergonomics file-size cap. Methods read ``self._template_rng`` and the
shared vocabulary tuples; behavior is unchanged.
"""

from __future__ import annotations

from aiperf.dataset.generator._coding_vocab import (
    _CLASSES,
    _ERROR_MESSAGES,
    _METHODS,
    _MODULES,
    _RUST_CRATES,
    _VARS,
)


class _RustMixin:
    def _gen_rust_code(self) -> str:
        return self._template_rng.choice(
            [
                self._gen_rust_struct,
                self._gen_rust_http_handler,
                self._gen_rust_errors,
                self._gen_rust_test,
            ]
        )()

    def _gen_rust_struct(self) -> str:
        r = self._template_rng
        cr1, cr2 = r.sample(list(_RUST_CRATES), 2)
        cls = r.choice(_CLASSES)
        m1, m2 = r.sample(_METHODS, 2)
        v1, v2, v3 = r.sample(_VARS, 3)
        err = r.choice(_ERROR_MESSAGES)

        return f"""\
use {cr1};
use {cr2};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct {cls} {{{{
    {v1}: String,
    {v2}: Vec<u8>,
    #[serde(default)]
    {v3}: Option<u64>,
    initialized: bool,
}}}}

impl {cls} {{{{
    pub fn new({v1}: impl Into<String>) -> Self {{{{
        Self {{{{
            {v1}: {v1}.into(),
            {v2}: Vec::new(),
            {v3}: None,
            initialized: false,
        }}}}
    }}}}

    pub async fn {m1}(&mut self) -> Result<(), anyhow::Error> {{{{
        if !self.initialized {{{{
            anyhow::bail!("{err}");
        }}}}
        self.{m2}().await
    }}}}

    async fn {m2}(&self) -> Result<(), anyhow::Error> {{{{
        let _{v2} = self.{v1}.as_bytes();
        tracing::debug!("{m2} completed for {{}}", self.{v1});
        Ok(())
    }}}}
}}}}
"""

    def _gen_rust_http_handler(self) -> str:
        r = self._template_rng
        cls = r.choice(_CLASSES)
        m1, m2 = r.sample(_METHODS, 2)
        v1, v2 = r.sample(_VARS, 2)
        mod = r.choice(_MODULES)

        return f"""\
use axum::{{extract::{{Path, State}}, http::StatusCode, Json}};
use serde::{{Deserialize, Serialize}};
use std::sync::Arc;

use crate::{mod}::{cls};

#[derive(Debug, Deserialize)]
pub struct {m1.title()}Request {{{{
    {v1}: String,
    {v2}: Option<i64>,
}}}}

#[derive(Debug, Serialize)]
pub struct {m1.title()}Response {{{{
    id: String,
    {v1}: String,
    created: bool,
}}}}

pub async fn {m1}_handler(
    State(svc): State<Arc<{cls}>>,
    Json(body): Json<{m1.title()}Request>,
) -> Result<Json<{m1.title()}Response>, StatusCode> {{{{
    let result = svc
        .{m1}(&body.{v1})
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok(Json({m1.title()}Response {{{{
        id: result.id.to_string(),
        {v1}: body.{v1},
        created: true,
    }}}}))
}}}}

pub async fn {m2}_handler(
    State(svc): State<Arc<{cls}>>,
    Path({v1}): Path<String>,
) -> Result<Json<serde_json::Value>, StatusCode> {{{{
    svc.{m2}(&{v1})
        .await
        .map(|v| Json(serde_json::json!({{{{"status": "ok", "data": v}}}})))
        .map_err(|_| StatusCode::NOT_FOUND)
}}}}
"""

    def _gen_rust_errors(self) -> str:
        r = self._template_rng
        cls = r.choice(_CLASSES)
        e1, e2, e3 = r.sample(_ERROR_MESSAGES, 3)
        v1 = r.choice(_VARS)
        mod = r.choice(_MODULES)

        return f"""\
use thiserror::Error;

#[derive(Debug, Error)]
pub enum {cls}Error {{{{
    #[error("{e1}")]
    NotInitialized,

    #[error("{e2}: {{{{{v1}}}}}")]
    InvalidInput {{{{ {v1}: String }}}},

    #[error("{e3}")]
    Internal(#[from] anyhow::Error),

    #[error("io error in {mod}")]
    Io(#[from] std::io::Error),

    #[error("serialization failed")]
    Serde(#[from] serde_json::Error),
}}}}

impl {cls}Error {{{{
    pub fn is_retryable(&self) -> bool {{{{
        matches!(self, Self::Internal(_) | Self::Io(_))
    }}}}

    pub fn status_code(&self) -> u16 {{{{
        match self {{{{
            Self::NotInitialized => 503,
            Self::InvalidInput {{{{ .. }}}} => 400,
            Self::Internal(_) => 500,
            Self::Io(_) => 502,
            Self::Serde(_) => 422,
        }}}}
    }}}}
}}}}
"""

    def _gen_rust_test(self) -> str:
        r = self._template_rng
        cls = r.choice(_CLASSES)
        m1, m2 = r.sample(_METHODS, 2)
        v1, v2 = r.sample(_VARS, 2)
        err = r.choice(_ERROR_MESSAGES)
        cr = r.choice(_RUST_CRATES)

        return f"""\
use {cr};

#[cfg(test)]
mod tests {{{{
    use super::*;

    fn make_{cls.lower()}() -> {cls} {{{{
        {cls}::new("{v1}_test")
    }}}}

    #[tokio::test]
    async fn test_{m1}_success() {{{{
        let mut svc = make_{cls.lower()}();
        svc.initialized = true;
        let result = svc.{m1}().await;
        assert!(result.is_ok(), "expected Ok, got {{:?}}", result);
    }}}}

    #[tokio::test]
    async fn test_{m1}_not_initialized() {{{{
        let mut svc = make_{cls.lower()}();
        let err = svc.{m1}().await.unwrap_err();
        assert!(err.to_string().contains("{err}"));
    }}}}

    #[test]
    fn test_{m2}_returns_bytes() {{{{
        let svc = make_{cls.lower()}();
        let {v2} = svc.{v1}.as_bytes();
        assert!(!{v2}.is_empty());
    }}}}

    #[tokio::test]
    async fn test_{m1}_concurrent() {{{{
        let svc = std::sync::Arc::new(tokio::sync::Mutex::new(make_{cls.lower()}()));
        let mut handles = vec![];
        for _ in 0..5 {{{{
            let svc = svc.clone();
            handles.push(tokio::spawn(async move {{{{
                svc.lock().await.{m1}().await
            }}}}));
        }}}}
        for h in handles {{{{
            let _ = h.await.unwrap();
        }}}}
    }}}}
}}}}
"""
