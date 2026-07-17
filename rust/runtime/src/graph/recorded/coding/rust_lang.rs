// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Rust code-template renderers.

use super::templates::TemplateRenderer;
use super::vocab::*;
use crate::graph::recorded::RecordedTraceError;

/// Dispatch across the Rust structural variants.
pub(super) fn render(r: &mut TemplateRenderer) -> Result<String, RecordedTraceError> {
    match r.index(4)? {
        0 => rust_struct(r),
        1 => rust_http_handler(r),
        2 => rust_errors(r),
        _ => rust_test(r),
    }
}

fn rust_struct(r: &mut TemplateRenderer) -> Result<String, RecordedTraceError> {
    let cr = r.sample(RUST_CRATES, 2)?;
    let (cr1, cr2) = (cr[0], cr[1]);
    let cls = r.pick(CLASSES)?;
    let m = r.sample(METHODS, 2)?;
    let (m1, m2) = (m[0], m[1]);
    let v = r.sample(VARS, 3)?;
    let (v1, v2, v3) = (v[0], v[1], v[2]);
    let err = r.pick(ERRORS)?;

    Ok(format!(
        r##"use {cr1};
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
"##
    ))
}

fn rust_http_handler(r: &mut TemplateRenderer) -> Result<String, RecordedTraceError> {
    let cls = r.pick(CLASSES)?;
    let m = r.sample(METHODS, 2)?;
    let (m1, m2) = (m[0], m[1]);
    let v = r.sample(VARS, 2)?;
    let (v1, v2) = (v[0], v[1]);
    let mod_ = r.pick(MODULES)?;
    let m1_title = TemplateRenderer::title_case(m1);

    Ok(format!(
        r##"use axum::{{extract::{{Path, State}}, http::StatusCode, Json}};
use serde::{{Deserialize, Serialize}};
use std::sync::Arc;

use crate::{mod_}::{cls};

#[derive(Debug, Deserialize)]
pub struct {m1_title}Request {{{{
    {v1}: String,
    {v2}: Option<i64>,
}}}}

#[derive(Debug, Serialize)]
pub struct {m1_title}Response {{{{
    id: String,
    {v1}: String,
    created: bool,
}}}}

pub async fn {m1}_handler(
    State(svc): State<Arc<{cls}>>,
    Json(body): Json<{m1_title}Request>,
) -> Result<Json<{m1_title}Response>, StatusCode> {{{{
    let result = svc
        .{m1}(&body.{v1})
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok(Json({m1_title}Response {{{{
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
"##
    ))
}

fn rust_errors(r: &mut TemplateRenderer) -> Result<String, RecordedTraceError> {
    let cls = r.pick(CLASSES)?;
    let e = r.sample(ERRORS, 3)?;
    let (e1, e2, e3) = (e[0], e[1], e[2]);
    let v1 = r.pick(VARS)?;
    let mod_ = r.pick(MODULES)?;

    Ok(format!(
        r##"use thiserror::Error;

#[derive(Debug, Error)]
pub enum {cls}Error {{{{
    #[error("{e1}")]
    NotInitialized,

    #[error("{e2}: {{{{{v1}}}}}")]
    InvalidInput {{{{ {v1}: String }}}},

    #[error("{e3}")]
    Internal(#[from] anyhow::Error),

    #[error("io error in {mod_}")]
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
"##
    ))
}

fn rust_test(r: &mut TemplateRenderer) -> Result<String, RecordedTraceError> {
    let cls = r.pick(CLASSES)?;
    let m = r.sample(METHODS, 2)?;
    let (m1, m2) = (m[0], m[1]);
    let v = r.sample(VARS, 2)?;
    let (v1, v2) = (v[0], v[1]);
    let err = r.pick(ERRORS)?;
    let cr = r.pick(RUST_CRATES)?;
    let cls_lower = cls.to_lowercase();

    Ok(format!(
        r##"use {cr};

#[cfg(test)]
mod tests {{{{
    use super::*;

    fn make_{cls_lower}() -> {cls} {{{{
        {cls}::new("{v1}_test")
    }}}}

    #[tokio::test]
    async fn test_{m1}_success() {{{{
        let mut svc = make_{cls_lower}();
        svc.initialized = true;
        let result = svc.{m1}().await;
        assert!(result.is_ok(), "expected Ok, got {{:?}}", result);
    }}}}

    #[tokio::test]
    async fn test_{m1}_not_initialized() {{{{
        let mut svc = make_{cls_lower}();
        let err = svc.{m1}().await.unwrap_err();
        assert!(err.to_string().contains("{err}"));
    }}}}

    #[test]
    fn test_{m2}_returns_bytes() {{{{
        let svc = make_{cls_lower}();
        let {v2} = svc.{v1}.as_bytes();
        assert!(!{v2}.is_empty());
    }}}}

    #[tokio::test]
    async fn test_{m1}_concurrent() {{{{
        let svc = std::sync::Arc::new(tokio::sync::Mutex::new(make_{cls_lower}()));
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
"##
    ))
}
