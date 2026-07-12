// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Advertised exposition formats and strict HTTP media-type selection.

use std::collections::BTreeMap;
use std::fmt::{Display, Formatter, Result as FmtResult};

/// One parser grammar advertised by the archive surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExpositionFormat {
    /// Prometheus text exposition format version 0.0.4.
    PrometheusText004,
    /// OpenMetrics text exposition format version 1.0.0.
    OpenMetricsText100,
}

impl ExpositionFormat {
    /// Selects an exact supported grammar from an HTTP `Content-Type` value.
    ///
    /// Selection never retries a body under another grammar. Prometheus 0.0.4
    /// permits omitted `version` and `charset` parameters for compatibility;
    /// OpenMetrics requires both normative parameters.
    pub fn from_content_type(value: &str) -> Result<Self, ContentTypeError> {
        let mut pieces = value.split(';');
        let media_type = pieces
            .next()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or(ContentTypeError::MissingMediaType)?
            .to_ascii_lowercase();
        let mut parameters = BTreeMap::<String, String>::new();
        for raw in pieces {
            let (name, value) = raw
                .split_once('=')
                .ok_or_else(|| ContentTypeError::MalformedParameter(raw.trim().to_string()))?;
            let name = name.trim().to_ascii_lowercase();
            let value = value.trim().trim_matches('"').to_ascii_lowercase();
            if name.is_empty() || value.is_empty() {
                return Err(ContentTypeError::MalformedParameter(raw.trim().to_string()));
            }
            if parameters.insert(name.clone(), value).is_some() {
                return Err(ContentTypeError::DuplicateParameter(name));
            }
        }

        for name in parameters.keys() {
            if name != "version" && name != "charset" {
                return Err(ContentTypeError::UnsupportedParameter(name.clone()));
            }
        }
        if let Some(charset) = parameters.get("charset")
            && charset != "utf-8"
        {
            return Err(ContentTypeError::UnsupportedCharset(charset.clone()));
        }

        match media_type.as_str() {
            "text/plain" => {
                if let Some(version) = parameters.get("version")
                    && version != "0.0.4"
                {
                    return Err(ContentTypeError::UnsupportedVersion {
                        media_type,
                        version: version.clone(),
                    });
                }
                Ok(Self::PrometheusText004)
            }
            "application/openmetrics-text" => {
                let version = parameters
                    .get("version")
                    .ok_or(ContentTypeError::MissingParameter("version"))?;
                if version != "1.0.0" {
                    return Err(ContentTypeError::UnsupportedVersion {
                        media_type,
                        version: version.clone(),
                    });
                }
                if !parameters.contains_key("charset") {
                    return Err(ContentTypeError::MissingParameter("charset"));
                }
                Ok(Self::OpenMetricsText100)
            }
            _ => Err(ContentTypeError::UnsupportedMediaType(media_type)),
        }
    }

    /// Returns the canonical media type emitted for this grammar.
    pub const fn canonical_content_type(self) -> &'static str {
        match self {
            Self::PrometheusText004 => "text/plain; version=0.0.4; charset=utf-8",
            Self::OpenMetricsText100 => {
                "application/openmetrics-text; version=1.0.0; charset=utf-8"
            }
        }
    }
}

/// Failure to select one supported exposition grammar from `Content-Type`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContentTypeError {
    /// The value contained no media type.
    MissingMediaType,
    /// A parameter did not use `name=value` syntax.
    MalformedParameter(String),
    /// A parameter name occurred more than once.
    DuplicateParameter(String),
    /// A parameter is not defined by the supported media type profile.
    UnsupportedParameter(String),
    /// The media type is not an advertised parser grammar.
    UnsupportedMediaType(String),
    /// The selected media type omitted a mandatory parameter.
    MissingParameter(&'static str),
    /// The selected media type advertised an unsupported version.
    UnsupportedVersion {
        /// Lowercase media type.
        media_type: String,
        /// Lowercase authored version.
        version: String,
    },
    /// The selected grammar is not UTF-8.
    UnsupportedCharset(String),
}

impl Display for ContentTypeError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::MissingMediaType => formatter.write_str("Content-Type has no media type"),
            Self::MalformedParameter(value) => {
                write!(formatter, "malformed Content-Type parameter {value:?}")
            }
            Self::DuplicateParameter(name) => {
                write!(formatter, "duplicate Content-Type parameter {name:?}")
            }
            Self::UnsupportedParameter(name) => {
                write!(formatter, "unsupported Content-Type parameter {name:?}")
            }
            Self::UnsupportedMediaType(value) => {
                write!(formatter, "unsupported metrics media type {value:?}")
            }
            Self::MissingParameter(name) => {
                write!(
                    formatter,
                    "Content-Type is missing required parameter {name:?}"
                )
            }
            Self::UnsupportedVersion {
                media_type,
                version,
            } => write!(
                formatter,
                "unsupported version {version:?} for media type {media_type:?}"
            ),
            Self::UnsupportedCharset(value) => {
                write!(formatter, "unsupported metrics charset {value:?}")
            }
        }
    }
}

impl std::error::Error for ContentTypeError {}
