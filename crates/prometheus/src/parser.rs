// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Strict bounded exposition parser implementation.

use std::fmt::Debug;

use crate::{Exposition, ExpositionFormat, ParseError, ParseLimits};
use crate::{semantic::assemble_exposition, syntax::parse_document};

/// Object-safe, IO-free parser seam for advertised exposition grammars.
pub trait ExpositionParser: Debug + Send + Sync {
    /// Parses one complete decoded entity atomically under exactly `format`.
    fn parse(
        &self,
        format: ExpositionFormat,
        exact_body: &[u8],
        limits: &ParseLimits,
    ) -> Result<Exposition, ParseError>;
}

/// Strict Prometheus 0.0.4 and OpenMetrics 1.0.0 parser.
#[derive(Debug, Default, Clone, Copy)]
pub struct StrictExpositionParser;

impl ExpositionParser for StrictExpositionParser {
    fn parse(
        &self,
        format: ExpositionFormat,
        exact_body: &[u8],
        limits: &ParseLimits,
    ) -> Result<Exposition, ParseError> {
        assemble_exposition(format, parse_document(format, exact_body, limits)?, limits)
    }
}
