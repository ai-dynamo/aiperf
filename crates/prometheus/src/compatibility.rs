// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Explicit native-compatibility projection seam.

use std::fmt::{Debug, Display, Formatter, Result as FmtResult};

use crate::model::Exposition;

/// Separately injected projection from the lossless archive model to a native domain record.
///
/// A caller that intentionally runs a classic compatibility grammar does so as
/// a second named parse operation; the strict parser never falls back by
/// itself and this projection never changes its outcome.
pub trait NativeCompatibilityProjection: Debug + Send + Sync {
    /// Native domain record produced by the projection.
    type Output;

    /// Projects one already successful strict exposition.
    fn project(
        &self,
        exposition: &Exposition,
    ) -> Result<Option<Self::Output>, CompatibilityProjectionError>;
}

/// Compatibility projection that deliberately produces no native record.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopNativeCompatibilityProjection;

impl NativeCompatibilityProjection for NoopNativeCompatibilityProjection {
    type Output = ();

    fn project(
        &self,
        _exposition: &Exposition,
    ) -> Result<Option<Self::Output>, CompatibilityProjectionError> {
        Ok(None)
    }
}

/// Typed native-compatibility projection failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompatibilityProjectionError {
    /// Bounded projection detail.
    pub message: String,
}

impl Display for CompatibilityProjectionError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for CompatibilityProjectionError {}
