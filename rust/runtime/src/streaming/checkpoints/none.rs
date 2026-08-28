// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Checkpoint-free backend selected by checkpoint mode `none`.
//!
//! Selecting `none` is an explicit statement that the run keeps no durable
//! state: there is nothing to resume from, so `open_latest` reports the absence
//! of a head rather than an error, and `begin_generation` refuses outright
//! instead of silently discarding a publication a caller believed had landed.

use async_trait::async_trait;

use crate::streaming::{
    checkpoint::{CheckpointError, StreamRunIdentity},
    checkpoint_backend::{
        CheckpointGenerationExpectations, CurrentV4CheckpointGeneration,
        LeasedCheckpointGeneration, StreamingCheckpointBackend, StreamingGenerationTransaction,
    },
};

/// Backend that stores nothing and publishes no generation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct NoneCheckpointBackend;

impl NoneCheckpointBackend {
    /// Construct the stateless checkpoint-free backend.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

#[async_trait(?Send)]
impl StreamingCheckpointBackend for NoneCheckpointBackend {
    /// Report no committed head. This is the absence of a resume claim, not a
    /// failure: a `none` run always starts from the beginning of its stream.
    async fn open_latest(
        &self,
        _run: &StreamRunIdentity,
        _expected: &CheckpointGenerationExpectations,
    ) -> Result<Option<LeasedCheckpointGeneration>, CheckpointError> {
        Ok(None)
    }

    /// Refuse publication. A caller reaching this point believes checkpoint
    /// coordination is active, and the mismatch must surface rather than be
    /// absorbed by a transaction that drops everything staged into it.
    async fn begin_generation(
        &self,
        _run: StreamRunIdentity,
        _expected: Option<CurrentV4CheckpointGeneration>,
        _expectations: CheckpointGenerationExpectations,
    ) -> Result<Box<dyn StreamingGenerationTransaction>, CheckpointError> {
        Err(CheckpointError::Storage {
            message: "checkpoint backend \"none\" publishes no generation".to_string(),
        })
    }
}
