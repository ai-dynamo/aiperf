// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Typed `tokenizer` section of the native `BenchmarkConfig`.
//!
//! Wire shape ported from
//! `src/aiperf/orchestrator/rust_wire.py::_authored_tokenizer_v2`. When no
//! tokenizer is authored, the loader fills `name` from the primary model (or
//! `"builtin"` for a fake model name) and the other fields with these defaults.

use serde::{Deserialize, Serialize};

/// The typed `tokenizer` acquisition policy.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Tokenizer {
    /// Tokenizer identity (a HF model id, a path, or `"builtin"`).
    pub name: String,
    /// Model revision / git ref.
    pub revision: String,
    /// Allow executing remote tokenizer code.
    pub trust_remote_code: bool,
    /// Apply the chat template when tokenizing.
    pub apply_chat_template: bool,
}
