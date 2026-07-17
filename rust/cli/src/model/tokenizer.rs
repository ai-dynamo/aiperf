// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Typed tokenizer acquisition policy.
//!
//! When unspecified, the tokenizer name comes from the primary model, or is
//! `"builtin"` for a synthetic model name.

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
