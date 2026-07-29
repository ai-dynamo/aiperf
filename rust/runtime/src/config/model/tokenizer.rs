// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Typed tokenizer acquisition policy.
//!
//! When unspecified, the tokenizer name comes from the primary model, or is
//! `"builtin"` for a synthetic model name.

use serde::{Deserialize, Serialize};

/// The default git ref, matching `resolve.rs`.
fn default_revision() -> String {
    "main".to_string()
}

/// The typed `tokenizer` acquisition policy.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Tokenizer {
    /// Tokenizer identity (a HF model id, a path, or `"builtin"`).
    pub name: String,
    /// Model revision / git ref. `resolve.rs` defaults this to `main`, so an
    /// authored protocol-v2 request omitting it must resolve the same way
    /// rather than hard-rejecting the run.
    #[serde(default = "default_revision")]
    pub revision: String,
    /// Allow executing remote tokenizer code. Defaulted off, matching `load.rs`.
    #[serde(default)]
    pub trust_remote_code: bool,
    /// Apply the chat template when tokenizing. Defaulted off, matching `load.rs`.
    #[serde(default)]
    pub apply_chat_template: bool,
    /// Opt-in server-side tokenizer origin (e.g. `http://host:8000`).
    ///
    /// When set, token counting is offloaded to the inference server's
    /// `/tokenize` and `/detokenize` endpoints and `name` is used only as the
    /// model selector forwarded to the server. Absent by default, keeping the
    /// local built-in / Hugging Face tokenizer in force.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub server_url: Option<String>,
}
