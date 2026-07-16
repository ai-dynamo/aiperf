// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Connection-reuse strategy. Transport-neutral: HTTP applies it to the hyper
//! connection pool and gRPC maps it onto channel reuse.

use serde::{Deserialize, Serialize};

/// How connections are reused across requests. Port of Python
/// `ConnectionReuseStrategy`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ConnectionReuseStrategy {
    /// Shared connection pool (default aiohttp behavior).
    #[default]
    Pooled,
    /// A fresh connection per request, closed after.
    Never,
    /// One connection per user session (correlation id), released on final turn.
    StickyUserSessions,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn connection_reuse_strategy_uses_config_v2_wire_values() {
        assert_eq!(
            serde_json::from_str::<ConnectionReuseStrategy>("\"sticky-user-sessions\"").unwrap(),
            ConnectionReuseStrategy::StickyUserSessions
        );
        assert_eq!(
            serde_json::to_string(&ConnectionReuseStrategy::Never).unwrap(),
            "\"never\""
        );
    }

    #[test]
    fn default_is_pooled() {
        assert_eq!(
            ConnectionReuseStrategy::default(),
            ConnectionReuseStrategy::Pooled
        );
    }
}
