// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Console logging setup matching the ai-dynamo readable log format.
//!
//! A compact `tracing_subscriber` fmt layer writing to **stderr** with UTC
//! ISO-8601 microsecond timestamps (`%Y-%m-%dT%H:%M:%S%.6fZ`) and an `EnvFilter`
//! whose default level is `info`. Mirrors `dynamo`'s readable formatter so log
//! lines are visually identical across the two tools.
//!
//! Configuration (env vars, matching the `dynamo` `DYN_*` scheme with an
//! `AIPERF_*` namespace):
//! - `AIPERF_LOG` — level filter directives (e.g. `debug`, `aiperf=trace`).
//!   Falls back to `RUST_LOG`. Default `info`.
//! - `AIPERF_LOGGING_JSONL=1` — emit one JSON object per line instead of the
//!   readable format.
//! - `NO_COLOR` (or a non-tty stderr) disables ANSI colors.

use std::io::IsTerminal;
use std::sync::Once;

use tracing_subscriber::EnvFilter;
use tracing_subscriber::fmt::format::Writer;
use tracing_subscriber::fmt::time::FormatTime;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{Layer, fmt};

static INIT: Once = Once::new();

/// UTC ISO-8601 timestamp with microsecond precision — byte-identical to the
/// ai-dynamo readable log timestamp.
struct TimeFormatter;

impl FormatTime for TimeFormatter {
    fn format_time(&self, w: &mut Writer<'_>) -> std::fmt::Result {
        write!(w, "{}", chrono::Utc::now().format("%Y-%m-%dT%H:%M:%S%.6fZ"))
    }
}

/// Build the level filter: default `info`, read from `AIPERF_LOG` (then
/// `RUST_LOG`), with the noisy transport/runtime crates pinned to `error` — the
/// same suppression set `dynamo` applies by default.
fn env_filter() -> EnvFilter {
    let base = if std::env::var_os("AIPERF_LOG").is_some() {
        EnvFilter::builder()
            .with_default_directive(tracing::level_filters::LevelFilter::INFO.into())
            .with_env_var("AIPERF_LOG")
            .from_env_lossy()
    } else {
        // Fall back to RUST_LOG, then default to `info`.
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"))
    };
    [
        "h2=error",
        "hyper=error",
        "hyper_util=error",
        "rustls=error",
    ]
    .into_iter()
    .fold(base, |f, d| f.add_directive(d.parse().unwrap()))
}

/// Initialize the global logger. Idempotent; safe to call more than once.
pub fn init() {
    INIT.call_once(|| {
        let jsonl = std::env::var("AIPERF_LOGGING_JSONL")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let ansi = std::io::stderr().is_terminal() && std::env::var_os("NO_COLOR").is_none();

        if jsonl {
            let layer = fmt::layer()
                .with_ansi(false)
                .with_timer(TimeFormatter)
                .with_writer(std::io::stderr)
                .json()
                .with_filter(env_filter());
            tracing_subscriber::registry().with(layer).init();
        } else {
            let layer = fmt::layer()
                .event_format(fmt::format().compact().with_timer(TimeFormatter))
                .with_ansi(ansi)
                .with_writer(std::io::stderr)
                .with_filter(env_filter());
            tracing_subscriber::registry().with(layer).init();
        }
    });
}
