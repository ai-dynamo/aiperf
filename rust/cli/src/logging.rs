// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Structured `tracing` setup for the `aiperf` binary.
//!
//! - The default level is INFO. `--extra-verbose` selects TRACE,
//!   `--verbose`/`-v` selects DEBUG, and `--log-level <lvl>` selects an explicit
//!   level. `AIPERF_LOG` overrides these flags.
//! - Console output goes to stderr because stdout is the JSONL protocol channel.
//! - **File**: every line is also written to `<artifact_dir>/logs/aiperf.log`
//!   after [`set_log_file`] is called. The parent owns the file and forwards
//!   child stderr through its subscriber.
//! - `AIPERF_LOG` propagates the resolved directive to the re-exec child.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::Path;
use std::sync::{Mutex, OnceLock};

use tracing_subscriber::EnvFilter;
use tracing_subscriber::fmt::MakeWriter;
use tracing_subscriber::fmt::time::FormatTime;
use tracing_subscriber::prelude::*;

/// Environment variable carrying the `tracing` filter directive.
pub const LOG_ENV: &str = "AIPERF_LOG";

static RESOLVED_DIRECTIVE: OnceLock<String> = OnceLock::new();

static LOG_FILE: OnceLock<Mutex<File>> = OnceLock::new();

/// Install the process-wide `tracing` subscriber before dispatch.
pub fn init(argv: &[String]) {
    let directive = std::env::var(LOG_ENV)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| level_directive_from_argv(argv));
    let _ = RESOLVED_DIRECTIVE.set(directive.clone());

    // Invalid operator directives fall back to INFO so startup can report errors.
    let env_filter =
        EnvFilter::try_new(&directive).unwrap_or_else(|_| EnvFilter::new(DEFAULT_LEVEL));

    let console_layer = tracing_subscriber::fmt::layer()
        .with_writer(io::stderr)
        .with_ansi(false)
        .with_target(false)
        .with_timer(LocalTime);

    let file_layer = tracing_subscriber::fmt::layer()
        .with_writer(LogFileMakeWriter)
        .with_ansi(false)
        .with_target(true)
        .with_timer(LocalDateTime);

    tracing_subscriber::registry()
        .with(env_filter)
        .with(console_layer)
        .with(file_layer)
        .init();
}

/// Return the resolved filter directive for execution children.
pub fn current_directive() -> String {
    RESOLVED_DIRECTIVE
        .get()
        .cloned()
        .unwrap_or_else(|| DEFAULT_LEVEL.to_owned())
}

/// Add best-effort logging to `<artifact_dir>/logs/aiperf.log`.
pub fn set_log_file(artifact_dir: &Path) {
    if LOG_FILE.get().is_some() {
        return;
    }
    let log_dir = artifact_dir.join("logs");
    if let Err(error) = fs::create_dir_all(&log_dir) {
        tracing::warn!(dir = %log_dir.display(), %error, "could not create log folder");
        return;
    }
    let log_path = log_dir.join("aiperf.log");
    match OpenOptions::new().create(true).append(true).open(&log_path) {
        Ok(file) => {
            let _ = LOG_FILE.set(Mutex::new(file));
            tracing::debug!(path = %log_path.display(), "file logging initialized");
        }
        Err(error) => {
            tracing::warn!(path = %log_path.display(), %error, "could not open log file");
        }
    }
}

const DEFAULT_LEVEL: &str = "info";

/// Targets the verbose flags raise, leaving dependencies at [`DEP_LEVEL`].
///
/// A bare `trace`/`debug` directive is global, and some dependencies are
/// pathologically verbose at those levels: `rustls` dumps every handshake message
/// field-by-field and `tokenizers` logs per-character alignment for every prompt it
/// encodes. Because the parent re-emits each forwarded child line through its own
/// subscriber, that output is written twice and a five-request run produces tens of
/// millions of lines — enough that the run never reaches its terminal JSON line.
/// Operators who do want dependency traces can still set `AIPERF_LOG`, which
/// overrides this entirely.
const VERBOSE_TARGETS: [&str; 4] = ["aiperf", "aiperf_cli", "aiperf_runtime", "aiperf_e2e_tests"];

/// The level dependencies keep when a verbose flag raises AIPerf's own targets.
const DEP_LEVEL: &str = "warn";

/// A directive setting dependencies to [`DEP_LEVEL`] and AIPerf targets to `level`.
fn scoped_directive(level: &str) -> String {
    let mut directive = String::from(DEP_LEVEL);
    for target in VERBOSE_TARGETS {
        directive.push(',');
        directive.push_str(target);
        directive.push('=');
        directive.push_str(level);
    }
    directive
}

fn level_directive_from_argv(argv: &[String]) -> String {
    if argv
        .iter()
        .any(|arg| arg == "--extra-verbose" || arg == "-vv" || arg == "--vv")
    {
        return scoped_directive("trace");
    }
    if argv.iter().any(|arg| arg == "--verbose" || arg == "-v") {
        return scoped_directive("debug");
    }
    if let Some(level) = option_value(argv, "--log-level") {
        let level = map_log_level(&level);
        // `--log-level trace|debug` is the same request as the verbose flags and
        // carries the same dependency-flood risk; scope it identically.
        return match level {
            "trace" | "debug" => scoped_directive(level),
            _ => level.to_owned(),
        };
    }
    DEFAULT_LEVEL.to_owned()
}

fn option_value(argv: &[String], flag: &str) -> Option<String> {
    let mut iter = argv.iter();
    while let Some(arg) = iter.next() {
        if arg == flag {
            return iter.next().cloned();
        }
        if let Some(value) = arg
            .strip_prefix(flag)
            .and_then(|rest| rest.strip_prefix('='))
        {
            return Some(value.to_owned());
        }
    }
    None
}

fn map_log_level(level: &str) -> &'static str {
    match level.trim().to_ascii_lowercase().as_str() {
        "trace" => "trace",
        "debug" => "debug",
        "info" | "notice" => "info",
        "warning" | "warn" | "success" => "warn",
        "error" | "critical" => "error",
        _ => DEFAULT_LEVEL,
    }
}

struct LocalTime;

impl FormatTime for LocalTime {
    fn format_time(&self, w: &mut tracing_subscriber::fmt::format::Writer<'_>) -> std::fmt::Result {
        write!(w, "{}", chrono::Local::now().format("%H:%M:%S%.3f"))
    }
}

struct LocalDateTime;

impl FormatTime for LocalDateTime {
    fn format_time(&self, w: &mut tracing_subscriber::fmt::format::Writer<'_>) -> std::fmt::Result {
        write!(
            w,
            "{}",
            chrono::Local::now().format("%Y-%m-%d %H:%M:%S%.3f")
        )
    }
}

/// Writer factory that discards events until the log file is bound.
struct LogFileMakeWriter;

impl<'a> MakeWriter<'a> for LogFileMakeWriter {
    type Writer = LogFileWriter;

    fn make_writer(&'a self) -> Self::Writer {
        LogFileWriter
    }
}

struct LogFileWriter;

impl Write for LogFileWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if let Some(file) = LOG_FILE.get()
            && let Ok(mut guard) = file.lock()
        {
            let _ = guard.write_all(buf);
        }
        // The optional file sink must not fail the tracing subscriber.
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        if let Some(file) = LOG_FILE.get()
            && let Ok(mut guard) = file.lock()
        {
            let _ = guard.flush();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extra_verbose_outranks_verbose_and_log_level() {
        let argv = vec![
            "profile".to_owned(),
            "--verbose".to_owned(),
            "--extra-verbose".to_owned(),
            "--log-level".to_owned(),
            "error".to_owned(),
        ];
        assert_eq!(level_directive_from_argv(&argv), scoped_directive("trace"));
    }

    #[test]
    fn verbose_outranks_log_level() {
        let argv = vec![
            "profile".to_owned(),
            "--log-level".to_owned(),
            "error".to_owned(),
            "-v".to_owned(),
        ];
        assert_eq!(level_directive_from_argv(&argv), scoped_directive("debug"));
    }

    #[test]
    fn log_level_named_and_equals_forms() {
        let spaced = vec![
            "profile".to_owned(),
            "--log-level".to_owned(),
            "warning".to_owned(),
        ];
        assert_eq!(level_directive_from_argv(&spaced), "warn");
        let equals = vec!["profile".to_owned(), "--log-level=debug".to_owned()];
        assert_eq!(
            level_directive_from_argv(&equals),
            scoped_directive("debug")
        );
    }

    /// The verbose flags must not raise dependency crates.
    ///
    /// `rustls` and `tokenizers` at TRACE emit millions of lines for a handful of
    /// requests, and the parent duplicates every forwarded child line, which is enough
    /// to stall a run past any timeout. Asserting the shape of the directive is the
    /// cheap proxy for that behavior.
    #[test]
    fn verbose_flags_leave_dependencies_at_warn() {
        let directive = scoped_directive("trace");
        assert!(
            directive.starts_with("warn,"),
            "dependencies must stay at warn: {directive}"
        );
        assert!(directive.contains("aiperf_runtime=trace"), "{directive}");
        assert!(
            !directive.split(',').any(|part| part == "trace"),
            "no bare global trace directive: {directive}"
        );
    }

    #[test]
    fn default_is_info() {
        let argv = vec!["profile".to_owned(), "--model".to_owned(), "m".to_owned()];
        assert_eq!(level_directive_from_argv(&argv), "info");
    }

    #[test]
    fn notice_and_success_collapse_to_nearest_tracing_level() {
        assert_eq!(map_log_level("NOTICE"), "info");
        assert_eq!(map_log_level("success"), "warn");
        assert_eq!(map_log_level("critical"), "error");
        assert_eq!(map_log_level("bogus"), "info");
    }
}
