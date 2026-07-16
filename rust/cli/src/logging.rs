// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Structured `tracing` setup for the native `aiperf` binary, brought to parity
//! with the Python frontend's logging (`src/aiperf/common/logging.py`).
//!
//! Parity contract (see the design under
//! `~/.aiperf/docs/superpowers/specs/2026-07-15-rust-logging-parity-design.md`):
//!
//! - **Default level is INFO** (Python `config/runtime.py::LOG_LEVEL`), not the
//!   old `warn`. `--extra-verbose` → TRACE, `--verbose`/`-v` → DEBUG,
//!   `--log-level <lvl>` → explicit (Python `_converter_runtime`). `AIPERF_LOG`
//!   (an env directive, e.g. `info,aiperf_runtime::foo=debug`) overrides everything.
//! - **Console** goes to stderr (stdout is the runner's JSONL protocol channel),
//!   ANSI off, `HH:MM:SS.mmm LEVEL message` — a close, not byte-exact, match for
//!   Python's basic handler.
//! - **File**: every line is also written to `<artifact_dir>/logs/aiperf.log`
//!   once [`set_log_file`] is called (Python's `FileHandler`). The parent front
//!   door owns the file; the `--execute` child logs only to its (piped) stderr,
//!   which the parent forwards via `tracing::info!("aiperf-runner: …")` — exactly
//!   mirroring Python's `_forward_runner_stderr_line`.
//! - The resolved level directive is propagated to the re-exec child through the
//!   `AIPERF_LOG` env (see [`current_directive`]); no reload handle is needed.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::Path;
use std::sync::{Mutex, OnceLock};

use tracing_subscriber::EnvFilter;
use tracing_subscriber::fmt::MakeWriter;
use tracing_subscriber::fmt::time::FormatTime;
use tracing_subscriber::prelude::*;

/// The env var carrying the `tracing` filter directive (also used to hand the
/// resolved level down to the `aiperf --execute` child).
pub const LOG_ENV: &str = "AIPERF_LOG";

/// The resolved filter directive, stored at [`init`] so the parent can pass it to
/// the re-exec child through [`current_directive`].
static RESOLVED_DIRECTIVE: OnceLock<String> = OnceLock::new();

/// The open `logs/aiperf.log` file, installed lazily by [`set_log_file`] once the
/// run's artifact dir is known. Absent until then, so the file layer discards.
static LOG_FILE: OnceLock<Mutex<File>> = OnceLock::new();

/// Install the process-wide `tracing` subscriber. Call once, early in `main`,
/// before any dispatch work (and before the `--execute` re-exec interception, so
/// the child inherits the same subscriber).
pub fn init(argv: &[String]) {
    let directive = std::env::var(LOG_ENV)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| level_directive_from_argv(argv));
    let _ = RESOLVED_DIRECTIVE.set(directive.clone());

    // A parse failure on an operator-supplied directive falls back to INFO rather
    // than aborting the process before it can report anything.
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

/// The resolved filter directive for this process, for propagation to the
/// `aiperf --execute` child via the [`LOG_ENV`] env. The child has no verbosity
/// flags of its own (its argv is just `--execute`), so the parent's resolved
/// level is the only way the child inherits it.
pub fn current_directive() -> String {
    RESOLVED_DIRECTIVE
        .get()
        .cloned()
        .unwrap_or_else(|| DEFAULT_LEVEL.to_owned())
}

/// Begin also writing every log line to `<artifact_dir>/logs/aiperf.log` (Python
/// `setup_rich_logging`'s `FileHandler`). Idempotent and best-effort: a failure to
/// create the folder or open the file logs a warning and leaves logging
/// console-only. Called by the front door once a run's artifact dir is resolved.
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

/// The default level when nothing else selects one — INFO, matching Python.
const DEFAULT_LEVEL: &str = "info";

/// Derive an `EnvFilter` directive from the CLI verbosity flags, mirroring
/// Python's `_converter_runtime`: `--extra-verbose` → TRACE, `--verbose`/`-v` →
/// DEBUG, `--log-level <lvl>` → the named level, else INFO. `--extra-verbose`
/// outranks `--verbose` outranks `--log-level`.
fn level_directive_from_argv(argv: &[String]) -> String {
    if argv.iter().any(|arg| arg == "--extra-verbose") {
        return "trace".to_owned();
    }
    if argv.iter().any(|arg| arg == "--verbose" || arg == "-v") {
        return "debug".to_owned();
    }
    if let Some(level) = option_value(argv, "--log-level") {
        return map_python_level(&level).to_owned();
    }
    DEFAULT_LEVEL.to_owned()
}

/// Read `--flag value` or `--flag=value` from `argv`.
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

/// Map a Python `AIPerfLogLevel` name onto a `tracing` level. Python's NOTICE
/// (between INFO and WARNING) and SUCCESS (between WARNING and ERROR) have no
/// `tracing` equivalent, so they collapse to the nearest lower `tracing` level
/// that still shows on a normal run (INFO / WARN respectively).
fn map_python_level(level: &str) -> &'static str {
    match level.trim().to_ascii_lowercase().as_str() {
        "trace" => "trace",
        "debug" => "debug",
        "info" | "notice" => "info",
        "warning" | "warn" | "success" => "warn",
        "error" | "critical" => "error",
        _ => DEFAULT_LEVEL,
    }
}

/// Console timer: `HH:MM:SS.mmm` in local time (Python basic-handler datefmt).
struct LocalTime;

impl FormatTime for LocalTime {
    fn format_time(&self, w: &mut tracing_subscriber::fmt::format::Writer<'_>) -> std::fmt::Result {
        write!(w, "{}", chrono::Local::now().format("%H:%M:%S%.3f"))
    }
}

/// File timer: `YYYY-MM-DD HH:MM:SS.mmm` in local time (Python file-handler datefmt).
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

/// A `MakeWriter` that writes to `logs/aiperf.log` once [`set_log_file`] has run,
/// and silently discards before then. This lets one subscriber be installed at
/// process start while the file target is bound later, when the artifact dir is
/// known.
struct LogFileMakeWriter;

impl<'a> MakeWriter<'a> for LogFileMakeWriter {
    type Writer = LogFileWriter;

    fn make_writer(&'a self) -> Self::Writer {
        LogFileWriter
    }
}

/// Per-event writer half of [`LogFileMakeWriter`].
struct LogFileWriter;

impl Write for LogFileWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if let Some(file) = LOG_FILE.get()
            && let Ok(mut guard) = file.lock()
        {
            let _ = guard.write_all(buf);
        }
        // Report success even when no file is bound: the file sink is optional and
        // must never surface a write error to the tracing machinery.
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
        assert_eq!(level_directive_from_argv(&argv), "trace");
    }

    #[test]
    fn verbose_outranks_log_level() {
        let argv = vec![
            "profile".to_owned(),
            "--log-level".to_owned(),
            "error".to_owned(),
            "-v".to_owned(),
        ];
        assert_eq!(level_directive_from_argv(&argv), "debug");
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
        assert_eq!(level_directive_from_argv(&equals), "debug");
    }

    #[test]
    fn default_is_info() {
        let argv = vec!["profile".to_owned(), "--model".to_owned(), "m".to_owned()];
        assert_eq!(level_directive_from_argv(&argv), "info");
    }

    #[test]
    fn notice_and_success_collapse_to_nearest_tracing_level() {
        assert_eq!(map_python_level("NOTICE"), "info");
        assert_eq!(map_python_level("success"), "warn");
        assert_eq!(map_python_level("critical"), "error");
        assert_eq!(map_python_level("bogus"), "info");
    }
}
