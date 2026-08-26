// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! `aiperf serve` — the always-on, standalone cross-run dashboard.
//!
//! Starts the [`crate::server`] dashboard as a long-lived process browsing every
//! run under a results root (no active benchmark required), and blocks until
//! SIGINT/SIGTERM, then shuts the server down gracefully. A live `aiperf profile`
//! / sweep additionally embeds the same server for its in-flight session
//! (`profile`), so the two share one module and one wire contract.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::server::{self, ServerConfig};

const DEFAULT_PORT: u16 = 8090;
const DEFAULT_HOST: &str = "127.0.0.1";

/// Parsed `aiperf serve` options.
struct Options {
    host: String,
    port: u16,
    results_dir: Option<PathBuf>,
}

fn parse(args: &[String]) -> anyhow::Result<Options> {
    let mut host = DEFAULT_HOST.to_string();
    let mut port = DEFAULT_PORT;
    let mut results_dir = std::env::current_dir().ok();
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--host" => {
                host = it
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--host requires a value"))?
                    .clone();
            }
            "--port" | "-p" => {
                port = it
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--port requires a value"))?
                    .parse()
                    .map_err(|e| anyhow::anyhow!("invalid --port: {e}"))?;
            }
            "--results-dir" | "--results" | "-d" => {
                results_dir =
                    Some(PathBuf::from(it.next().ok_or_else(|| {
                        anyhow::anyhow!("--results-dir requires a value")
                    })?));
            }
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            other => anyhow::bail!("unknown `aiperf serve` argument: {other}"),
        }
    }
    Ok(Options {
        host,
        port,
        results_dir,
    })
}

fn print_help() {
    eprintln!(
        "aiperf serve — always-on cross-run dashboard\n\n\
         Usage: aiperf serve [--host H] [--port N] [--results-dir DIR]\n\n\
         Options:\n  \
         --host H            bind address (default {DEFAULT_HOST})\n  \
         --port, -p N        bind port (default {DEFAULT_PORT}; 0 = OS-assigned)\n  \
         --results-dir, -d   root scanned for native-v2.json runs (default: cwd)\n"
    );
}

/// Run the standalone dashboard until interrupted.
pub fn run(args: &[String]) -> anyhow::Result<i32> {
    let opts = parse(args)?;
    let bind = format!("{}:{}", opts.host, opts.port)
        .parse()
        .map_err(|e| anyhow::anyhow!("invalid host:port {}:{}: {e}", opts.host, opts.port))?;

    // Must precede `server::start`: threads inherit the blocking thread's signal
    // mask, and a server thread that still has SIGINT/SIGTERM unblocked takes the
    // default action (terminate the process) the moment the signal lands — the
    // graceful path below would then never run. The `profile --serve` path gets the
    // same mask from `signals::install`, which runs before its `start_dashboard`.
    block_shutdown_signals();

    // Standalone serve has no live run loop feeding it, so its session index starts
    // empty (every run comes from the disk scan) and its live slot stays `None` (the
    // `/api/live` SSE then emits no event at all, only its keep-alive).
    let session = Arc::new(Mutex::new(Vec::new()));
    let live = Arc::new(Mutex::new(None));
    let handle = server::start(
        ServerConfig {
            bind,
            results_root: opts.results_dir.clone(),
        },
        session,
        live,
    )?;

    eprintln!(
        "aiperf: dashboard listening on http://{}",
        handle.local_addr()
    );
    match &opts.results_dir {
        Some(dir) => eprintln!("aiperf: browsing runs under {}", dir.display()),
        None => eprintln!("aiperf: no results dir (session-only)"),
    }
    eprintln!("aiperf: press Ctrl-C to stop");

    wait_for_shutdown();
    eprintln!("aiperf: shutting down dashboard");
    handle.shutdown();
    Ok(0)
}

/// The terminating signals the dashboard treats as "stop serving".
#[cfg(unix)]
fn shutdown_signals() -> nix::sys::signal::SigSet {
    use nix::sys::signal::{SigSet, Signal};
    let mut set = SigSet::empty();
    set.add(Signal::SIGINT);
    set.add(Signal::SIGTERM);
    set
}

/// Block SIGINT/SIGTERM in the calling thread so [`wait_for_shutdown`] receives
/// them synchronously. Call before spawning any thread the process needs alive
/// for the graceful path.
#[cfg(unix)]
pub(crate) fn block_shutdown_signals() {
    let _ = shutdown_signals().thread_block();
}

#[cfg(not(unix))]
pub(crate) fn block_shutdown_signals() {}

/// Block the calling thread until SIGINT/SIGTERM.
#[cfg(unix)]
pub(crate) fn wait_for_shutdown() {
    let set = shutdown_signals();
    // Idempotent with `block_shutdown_signals`; also covers a caller that skipped it.
    let _ = set.thread_block();
    let _ = set.wait();
}

#[cfg(not(unix))]
pub(crate) fn wait_for_shutdown() {
    // Non-unix (non-product): park until the process is killed.
    loop {
        std::thread::sleep(std::time::Duration::from_secs(3600));
    }
}
