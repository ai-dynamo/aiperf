// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stdio entry point for one orchestrator-authored benchmark run.

use std::collections::BTreeMap;
use std::io::{self, Read, Write};

use aiperf_runner::protocol_v2::{
    RunTerminalV2, RunValidationV2, RunnerDiagnosticV2, RunnerEnvelopeV2, RunnerFailureStageV2,
    RunnerOperationV2, ValidationCompletenessV2, RUNNER_PROTOCOL_V2,
};
use aiperf_runner::redaction::redact_diagnostic;
use aiperf_runner::{current_distribution_id, RunnerApplication};
use serde::Deserialize;
use serde_json::{value::RawValue, Value};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[cfg(target_os = "linux")]
#[used]
#[unsafe(link_section = ".init_array.00100")]
static AIPERF_MIMALLOC_PREINIT: unsafe extern "C" fn() = configure_mimalloc_before_process_init;

#[cfg(target_os = "linux")]
unsafe extern "C" fn configure_mimalloc_before_process_init() {
    // mimalloc's own Linux constructor has priority 101. This priority-100 hook
    // changes only its uninitialized default before that constructor commits the
    // initial arena. Leaving the option uninitialized lets mimalloc's own parser
    // honor canonical, case-insensitive, and legacy environment spellings.
    // The C shim resolves the experimental enum from the exact header compiled
    // by libmimalloc-sys instead of duplicating its unstable numeric value.
    // SAFETY: mimalloc has not run process initialization and no Rust heap
    // allocation can precede an ELF init-array constructor.
    unsafe { libmimalloc_sys::mi_option_set_default(aiperf_mi_option_arena_eager_commit(), 0) };
}

unsafe extern "C" {
    fn aiperf_mi_option_arena_eager_commit() -> libmimalloc_sys::mi_option_t;
    fn aiperf_mi_option_purge_delay() -> libmimalloc_sys::mi_option_t;
}

fn main() {
    let arguments = std::env::args_os().skip(1).collect::<Vec<_>>();
    if arguments.len() == 1 && arguments[0] == "--capabilities" {
        let application = compose_stock_application();
        write_json_line(&application.catalog(), 0);
    }
    if !arguments.is_empty() {
        eprintln!("usage: aiperf-runner [--capabilities]");
        std::process::exit(2);
    }

    let mut input = Vec::new();
    if let Err(error) = io::stdin().read_to_end(&mut input) {
        eprintln!("failed to read runner request from stdin: {error}");
        std::process::exit(2);
    }
    configure_dynosim_process_defaults(&input);
    let application = compose_stock_application();
    // The runner speaks only protocol v2. run_v2 rejects a non-v2 or malformed
    // request as a v2 failure envelope (a v1 request fails EnvelopeBootstrapV2
    // parsing and is reported as an invalid protocol-v2 request).
    run_v2(&input, &application);
}

fn compose_stock_application() -> RunnerApplication {
    let distribution_id = match current_distribution_id() {
        Ok(distribution_id) => distribution_id,
        Err(error) => {
            eprintln!("failed to identify executing aiperf-runner image: {error}");
            std::process::exit(2);
        }
    };
    match RunnerApplication::stock(distribution_id) {
        Ok(application) => application,
        Err(error) => {
            eprintln!("failed to compose executing aiperf-runner image: {error:#}");
            std::process::exit(2);
        }
    }
}

fn configure_dynosim_process_defaults(input: &[u8]) {
    let Ok(envelope) = serde_json::from_slice::<Value>(input) else {
        return;
    };
    if !matches!(
        envelope
            .pointer("/run/cfg/transport/type")
            .and_then(Value::as_str),
        Some("dynosim_offline" | "dynosim_online")
    ) {
        return;
    }

    // AIC imports SciPy, whose OpenBLAS builds otherwise create one worker per
    // host CPU. Offline replay uses AIC's scalar Rust interpolation kernels, so
    // those pools only spin and contend with the deterministic event loop. Keep
    // explicit operator settings authoritative, keep the simulator on its one
    // deterministic event-loop thread, and avoid allocator purge syscalls before
    // immediate process exit. Rayon is used only by the post-drain report
    // reduction; bound it to the available affinity and eight workers so large
    // sweep sorts can leave Tokio without oversubscribing small hosts. These
    // defaults are installed before Python, OpenMP, or either bundled OpenBLAS
    // library is initialized and before offline reduction allocates its buffers.
    let rayon_threads = std::thread::available_parallelism()
        .map_or(1, std::num::NonZeroUsize::get)
        .min(8)
        .to_string();
    for (name, value) in [
        ("OPENBLAS_NUM_THREADS", "1"),
        ("OMP_NUM_THREADS", "1"),
        ("MKL_NUM_THREADS", "1"),
        ("BLIS_NUM_THREADS", "1"),
        ("GOTO_NUM_THREADS", "1"),
        ("NUMEXPR_NUM_THREADS", "1"),
        ("VECLIB_MAXIMUM_THREADS", "1"),
        ("OMP_WAIT_POLICY", "PASSIVE"),
        ("RAYON_NUM_THREADS", rayon_threads.as_str()),
    ] {
        if std::env::var_os(name).is_none() {
            // SAFETY: this runs on the sole process thread before the runner
            // constructs a runtime or initializes any native numeric library.
            unsafe { std::env::set_var(name, value) };
        }
    }

    if std::env::var_os("MIMALLOC_PURGE_DELAY").is_none() {
        // The runner exits immediately after committing its report, so purging
        // temporary sweep pages during the run only adds syscalls and cannot
        // improve a later phase's footprint. The C shim resolves the option
        // against the same exact mimalloc header as the linked allocator.
        // SAFETY: option mutation is not thread-safe, so it is performed here on
        // the sole process thread before Rayon or any benchmark runtime exists.
        unsafe { libmimalloc_sys::mi_option_set(aiperf_mi_option_purge_delay(), -1) };
    }
}

#[derive(Deserialize)]
struct EnvelopeBootstrapV2 {
    protocol_version: u32,
    operation: RunnerOperationV2,
    run: Box<RawValue>,
}

fn run_v2(input: &[u8], application: &RunnerApplication) -> ! {
    let distribution_id = application.distribution_id().to_owned();
    let bootstrap = match serde_json::from_slice::<EnvelopeBootstrapV2>(input) {
        Ok(bootstrap) => bootstrap,
        Err(error) => {
            write_v2_protocol_failure(
                operation_hint(input),
                distribution_id,
                benchmark_id_hint(input),
                "invalid_request",
                format!("invalid protocol-v2 request: {error}"),
            );
        }
    };
    if bootstrap.protocol_version != RUNNER_PROTOCOL_V2 {
        write_v2_protocol_failure(
            Some(bootstrap.operation),
            distribution_id,
            benchmark_id_from_raw(&bootstrap.run),
            "unsupported_protocol",
            format!(
                "runner protocol {} is unsupported; expected {RUNNER_PROTOCOL_V2}",
                bootstrap.protocol_version
            ),
        );
    }
    let envelope = match serde_json::from_slice::<RunnerEnvelopeV2>(input) {
        Ok(envelope) => envelope,
        Err(error) => write_v2_protocol_failure(
            Some(bootstrap.operation),
            distribution_id,
            benchmark_id_from_raw(&bootstrap.run),
            "invalid_request",
            format!("invalid protocol-v2 request: {error}"),
        ),
    };
    // The runner's contract is exactly one terminal/validation JSONL line. A
    // panic anywhere in prepare/execute would otherwise unwind past this writer
    // and abort the child (exit 101) with no envelope, so Python sees a crashed
    // subprocess instead of a typed failure. Convert a caught panic into the
    // corresponding v2 failure envelope.
    let operation = bootstrap.operation;
    let benchmark_id = benchmark_id_from_raw(&bootstrap.run);
    let result = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        application.handle_v2(envelope)
    })) {
        Ok(result) => result,
        Err(payload) => write_v2_internal_panic(
            operation,
            distribution_id,
            benchmark_id,
            panic_payload_message(payload.as_ref()),
        ),
    };
    write_json_line(&result.response, result.exit_code);
}

/// Best-effort human-readable message from a caught panic payload.
fn panic_payload_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_owned()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "runner panicked with a non-string payload".to_owned()
    }
}

/// Emit the v2 failure envelope for an internal panic, matching the requested
/// operation (validation failure for `validate`, execution-stage terminal for
/// `execute`).
fn write_v2_internal_panic(
    operation: RunnerOperationV2,
    distribution_id: String,
    benchmark_id: Option<String>,
    message: String,
) -> ! {
    let message = format!("aiperf-runner internal panic: {message}");
    match operation {
        RunnerOperationV2::Validate => write_json_line(
            &RunValidationV2 {
                protocol_version: RUNNER_PROTOCOL_V2,
                event: "run_validation",
                benchmark_id,
                success: false,
                completeness: ValidationCompletenessV2::Static,
                deferred_checks: Vec::new(),
                errors: vec![diagnostic("internal_panic", message)],
            },
            2,
        ),
        RunnerOperationV2::Execute => write_v2_terminal_failure(
            distribution_id,
            benchmark_id,
            RunnerFailureStageV2::Execution,
            "internal_panic",
            message,
            2,
        ),
    }
}

fn operation_hint(input: &[u8]) -> Option<RunnerOperationV2> {
    let value: Value = serde_json::from_slice(input).ok()?;
    match value.get("operation")?.as_str()? {
        "validate" => Some(RunnerOperationV2::Validate),
        "execute" => Some(RunnerOperationV2::Execute),
        _ => None,
    }
}

fn benchmark_id_hint(input: &[u8]) -> Option<String> {
    let value: Value = serde_json::from_slice(input).ok()?;
    value
        .pointer("/run/benchmark_id")?
        .as_str()
        .map(str::to_owned)
}

fn benchmark_id_from_raw(run: &RawValue) -> Option<String> {
    let value: Value = serde_json::from_str(run.get()).ok()?;
    value.pointer("/benchmark_id")?.as_str().map(str::to_owned)
}

fn write_v2_protocol_failure(
    operation: Option<RunnerOperationV2>,
    distribution_id: String,
    benchmark_id: Option<String>,
    code: &str,
    message: String,
) -> ! {
    match operation {
        Some(RunnerOperationV2::Validate) => write_json_line(
            &RunValidationV2 {
                protocol_version: RUNNER_PROTOCOL_V2,
                event: "run_validation",
                benchmark_id,
                success: false,
                completeness: ValidationCompletenessV2::Static,
                deferred_checks: Vec::new(),
                errors: vec![diagnostic(code, message)],
            },
            2,
        ),
        Some(RunnerOperationV2::Execute) | None => write_v2_terminal_failure(
            distribution_id,
            benchmark_id,
            RunnerFailureStageV2::Protocol,
            code,
            message,
            2,
        ),
    }
}

fn write_v2_terminal_failure(
    _distribution_id: String,
    benchmark_id: Option<String>,
    stage: RunnerFailureStageV2,
    code: &str,
    message: String,
    exit_code: i32,
) -> ! {
    write_json_line(
        &RunTerminalV2 {
            protocol_version: RUNNER_PROTOCOL_V2,
            event: "run_terminal",
            benchmark_id,
            success: false,
            report_path: None,
            stage: Some(stage),
            errors: vec![diagnostic(code, message)],
            diagnostic_artifacts: Vec::new(),
            provenance: BTreeMap::new(),
        },
        exit_code,
    )
}

fn diagnostic(code: &str, message: String) -> RunnerDiagnosticV2 {
    RunnerDiagnosticV2 {
        code: code.to_owned(),
        message: redact_diagnostic(message),
        path: None,
    }
}

fn write_json_line(value: &impl serde::Serialize, exit_code: i32) -> ! {
    let mut stdout = io::stdout().lock();
    if serde_json::to_writer(&mut stdout, value).is_err()
        || stdout.write_all(b"\n").is_err()
        || stdout.flush().is_err()
    {
        std::process::exit(2);
    }
    std::process::exit(exit_code);
}
