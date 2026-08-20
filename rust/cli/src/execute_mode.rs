// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Internal execution modes for one benchmark run over stdio.
//!
//! The entry point re-execs itself for each run, probe, or cell. The child enters
//! [`dispatch`], preserving process, signal, and panic isolation.
//!
//! The child reads the entire request from stdin, runs it through
//! `Application::handle_v2`, and writes exactly one JSONL envelope to stdout
//! (stdout is the protocol channel; all diagnostics go to stderr). It accepts
//! protocol v2 only; a panic anywhere in
//! prepare/execute is caught and converted to a typed v2 failure envelope so the
//! parent sees a clean failure instead of a crashed subprocess.
//!
//! The stdin payload is a bare protocol-v2 `BenchmarkRun`; the requested
//! operation is selected by the re-exec MODE (argv), not a wire field. The child
//! reconstructs the internal `EnvelopeV2` with the fixed protocol version and
//! selected operation.
//!
//! Modes (selected by the first argument, set by the re-exec spawn site):
//! `--execute` (single-process run / controller self-promotion on `cells>1`),
//! `--validate` (side-effect-free validate of the same bare run), `--cell` and
//! `--aggregator` (velo-gated multi-cell tiers). Capabilities is an in-process
//! function, not a mode.

use std::collections::BTreeMap;
use std::io::{self, Read, Write};

use aiperf_runtime::engine::application::Application;
#[cfg(feature = "cellular")]
use aiperf_runtime::engine::cellular_kind::CellularRunKind;
use aiperf_runtime::engine::distribution_identity::current_distribution_id;
use aiperf_runtime::engine::protocol_v2::{
    DiagnosticV2, EnvelopeV2, FailureStageV2, OperationV2, PROTOCOL_V2, RunTerminalV2,
    RunValidationV2, ValidationCompletenessV2,
};
use aiperf_runtime::engine::redaction::redact_diagnostic;
use serde_json::Value;

// The C shim resolves this option against the same mimalloc header as the linked
// allocator.
unsafe extern "C" {
    fn aiperf_mi_option_purge_delay() -> libmimalloc_sys::mi_option_t;
}

/// The internal re-exec flag: `aiperf --execute` reads one bare protocol-v2
/// `BenchmarkRun` from stdin and executes it. It is hidden from `--help`.
pub const EXECUTE_FLAG: &str = "--execute";
/// The internal re-exec flag: `aiperf --validate` reads the same bare
/// `BenchmarkRun` from stdin but runs it as a side-effect-free `validate`
/// operation.
pub const VALIDATE_FLAG: &str = "--validate";
/// `aiperf --cell` runs this process as one cell of a multi-cell run (velo).
pub const CELL_FLAG: &str = "--cell";
/// `aiperf --aggregator` runs this process as a tier-T2 merge aggregator (velo).
pub const AGGREGATOR_FLAG: &str = "--aggregator";

/// Return whether the arguments select an internal execution mode.
pub fn is_execution_mode(args: &[String]) -> bool {
    matches!(
        args,
        [flag] if flag == EXECUTE_FLAG
            || flag == VALIDATE_FLAG
            || flag == CELL_FLAG
            || flag == AGGREGATOR_FLAG
    )
}

/// Compose the stock application and return its capabilities catalog.
///
/// The catalog reports the linked distribution's plugins.yaml-shaped inventory.
pub fn capabilities_catalog() -> anyhow::Result<aiperf_runtime::engine::protocol::Catalog> {
    let distribution_id = current_distribution_id()
        .map_err(|error| anyhow::anyhow!("failed to identify aiperf distribution: {error}"))?;
    let application = Application::stock(distribution_id)
        .map_err(|error| anyhow::anyhow!("failed to compose aiperf distribution: {error:#}"))?;
    Ok(application.catalog())
}

/// Enter the requested execution mode and drive one run to its terminal envelope.
/// Always terminates the process (`-> !`); callers do not return from here.
pub fn dispatch(args: &[String]) -> ! {
    crate::diagnostics::register_sigusr1_faulthandler();
    let flag = args.first().map(String::as_str).unwrap_or("");
    let cell_mode = flag == CELL_FLAG;
    let aggregator_mode = flag == AGGREGATOR_FLAG;
    let validate_mode = flag == VALIDATE_FLAG;
    let operation = if validate_mode {
        OperationV2::Validate
    } else {
        OperationV2::Execute
    };

    // Cell mode receives its envelope over velo; other modes consume stdin.
    let mut input = Vec::new();
    if !cell_mode {
        if let Err(error) = io::stdin().read_to_end(&mut input) {
            tracing::error!(error = %error, "failed to read run request from stdin");
            std::process::exit(2);
        }
    }

    if cell_mode {
        run_cell();
    }
    if aggregator_mode {
        run_aggregator(&input);
    }

    // Cellular helpers require `/run/...` pointers, so controller requests use the
    // wrapped `{"run": …}` representation over a **resolved** run: the parent ships
    // authoring inputs, so `cfg.runtime.cells` only exists after resolution.
    if !validate_mode
        && std::env::var(aiperf_runtime::cellular::partition::CELL_ID_ENV).is_err()
        && let Some((wrapped, cells)) =
            aiperf_runtime::engine::cell_launcher::resolved_envelope_from_input(&input)
    {
        // Promote to the cellular controller when the run partitions across more than
        // one cell, OR when a cross-host launcher (k8s/slurm) is active even for a
        // single cell: there a separate cell task already exists and is dialing this
        // controller (e.g. a 2-task SLURM allocation, `cells == 1`), so the controller
        // must bind velo rather than run as a lone single process. The same-host
        // default keeps `cells == 1` as a plain single-process run.
        if cells > 1
            || (cells >= 1 && aiperf_runtime::engine::cell_launcher::is_cross_host_launcher())
        {
            run_controller(&wrapped);
        }
    }

    configure_dynosim_process_defaults(&input);
    let application = compose_stock_application();
    run_v2(&input, operation, &application);
}

/// Run one cell using the launcher-provided `AIPERF_CELL_*` environment.
#[cfg(feature = "cellular")]
fn run_cell() -> ! {
    // Drop the fetch runtime before execution creates its thread-per-core runtime.
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(error) => {
            tracing::error!(error = %error, "failed to build cell fetch runtime");
            std::process::exit(2);
        }
    };
    let envelope =
        match runtime.block_on(aiperf_runtime::engine::cellular_cell::fetch_cell_envelope()) {
            Ok(envelope) => envelope,
            Err(error) => {
                tracing::error!(
                    error = format!("{error:#}"),
                    "cell failed to fetch its envelope"
                );
                std::process::exit(2);
            }
        };
    let (envelope_bytes, landing_guard) = envelope.into_execution_parts();
    // Dataset fan-out must complete on the fetch runtime before it is dropped.
    if let Err(error) =
        runtime.block_on(aiperf_runtime::engine::cellular_cell::verify_dataset_fanout())
    {
        tracing::error!(error = format!("{error:#}"), "cell dataset fan-out failed");
        if let Err(error) = landing_guard.close() {
            tracing::warn!(error = %error, "cell dataset landing cleanup failed");
        }
        std::process::exit(2);
    }
    drop(runtime);
    // Execution consumes a bare `BenchmarkRun`, while cellular helpers use the
    // wrapped `{"run": …}` envelope. The landing guard stays in scope through
    // `run_v2`, which owns every read of the rewritten local dataset path.
    let run_bytes = run_object_bytes(&envelope_bytes);
    configure_dynosim_process_defaults(&run_bytes);
    let application = compose_stock_application();
    run_v2_with_cleanup(&run_bytes, OperationV2::Execute, &application, move || {
        landing_guard.close()
    });
}

/// Extract the bare `BenchmarkRun`, preserving malformed input for typed errors.
fn run_object_bytes(envelope: &[u8]) -> Vec<u8> {
    match serde_json::from_slice::<Value>(envelope) {
        Ok(value) => match value.get("run") {
            Some(run) => serde_json::to_vec(run).unwrap_or_else(|_| envelope.to_vec()),
            None => envelope.to_vec(),
        },
        Err(_) => envelope.to_vec(),
    }
}

/// Run a tier-T2 aggregator and send its merged store to the controller.
#[cfg(feature = "cellular")]
fn run_aggregator(input: &[u8]) -> ! {
    let envelope = match serde_json::from_slice::<Value>(input) {
        Ok(envelope) => envelope,
        Err(error) => {
            tracing::error!(error = %error, "aggregator failed to parse its envelope");
            std::process::exit(2);
        }
    };
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(error) => {
            tracing::error!(error = %error, "failed to build aggregator runtime");
            std::process::exit(2);
        }
    };
    match runtime.block_on(aiperf_runtime::engine::cellular_aggregator::run_aggregator(
        &envelope,
    )) {
        Ok(()) => std::process::exit(0),
        Err(error) => {
            tracing::error!(error = format!("{error:#}"), "aggregator failed");
            std::process::exit(2);
        }
    }
}

#[cfg(not(feature = "cellular"))]
fn run_aggregator(_input: &[u8]) -> ! {
    tracing::error!(
        "aiperf was built without the `velo` feature; `--aggregator` (tier-T2 tree merge) requires it"
    );
    std::process::exit(2);
}

#[cfg(not(feature = "cellular"))]
fn run_cell() -> ! {
    tracing::error!(
        "aiperf was built without the `velo` feature; `--cell` (multi-cell runs) requires it"
    );
    std::process::exit(2);
}

#[cfg(not(feature = "cellular"))]
fn run_controller(envelope: &Value) -> ! {
    let benchmark_id = envelope
        .pointer("/run/benchmark_id")
        .and_then(Value::as_str)
        .map(str::to_owned);
    emit_cellular_failure(
        benchmark_id,
        "velo_feature_required",
        "aiperf was built without the `velo` feature; multi-cell runs (cells>1) require it"
            .to_owned(),
    );
}

/// Drive cellular execution and emit one terminal envelope.
#[cfg(feature = "cellular")]
fn run_controller(envelope: &Value) -> ! {
    let benchmark_id = envelope
        .pointer("/run/benchmark_id")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let artifact_dir = envelope
        .pointer("/run/artifact_dir")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let report_path = std::path::Path::new(artifact_dir).join("native-v2.json");
    let cell_count = aiperf_runtime::engine::cell_launcher::cell_count_from_envelope(envelope);
    let application = compose_stock_application();
    let exporters = application.product_registry().exporters();
    // Catch panics across merge, serialization, and export so the parent always
    // receives a typed terminal envelope.
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        aiperf_runtime::engine::cellular_controller::run_cellular(
            envelope,
            cell_count,
            &report_path,
            exporters,
        )
    }));
    match outcome {
        Ok(Ok(outcome)) => {
            let mut terminal_metadata = BTreeMap::new();
            terminal_metadata.insert(
                "transport".to_owned(),
                envelope
                    .pointer("/run/cfg/transport/type")
                    .and_then(Value::as_str)
                    .unwrap_or("http")
                    .to_owned(),
            );
            terminal_metadata.insert(
                "workload".to_owned(),
                CellularRunKind::detect(&envelope)
                    .workload_label()
                    .to_owned(),
            );
            terminal_metadata.insert("cells".to_owned(), outcome.cell_count.to_string());
            terminal_metadata.insert("record_count".to_owned(), outcome.record_count.to_string());
            write_json_line(
                &RunTerminalV2 {
                    protocol_version: PROTOCOL_V2,
                    event: "run_terminal",
                    benchmark_id,
                    success: true,
                    report_path: Some(outcome.report_path),
                    stage: None,
                    errors: Vec::new(),
                    diagnostic_artifacts: Vec::new(),
                    run_metadata: terminal_metadata,
                },
                0,
            );
        }
        Ok(Err(error)) => {
            emit_cellular_failure(benchmark_id, "cellular_run_failed", format!("{error:#}"));
        }
        Err(payload) => {
            emit_cellular_failure(
                benchmark_id,
                "internal_panic",
                panic_payload_message(payload.as_ref()),
            );
        }
    }
}

fn emit_cellular_failure(benchmark_id: Option<String>, code: &'static str, message: String) -> ! {
    tracing::error!(error = %message, "cellular run failed");
    write_json_line(
        &RunTerminalV2 {
            protocol_version: PROTOCOL_V2,
            event: "run_terminal",
            benchmark_id,
            success: false,
            report_path: None,
            stage: Some(FailureStageV2::Execution),
            errors: vec![diagnostic(code, message)],
            diagnostic_artifacts: Vec::new(),
            run_metadata: BTreeMap::new(),
        },
        1,
    );
}

fn compose_stock_application() -> Application {
    let distribution_id = match current_distribution_id() {
        Ok(distribution_id) => distribution_id,
        Err(error) => {
            tracing::error!(error = %error, "failed to identify executing aiperf image");
            std::process::exit(2);
        }
    };
    match Application::stock(distribution_id) {
        Ok(application) => application,
        Err(error) => {
            tracing::error!(
                error = format!("{error:#}"),
                "failed to compose executing aiperf image"
            );
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
            .pointer("/cfg/transport/type")
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
            // SAFETY: this runs on the sole process thread before the execution
            // path constructs a runtime or initializes any native numeric library.
            unsafe { std::env::set_var(name, value) };
        }
    }

    if std::env::var_os("MIMALLOC_PURGE_DELAY").is_none() {
        // The process exits immediately after committing its report, so purging
        // temporary sweep pages during the run only adds syscalls and cannot
        // improve a later phase's footprint. The C shim resolves the option
        // against the same exact mimalloc header as the linked allocator.
        // SAFETY: option mutation is not thread-safe, so it is performed here on
        // the sole process thread before Rayon or any benchmark runtime exists.
        unsafe { libmimalloc_sys::mi_option_set(aiperf_mi_option_purge_delay(), -1) };
    }
}

/// Drive one bare-run request to its terminal or validation envelope.
///
/// The stdin payload is the authoring execute wire decoded by
/// [`decode_execute_wire`](aiperf_runtime::engine::protocol_v2::decode_execute_wire):
/// an authoring `{"authoring": <Inputs>}` envelope the runtime resolves here (every
/// profile path — single run, sweeps, and adaptive search — ships authoring). The
/// `operation` is
/// selected by the re-exec mode (`--execute` or `--validate`), not carried on the
/// wire. A malformed run produces a typed v2 protocol failure.
fn run_v2(input: &[u8], operation: OperationV2, application: &Application) -> ! {
    run_v2_with_cleanup(input, operation, application, || Ok(()));
}

fn run_v2_with_cleanup<F>(
    input: &[u8],
    operation: OperationV2,
    application: &Application,
    cleanup: F,
) -> !
where
    F: FnOnce() -> anyhow::Result<()>,
{
    let mut cleanup = Some(cleanup);
    let distribution_id = application.distribution_id().to_owned();
    let run = match aiperf_runtime::engine::protocol_v2::decode_execute_wire(input) {
        Ok(run) => run,
        Err(error) => {
            run_v2_cleanup(&mut cleanup);
            write_v2_protocol_failure(
                Some(operation),
                distribution_id,
                benchmark_id_hint(input),
                "invalid_request",
                format!("invalid protocol-v2 request: {error}"),
            )
        }
    };
    let envelope = EnvelopeV2 {
        protocol_version: PROTOCOL_V2,
        operation,
        run,
    };
    // Convert panics into typed responses to preserve the one-line JSONL contract.
    let benchmark_id = benchmark_id_hint(input);
    let result = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        application.handle_v2(envelope)
    })) {
        Ok(result) => result,
        Err(payload) => {
            run_v2_cleanup(&mut cleanup);
            write_v2_internal_panic(
                operation,
                distribution_id,
                benchmark_id,
                panic_payload_message(payload.as_ref()),
            )
        }
    };
    run_v2_cleanup(&mut cleanup);
    write_json_line(&result.response, result.exit_code);
}

fn run_v2_cleanup<F>(cleanup: &mut Option<F>)
where
    F: FnOnce() -> anyhow::Result<()>,
{
    if let Some(cleanup) = cleanup.take() {
        if let Err(error) = cleanup() {
            tracing::warn!(error = %error, "cell dataset landing cleanup failed");
        }
    }
}

fn panic_payload_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_owned()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "aiperf execution panicked with a non-string payload".to_owned()
    }
}

fn write_v2_internal_panic(
    operation: OperationV2,
    distribution_id: String,
    benchmark_id: Option<String>,
    message: String,
) -> ! {
    let message = format!("aiperf internal panic: {message}");
    match operation {
        OperationV2::Validate => write_json_line(
            &RunValidationV2 {
                protocol_version: PROTOCOL_V2,
                event: "run_validation",
                benchmark_id,
                success: false,
                completeness: ValidationCompletenessV2::Static,
                deferred_checks: Vec::new(),
                errors: vec![diagnostic("internal_panic", message)],
            },
            2,
        ),
        OperationV2::Execute => write_v2_terminal_failure(
            distribution_id,
            benchmark_id,
            FailureStageV2::Execution,
            "internal_panic",
            message,
            2,
        ),
    }
}

/// Recover a `benchmark_id` from possibly malformed input.
fn benchmark_id_hint(input: &[u8]) -> Option<String> {
    let value: Value = serde_json::from_slice(input).ok()?;
    value.pointer("/benchmark_id")?.as_str().map(str::to_owned)
}

fn write_v2_protocol_failure(
    operation: Option<OperationV2>,
    distribution_id: String,
    benchmark_id: Option<String>,
    code: &str,
    message: String,
) -> ! {
    match operation {
        Some(OperationV2::Validate) => write_json_line(
            &RunValidationV2 {
                protocol_version: PROTOCOL_V2,
                event: "run_validation",
                benchmark_id,
                success: false,
                completeness: ValidationCompletenessV2::Static,
                deferred_checks: Vec::new(),
                errors: vec![diagnostic(code, message)],
            },
            2,
        ),
        Some(OperationV2::Execute) | None => write_v2_terminal_failure(
            distribution_id,
            benchmark_id,
            FailureStageV2::Protocol,
            code,
            message,
            2,
        ),
    }
}

fn write_v2_terminal_failure(
    _distribution_id: String,
    benchmark_id: Option<String>,
    stage: FailureStageV2,
    code: &str,
    message: String,
    exit_code: i32,
) -> ! {
    write_json_line(
        &RunTerminalV2 {
            protocol_version: PROTOCOL_V2,
            event: "run_terminal",
            benchmark_id,
            success: false,
            report_path: None,
            stage: Some(stage),
            errors: vec![diagnostic(code, message)],
            diagnostic_artifacts: Vec::new(),
            run_metadata: BTreeMap::new(),
        },
        exit_code,
    )
}

fn diagnostic(code: &str, message: String) -> DiagnosticV2 {
    DiagnosticV2 {
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

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::rc::Rc;

    use super::run_v2_cleanup;

    #[test]
    fn owned_cleanup_run_v2_terminal_cleanup_runs_once() {
        let cleanup_ran = Rc::new(Cell::new(0));
        let observed = Rc::clone(&cleanup_ran);
        let mut cleanup = Some(move || {
            observed.set(observed.get() + 1);
            Ok(())
        });

        run_v2_cleanup(&mut cleanup);
        run_v2_cleanup(&mut cleanup);

        assert_eq!(cleanup_ran.get(), 1);
    }
}
