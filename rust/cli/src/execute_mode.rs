// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The `aiperf` binary's execution modes — one benchmark run over the stdio seam.
//!
//! This is the relocated body of the deleted `aiperf-runner` binary. The single
//! `aiperf` binary is BOTH the entry point and the execution engine: for each
//! run/probe/cell the entry point re-execs **itself** (`aiperf --execute`) and the
//! child enters [`dispatch`] here, preserving the process/SIGINT/panic isolation
//! boundary the separate runner binary used to provide.
//!
//! The protocol is unchanged: read the entire request from stdin, run it through
//! `RunnerApplication::handle_v2`, and write exactly one JSONL envelope to stdout
//! (STDOUT is the protocol channel; all diagnostics go to STDERR via the CLI's
//! `tracing` subscriber). Speaks protocol v2 only; a panic anywhere in
//! prepare/execute is caught and converted to a typed v2 failure envelope so the
//! parent sees a clean failure instead of a crashed subprocess.
//!
//! Modes (selected by the first argument, set by the re-exec spawn site):
//! `--execute` (single-process run / controller self-promotion on `cells>1`),
//! `--capabilities` (print the catalog), `--cell` and `--aggregator` (velo-gated
//! multi-cell tiers). `current_exe()` is the same `aiperf` binary, so the cellular
//! launcher's `current_exe() --cell` re-exec resolves back here.

use std::collections::BTreeMap;
use std::io::{self, Read, Write};

use aiperf_runtime::runner_protocol::application::RunnerApplication;
use aiperf_runtime::runner_protocol::cellular_kind::CellularRunKind;
use aiperf_runtime::runner_protocol::distribution_identity::current_distribution_id;
use aiperf_runtime::runner_protocol::protocol_v2::{
    RUNNER_PROTOCOL_V2, RunTerminalV2, RunValidationV2, RunnerDiagnosticV2, RunnerEnvelopeV2,
    RunnerFailureStageV2, RunnerOperationV2, ValidationCompletenessV2,
};
use aiperf_runtime::runner_protocol::redaction::redact_diagnostic;
use serde::Deserialize;
use serde_json::{Value, value::RawValue};

// Declared here (not shared with main.rs's arena-preinit block) so the dynosim
// purge-delay tweak resolves the option constant against the same exact mimalloc
// header the linked allocator uses. The symbol is provided by build.rs's C shim.
unsafe extern "C" {
    fn aiperf_mi_option_purge_delay() -> libmimalloc_sys::mi_option_t;
}

/// The internal re-exec flag: `aiperf --execute` reads one protocol-v2 request
/// from stdin and runs it. Hidden from `--help`; it is a re-exec target, not a
/// user command.
pub const EXECUTE_FLAG: &str = "--execute";
/// `aiperf --cell` runs this process as one cell of a multi-cell run (velo).
pub const CELL_FLAG: &str = "--cell";
/// `aiperf --aggregator` runs this process as a tier-T2 merge aggregator (velo).
pub const AGGREGATOR_FLAG: &str = "--aggregator";

/// Whether `args` (the arguments after argv[0]) select an execution mode. The
/// entry point routes these to [`dispatch`] before ordinary subcommand parsing.
///
/// Capabilities is NOT here: it is an in-process function ([`capabilities_catalog`]),
/// never a subprocess mode — the entry point and execution engine are one binary.
pub fn is_execution_mode(args: &[String]) -> bool {
    matches!(
        args,
        [flag] if flag == EXECUTE_FLAG || flag == CELL_FLAG || flag == AGGREGATOR_FLAG
    )
}

/// Compose the stock application and return its capabilities catalog.
///
/// The catalog is a direct in-process call — there is no `--capabilities`
/// subprocess/argv mode and no Python preflight. Any Rust caller that needs the
/// linked distribution's plugins.yaml-shaped inventory calls this.
pub fn capabilities_catalog()
-> anyhow::Result<aiperf_runtime::runner_protocol::protocol::RunnerCatalog> {
    let distribution_id = current_distribution_id()
        .map_err(|error| anyhow::anyhow!("failed to identify aiperf distribution: {error}"))?;
    let application = RunnerApplication::stock(distribution_id)
        .map_err(|error| anyhow::anyhow!("failed to compose aiperf distribution: {error:#}"))?;
    Ok(application.catalog())
}

/// Enter the requested execution mode and drive one run to its terminal envelope.
/// Always terminates the process (`-> !`); callers do not return from here.
pub fn dispatch(args: &[String]) -> ! {
    let flag = args.first().map(String::as_str).unwrap_or("");
    let cell_mode = flag == CELL_FLAG;
    let aggregator_mode = flag == AGGREGATOR_FLAG;

    // A cell child fetches its sliced envelope over velo, not stdin. Every other
    // mode reads the full request from stdin to EOF.
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

    // A non-cell execute request asking for more than one cell becomes the
    // controller: it drives the cells over velo and merges their records.
    if std::env::var(aiperf_runtime::cellular::partition::CELL_ID_ENV).is_err()
        && let Ok(envelope) = serde_json::from_slice::<Value>(&input)
        && envelope.pointer("/operation").and_then(Value::as_str) == Some("execute")
        && aiperf_runtime::runner_protocol::cell_launcher::cell_count_from_envelope(&envelope) > 1
    {
        run_controller(&envelope);
    }

    configure_dynosim_process_defaults(&input);
    let application = compose_stock_application();
    // The execution path speaks only protocol v2. run_v2 rejects a non-v2 or
    // malformed request as a v2 failure envelope.
    run_v2(&input, &application);
}

/// Runs this process as one cell of a multi-cell run. The launcher has already set
/// the partition/controller environment (`AIPERF_CELL_*`); the cell fetches its
/// sliced execute envelope from the controller over velo, then runs it through the
/// ordinary single-process path (which the environment makes cell-aware).
#[cfg(feature = "velo")]
fn run_cell() -> ! {
    // A dedicated runtime just to fetch the envelope over velo; dropped before the
    // execute path builds its own thread-per-core runtime.
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
    let envelope_bytes = match runtime
        .block_on(aiperf_runtime::runner_protocol::cellular_cell::fetch_cell_envelope())
    {
        Ok(bytes) => bytes,
        Err(error) => {
            tracing::error!(
                error = format!("{error:#}"),
                "cell failed to fetch its envelope"
            );
            std::process::exit(2);
        }
    };
    // Ultimate spec §3 + §4.5: when dataset fan-out is enabled, build this cell's owned
    // index over the controller's broadcast and run the dispatch state machine over it
    // (a no-op otherwise). Done before dropping the fetch runtime, after START.
    if let Err(error) =
        runtime.block_on(aiperf_runtime::runner_protocol::cellular_cell::verify_dataset_fanout())
    {
        tracing::error!(error = format!("{error:#}"), "cell dataset fan-out failed");
        std::process::exit(2);
    }
    drop(runtime);
    // Stage G: before the ordinary execute path compiles the dataset, ship a
    // cross-host `file`/`path` dataset source from the controller and rewrite the
    // envelope to point at the landed cell-local copy. A no-op for synthetic /
    // inline-records / public datasets and for a same-host cell (which reads the
    // controller-local path directly).
    let envelope_bytes =
        match aiperf_runtime::runner_protocol::cellular_cell::download_cell_dataset_if_needed(
            envelope_bytes,
        ) {
            Ok(bytes) => bytes,
            Err(error) => {
                tracing::error!(
                    error = format!("{error:#}"),
                    "cell failed to download its dataset source"
                );
                std::process::exit(2);
            }
        };
    configure_dynosim_process_defaults(&envelope_bytes);
    let application = compose_stock_application();
    run_v2(&envelope_bytes, &application);
}

/// Runs this process as a tier-T2 aggregator: bind at the controller-assigned fixed
/// loopback port, collect its subtree of cells' folded stores, merge them, and ship
/// the one merged store up to the controller. Reads the run envelope (piped by the
/// controller on stdin) only for the merge `MetricsConfig`.
#[cfg(feature = "velo")]
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
    match runtime
        .block_on(aiperf_runtime::runner_protocol::cellular_aggregator::run_aggregator(&envelope))
    {
        Ok(()) => std::process::exit(0),
        Err(error) => {
            tracing::error!(error = format!("{error:#}"), "aggregator failed");
            std::process::exit(2);
        }
    }
}

/// Without the `velo` feature there is no cell transport, so `--aggregator` cannot run.
#[cfg(not(feature = "velo"))]
fn run_aggregator(_input: &[u8]) -> ! {
    tracing::error!(
        "aiperf was built without the `velo` feature; `--aggregator` (tier-T2 tree merge) requires it"
    );
    std::process::exit(2);
}

/// Without the `velo` feature there is no cell transport, so `--cell` cannot run.
#[cfg(not(feature = "velo"))]
fn run_cell() -> ! {
    tracing::error!(
        "aiperf was built without the `velo` feature; `--cell` (multi-cell runs) requires it"
    );
    std::process::exit(2);
}

/// Without the `velo` feature there is no cell transport, so a `cells>1` run
/// cannot be driven; fail closed with a typed execution envelope.
#[cfg(not(feature = "velo"))]
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

/// Runs this process as the cellular controller: drive the cells over velo, merge
/// their records into the single report, and emit the one terminal envelope.
#[cfg(feature = "velo")]
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
    let cell_count =
        aiperf_runtime::runner_protocol::cell_launcher::cell_count_from_envelope(envelope);
    // Compose the stock application so the merged-report export plane resolves the
    // built-in exporter sinks from the one unified `AIPerfRegistry`, exactly as the
    // single-process coordinator path does via `product_registry().exporters()`.
    let application = compose_stock_application();
    let exporters = application.product_registry().exporters();
    // Mirror run_v2's catch_unwind (see `handle_v2`): the controller runs the records
    // merge, native-v2 serialization, and the best-effort export plane inline in
    // run_cellular; a panic in any of them would otherwise unwind past this writer and
    // abort the controller (exit 101) with no envelope, so the parent would see a
    // crashed subprocess instead of a typed execution failure.
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        aiperf_runtime::runner_protocol::cellular_controller::run_cellular(
            envelope,
            cell_count,
            &report_path,
            exporters,
        )
    }));
    match outcome {
        Ok(Ok(outcome)) => {
            let mut provenance = BTreeMap::new();
            provenance.insert(
                "transport".to_owned(),
                envelope
                    .pointer("/run/cfg/transport/type")
                    .and_then(Value::as_str)
                    .unwrap_or("http")
                    .to_owned(),
            );
            // Label the workload by the run's kind (scheduled vs graph), not a
            // hardcoded "scheduled" — a graph cellular run must not report itself as
            // scheduled. Transport (http/grpc) is the orthogonal label set above.
            provenance.insert(
                "workload".to_owned(),
                CellularRunKind::detect(&envelope)
                    .workload_label()
                    .to_owned(),
            );
            provenance.insert("cells".to_owned(), outcome.cell_count.to_string());
            provenance.insert("record_count".to_owned(), outcome.record_count.to_string());
            write_json_line(
                &RunTerminalV2 {
                    protocol_version: RUNNER_PROTOCOL_V2,
                    event: "run_terminal",
                    benchmark_id,
                    success: true,
                    report_path: Some(outcome.report_path),
                    stage: None,
                    errors: Vec::new(),
                    diagnostic_artifacts: Vec::new(),
                    provenance,
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

/// Emit the cellular controller's execution-stage failure envelope, carrying the
/// error/panic message as a typed diagnostic (so the parent surfaces the reason
/// rather than an empty failure), then exit non-zero.
fn emit_cellular_failure(benchmark_id: Option<String>, code: &'static str, message: String) -> ! {
    tracing::error!(error = %message, "cellular run failed");
    write_json_line(
        &RunTerminalV2 {
            protocol_version: RUNNER_PROTOCOL_V2,
            event: "run_terminal",
            benchmark_id,
            success: false,
            report_path: None,
            stage: Some(RunnerFailureStageV2::Execution),
            errors: vec![diagnostic(code, message)],
            diagnostic_artifacts: Vec::new(),
            provenance: BTreeMap::new(),
        },
        1,
    );
}

fn compose_stock_application() -> RunnerApplication {
    let distribution_id = match current_distribution_id() {
        Ok(distribution_id) => distribution_id,
        Err(error) => {
            tracing::error!(error = %error, "failed to identify executing aiperf image");
            std::process::exit(2);
        }
    };
    match RunnerApplication::stock(distribution_id) {
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
    // The execution contract is exactly one terminal/validation JSONL line. A
    // panic anywhere in prepare/execute would otherwise unwind past this writer
    // and abort the child (exit 101) with no envelope, so the parent sees a
    // crashed subprocess instead of a typed failure. Convert a caught panic into
    // the corresponding v2 failure envelope.
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
        "aiperf execution panicked with a non-string payload".to_owned()
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
    let message = format!("aiperf internal panic: {message}");
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
