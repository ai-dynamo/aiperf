// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stdio entry point for one orchestrator-authored benchmark run.

use std::collections::BTreeMap;
use std::io::{self, Read, Write};

use aiperf::runner_protocol::protocol_v2::{
    RUNNER_PROTOCOL_V2, RunTerminalV2, RunValidationV2, RunnerDiagnosticV2, RunnerEnvelopeV2,
    RunnerFailureStageV2, RunnerOperationV2, ValidationCompletenessV2,
};
use aiperf::runner_protocol::redaction::redact_diagnostic;
use aiperf_runner::{RunnerApplication, current_distribution_id};
use serde::Deserialize;
use serde_json::{Value, value::RawValue};

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

/// Install the stderr `tracing` subscriber for the runner's diagnostics.
///
/// STDOUT is the stdio protocol channel to the Python orchestrator (JSONL
/// envelopes via `write_json_line`); the subscriber therefore writes only to
/// STDERR so it can never corrupt the protocol stream. ANSI is disabled because
/// Python inherits/pipes STDERR. The default filter is `warn` so the converted
/// diagnostics (all `warn`/`error` today) stay visible without configuration;
/// `AIPERF_RUNNER_LOG` overrides it with standard `EnvFilter` syntax.
fn init_tracing() {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_env("AIPERF_RUNNER_LOG")
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();
}

fn main() {
    init_tracing();
    let arguments = std::env::args_os().skip(1).collect::<Vec<_>>();
    if arguments.len() == 1 && arguments[0] == "--capabilities" {
        let application = compose_stock_application();
        write_json_line(&application.catalog(), 0);
    }
    let cell_mode = arguments.len() == 1 && arguments[0] == "--cell";
    let aggregator_mode = arguments.len() == 1 && arguments[0] == "--aggregator";
    if !arguments.is_empty() && !cell_mode && !aggregator_mode {
        tracing::error!("usage: aiperf-runner [--capabilities|--cell|--aggregator]");
        std::process::exit(2);
    }

    let mut input = Vec::new();
    if let Err(error) = io::stdin().read_to_end(&mut input) {
        tracing::error!(error = %error, "failed to read runner request from stdin");
        std::process::exit(2);
    }

    // A cell child runs its budget slice single-process, with the autonomous issuer
    // and per-cell sampler selected from the controller-set environment; it fetches
    // its sliced envelope over velo (not stdin).
    if cell_mode {
        run_cell();
    }

    // A tier-T2 aggregator collects a subtree of cells' folded stores, merges them,
    // and ships one merged store up to the controller. It reads the run envelope from
    // stdin (piped by the controller) for the merge config.
    if aggregator_mode {
        run_aggregator(&input);
    }

    // A non-cell execute request asking for more than one cell becomes the
    // controller: it drives the cells over velo and merges their records.
    if std::env::var(aiperf::cellular::partition::CELL_ID_ENV).is_err()
        && let Ok(envelope) = serde_json::from_slice::<Value>(&input)
        && envelope.pointer("/operation").and_then(Value::as_str) == Some("execute")
        && aiperf::runner_protocol::cell_launcher::cell_count_from_envelope(&envelope) > 1
    {
        run_controller(&envelope);
    }

    configure_dynosim_process_defaults(&input);
    let application = compose_stock_application();
    // The runner speaks only protocol v2. run_v2 rejects a non-v2 or malformed
    // request as a v2 failure envelope (a v1 request fails EnvelopeBootstrapV2
    // parsing and is reported as an invalid protocol-v2 request).
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
    let envelope_bytes =
        match runtime.block_on(aiperf::runner_protocol::cellular_cell::fetch_cell_envelope()) {
            Ok(bytes) => bytes,
            Err(error) => {
                tracing::error!(
                    error = format!("{error:#}"),
                    "cell failed to fetch its envelope"
                );
                std::process::exit(2);
            }
        };
    drop(runtime);
    // Stage G: before the ordinary execute path compiles the dataset, ship a
    // cross-host `file`/`path` dataset source from the controller and rewrite the
    // envelope to point at the landed cell-local copy. A no-op for synthetic /
    // inline-records / public datasets and for a same-host cell (which reads the
    // controller-local path directly).
    let envelope_bytes =
        match aiperf::runner_protocol::cellular_cell::download_cell_dataset_if_needed(
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
    match runtime.block_on(aiperf::runner_protocol::cellular_aggregator::run_aggregator(
        &envelope,
    )) {
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
        "aiperf-runner was built without the `velo` feature; `--aggregator` (tier-T2 tree merge) requires it"
    );
    std::process::exit(2);
}

/// Without the `velo` feature there is no cell transport, so `--cell` cannot run.
#[cfg(not(feature = "velo"))]
fn run_cell() -> ! {
    tracing::error!(
        "aiperf-runner was built without the `velo` feature; `--cell` (multi-cell runs) requires it"
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
        "aiperf-runner was built without the `velo` feature; multi-cell runs (cells>1) require it"
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
    let cell_count = aiperf::runner_protocol::cell_launcher::cell_count_from_envelope(envelope);
    // Compose the stock application so the merged-report export plane resolves the
    // built-in exporter sinks from the one unified `AIPerfRegistry`, exactly as the
    // single-process coordinator path does via `product_registry().exporters()`.
    let application = compose_stock_application();
    let exporters = application.product_registry().exporters();
    // Mirror run_v2's catch_unwind (see `handle_v2`): the controller runs the records
    // merge, native-v2 serialization, and the best-effort export plane inline in
    // run_cellular; a panic in any of them would otherwise unwind past this writer and
    // abort the controller (exit 101) with no envelope, so Python would see a crashed
    // subprocess instead of a typed execution failure.
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        aiperf::runner_protocol::cellular_controller::run_cellular(
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
            provenance.insert("workload".to_owned(), "scheduled".to_owned());
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
/// error/panic message as a typed diagnostic (so Python surfaces the reason rather
/// than an empty failure), then exit non-zero.
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
            tracing::error!(error = %error, "failed to identify executing aiperf-runner image");
            std::process::exit(2);
        }
    };
    match RunnerApplication::stock(distribution_id) {
        Ok(application) => application,
        Err(error) => {
            tracing::error!(
                error = format!("{error:#}"),
                "failed to compose executing aiperf-runner image"
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
