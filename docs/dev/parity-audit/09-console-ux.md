<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Console output, warnings, and process behavior parity audit

Domain: console output, warnings/advisories, logging, signals, exit codes.
Audited against the pre-existing backlog `docs/dev/python-rust-parity-gaps.md`
(dated 2026-07-17). Every finding is marked NEW, KNOWN(still-true), or
KNOWN(now-fixed).

**Python baseline:** `origin/main` at rev `bc359bf8fd`, read from
`/mnt/4tb/aiperf-parity-py-main/src/aiperf/`. Every `src/aiperf/...` citation
below is a baseline path and a baseline line number. Rust citations are unchanged
(`origin/main` carries no `rust/` tree). An earlier draft of this report cited a
local feature branch 4345 commits ahead of `origin/main`; the rebase is recorded
in "Baseline correction" below. No finding was withdrawn.

### Baseline correction

Of the files this report cites, all but three are byte-identical between the
branch and `bc359bf8fd`, including `controller/system_controller.py` (which
carries the evidence for findings 1, 2, 3, 4, and 7) and every console
advisory exporter. The three that differ do so only in line numbering or in code
this report does not cite:

| Cited file | Baseline vs. branch | Effect on this report |
|---|---|---|
| `exporters/console_metrics_exporter.py` | branch adds a 7-line `try/except` in `_get_title` after baseline line 240 | citations at or below line 232 unchanged; `_get_group_title` re-cited 251-259 → **244-252** |
| `common/environment.py` | unrelated additions above the cited blocks; all cited *values* identical | thresholds re-cited 975-1007 → **899, 914-930**; console width 1613-1616 → **1550-1553** |
| `plugin/plugins.yaml` | unrelated additions above the `console_exporter:` block; registrations identical | disclaimer re-cited 1327-1333 → **1353-1359** |
| `config/artifacts.py` | unrelated additions; `LOG_FOLDER`/`LOG_FILE` values identical | re-cited to **46-47** |

Separately, the branch rewrote `cli_runner/_single_run.py`, `cli_runner/__init__.py`,
`cli_runner/_multi_run.py`, `cli_runner/_preflight.py`, `entrypoint.py`, `cli.py`,
and `exporters/metrics_base_exporter.py`. Only one sentence in this report ever
depended on any of them — the description of the exit-code mechanism in
"Unverified", now re-derived against baseline (`common/bootstrap.py:320-334` plus
`cli_runner/_single_run.py:111-112,132`). The observable exit codes are the same
on both, so no finding changes. No finding cited `entrypoint.py` or `cli.py`,
whose branch deltas are pure additions and therefore describe behavior that does
not exist upstream.

### Finding status after rebase

| # | Finding | Sev | Status |
|---|---|---|---|
| 1 | Cancelled run indistinguishable from complete run | P0 | **STILL VALID** — notice confirmed upstream at `system_controller.py:1185`, `:1230` |
| 2 | Second Ctrl-C does nothing | P0 | **STILL VALID** — force-quit advertisement confirmed upstream at `system_controller.py:872` |
| 3 | Three pre-run advisories absent (incl. TLS) | P1 | **STILL VALID** — unchanged files |
| 4 | Nothing printed when every request fails | P1 | **STILL VALID** — rests on `system_controller.py`, not `cli_runner/` |
| 5 | `--show-trace-timing` shows nothing | P1 | **STILL VALID** — `http_trace_console_exporter.py` byte-identical |
| 6 | Logs stdout → stderr; `-v` scoping | P1 | **STILL VALID** — unchanged files |
| 7 | End-of-run footer dropped | P1 | **STILL VALID** — rests on `system_controller.py`, not `cli_runner/` |
| 8 | GPU console reduced to four unattributed rows | P1 | **STILL VALID** — unchanged files |
| 9 | Advertised threshold env vars never read | P1 | **CHANGED** — promoted P2 → P1, evidence re-cited |
| 10 | `decode_duration` renders as raw tag | P2 | **STILL VALID** — Rust-side only |
| 11 | Counters repeat their value in min/max | P2 | **STILL VALID** — `metrics/accumulator.py` byte-identical |
| 12 | Dev-mode console surfaces unreachable | P2 | **STILL VALID** — unchanged files |
| 13 | Aggregate cosmetic bullet | P2 | **STILL VALID** — line numbers rebased |

Totals: 13 surviving (2 P0, 7 P1, 4 P2), 0 withdrawn, 1 changed.

## Summary

The two headline advisories the backlog worried about — OSL mismatch and token
usage discrepancy — are at full parity in Rust, including panel titles, body
text, and thresholds. The real regressions are elsewhere. First, a Rust run that
the user cancels with Ctrl-C prints a console report that is byte-indistinguishable
from a complete run: Python appends "The profile run was cancelled early. Results
shown may be incomplete or inaccurate." and tags the duration line
"(cancelled early)", while Rust emits no cancellation marker anywhere in its
console output and still exits 0. Second, a second Ctrl-C is a no-op in Rust
(tokio's handler has replaced the default SIGINT disposition and the listener
task has already completed), so Python's force-quit escape hatch is gone and a
stuck drain can only be killed with SIGKILL from another terminal. Third, three
pre-run advisories are silently absent: `--osl` without `ignore_eos`/`min_tokens`,
accuracy without `temperature=0`, and — security-relevant — "SSL certificate
verification is DISABLED". Fourth, the whole footer block (CLI command,
benchmark duration, exported-file list, log-file path) is gone, and when every
request fails Rust prints nothing at all to the terminal even though it wrote a
complete `profile_export_console.txt` with the error table and API-error advisory.
Fifth, the OSL and usage advisories print an instruction to set
`AIPERF_METRICS_OSL_MISMATCH_PCT_THRESHOLD`, which nothing under `rust/` reads —
the thresholds are compile-time constants, so the only tuning surface either
advisory offers is inert. Finally, logs moved from stdout to stderr, which is the
better design but will break any script that greps the pipeline's stdout.

## Advisory/warning inventory

| Python advisory | Trigger condition | Rust equivalent? | path:line |
|---|---|---|---|
| OSL Mismatch Warning panel | `osl_mismatch_count.avg > 0` and `request_count > 0` | Yes — same title, same body lines, same thresholds | Py `src/aiperf/exporters/console_osl_mismatch_exporter.py:34-117` / Rs `rust/runtime/src/export/console_txt.rs:253-300` |
| Token Count Discrepancy Warning panel | `usage_discrepancy_count.avg > 0` and `request_count > 0` | Yes — same title and body | Py `src/aiperf/exporters/console_usage_discrepancy_exporter.py:31-89` / Rs `rust/runtime/src/export/console_txt.rs:303-340` |
| API error advisory: `max_completion_tokens` rejected | error message contains `extra_forbidden` + `max_completion_tokens` + `Extra inputs are not permitted` | Yes — same title/body; Rust adds a second Dynamo detector | Py `src/aiperf/exporters/console_api_error_exporter.py:29-77` / Rs `rust/runtime/src/export/console_txt.rs:343-352` |
| Error Summary table | `results.error_summary` non-empty | Yes — same title, same 4 columns, same `N/A` fallbacks | Py `src/aiperf/exporters/console_error_exporter.py:17-50` / Rs `rust/runtime/src/export/console_txt.rs:154` |
| Spec-decode table + pooled histogram | any request carried acceptance stats | Yes | Py `src/aiperf/exporters/console_spec_decode_exporter.py` / Rs `rust/runtime/src/export/console_txt.rs:160,628-635` |
| Pre-run: `--osl` without `ignore_eos`/`min_tokens` | `dataset.prompts.osl` set and neither extra input truthy | **No counterpart** | Py `src/aiperf/cli_utils.py:154-162`, called at `src/aiperf/controller/system_controller.py:103-104` |
| Pre-run: accuracy without `temperature=0` | `accuracy.enabled` and `extra.temperature != 0` | **No counterpart** | Py `src/aiperf/cli_utils.py:165-172`, called at `src/aiperf/controller/system_controller.py:106-107` |
| "SSL certificate verification is DISABLED — this is insecure" | `ssl_verify` false | **No counterpart** (Rust installs the no-op verifier silently) | Py `src/aiperf/controller/system_controller.py:318-321` / Rs `rust/runtime/src/transport/http/client/connection.rs:356-359` |
| Cache-reporting hint | usage prompt tokens reported but no `cached_tokens` total | **No counterpart** | Py `src/aiperf/metrics/cache_reporting_hint.py:24-30`, printed at `src/aiperf/exporters/console_metrics_exporter.py:111-112` |
| GPU vendor disclaimer panel | GPU telemetry enabled | **No counterpart** | Py `src/aiperf/exporters/console_gpu_vendor_disclaimer_exporter.py`, registered `src/aiperf/plugin/plugins.yaml:1353-1359` |
| Cancellation panel (first Ctrl-C) | first SIGINT | **No counterpart** | Py `src/aiperf/controller/system_controller.py:856-880` / Rs `rust/runtime/src/timing/phase/orchestrator.rs:444-479` (cancels silently) |
| Force-quit panel (second Ctrl-C) | second SIGINT | **No counterpart** — second signal has no effect at all | Py `src/aiperf/controller/system_controller.py:882-903` / Rs `rust/cli/src/signals.rs:48-58` |
| "cancelled early" console notice | `_was_cancelled` at report time | **No counterpart** | Py `src/aiperf/controller/system_controller.py:1183-1186`, `1229-1230` / Rs `rust/runtime/src/export/console_txt.rs:145-180` (no `was_cancelled` reference) |
| Developer Mode panel | `AIPERF_DEV_MODE` | **No counterpart** | Py `src/aiperf/cli_utils.py:138-151`, called `src/aiperf/controller/system_controller.py:98-100` |
| Unreachable telemetry / server-metrics endpoints | configured minus reachable non-empty | Partially (telemetry status is logged, not in the console report) | Py `src/aiperf/controller/system_controller.py:560-568` |
| Startup error panel | any startup exception | Plain one-line stderr message instead of a panel; same exit code 1 | Py `src/aiperf/cli_utils.py:26-58` / Rs `rust/cli/src/main.rs:49-52` |
| GPU telemetry per-GPU table | telemetry enabled and results present | **No counterpart** | Py `src/aiperf/exporters/gpu_telemetry_console_exporter.py:30-45` |
| HTTP Trace Timing table | `--show-trace-timing` | **No counterpart** — all 14 `http_req_*` tags are `group: none` | Py `src/aiperf/exporters/http_trace_console_exporter.py:10-25` / Rs `rust/runtime/resources/metric_metadata.json` |
| Internal / Experimental metrics tables | `AIPERF_DEV_MODE` + `SHOW_INTERNAL/EXPERIMENTAL_METRICS` | **No counterpart** — `dev` is hardcoded `false` | Py `src/aiperf/exporters/internal_metrics_console_exporter.py:18-22` / Rs `rust/runtime/src/config/model/export.rs:477` |
| GPU power-efficiency tables (NVIDIA/AMD) | vendor totals present | Partially — 4 of 12 rows, unattributed, folded into the main table | Py `src/aiperf/exporters/console_power_efficiency_exporter.py:8-39` / Rs `rust/runtime/resources/metric_metadata.json` |
| Accuracy console table | accuracy results present | **No counterpart** (CSV sink only) | Py `src/aiperf/accuracy/accuracy_console_exporter.py` / Rs `rust/runtime/src/export/mod.rs:414` |

## Findings

### 1. A cancelled Rust run's console report is indistinguishable from a complete run

**Severity:** P0
**Status:** NEW (the backlog's P2.9 covers "quiet behavior" and live progress, not
the cancellation marker)

**Python evidence** — `src/aiperf/controller/system_controller.py:1183-1186`:

```python
        if self._was_cancelled:
            console.print(
                "[italic yellow]The profile run was cancelled early. Results shown may be incomplete or inaccurate.[/italic yellow]"
            )
```

and `src/aiperf/controller/system_controller.py:1226-1231`:

```python
        duration = self._profile_results.get(BenchmarkDurationMetric.tag)
        if duration:
            duration_str = f"[bold green]{BenchmarkDurationMetric.header}[/bold green]: {duration.avg:.2f} {duration.unit}"
            if self._was_cancelled:
                duration_str += " [italic yellow](cancelled early)[/italic yellow]"
            console.print(duration_str)
```

**Rust evidence** — `rust/runtime/src/export/console_txt.rs:145-180` is the complete
section list for the console artifact, and it never reads
`report.summary.was_cancelled`:

```rust
pub(crate) fn render_console_txt(report: &NativeReport, cfg: &ConsoleTxtExportConfig) -> String {
    ...
    for warning in detect_api_errors(report) { ... }
    if let Some(table) = error_summary_table(report, width) { ... }
    if let Some(tables) = metrics_tables(report, cfg, width) { ... }
    if let Some(table) = speculative_decoding_table(report, cfg, width) { ... }
    if let Some(warning) = detect_usage_discrepancy(report) { ... }
    if let Some(warning) = detect_osl_mismatch(report) { ... }
```

The flag exists on the report (`rust/runtime/src/metrics_core/report.rs:798`) and
reaches the JSON and MLflow sinks (`rust/runtime/src/export/genai_perf.rs:573-574`,
`rust/runtime/src/export/mlflow.rs:275-276`) — only the console loses it.
`rust/cli/src/render.rs:11-22` echoes that file verbatim and adds nothing.

**Observable user impact:** Ctrl-C a Rust run at 40% and the terminal shows the
same table shape, the same headers, and a plausible set of latency/throughput
numbers as a completed run, with no marker distinguishing the two, and the process
exits 0 (`rust/cli/src/profile.rs:513-521`, reached because
`rust/runtime/src/engine/coordinator.rs:381-394` sets `success: true` for a
cancelled-but-drained run). Anyone reading a scrollback, a CI log, or a pasted
result cannot tell the run was truncated. Python made it impossible to miss.

**Confidence:** High. Verified by reading the full Rust console render path and
confirming no `was_cancelled` reference exists in it.

### 2. Second Ctrl-C does nothing in Rust; Python's force quit is gone

**Severity:** P0
**Status:** NEW

**Python evidence** — `src/aiperf/controller/system_controller.py:832-854`:

```python
    async def _handle_signal(self, sig: int) -> None:
        if self._was_cancelled:
            # SECOND Ctrl+C - Force quit immediately
            self._print_force_quit_warning()
            self.warning(f"Force quit requested (signal {sig})")
            await self._kill()
            return

        # FIRST Ctrl+C - Graceful cancel with warning
        self._print_cancel_warning()
        self.warning(f"Graceful shutdown requested (signal {sig})")
        await self._cancel_profiling()
```

The first panel explicitly advertises the second stage
(`system_controller.py:872`): `"[dim]Press Ctrl+C again to force quit immediately[/dim]"`.

**Rust evidence** — `rust/runtime/src/timing/phase/orchestrator.rs:437-443`
states the design, and the implementation matches:

```rust
/// Only `tokio::signal` (async, driven by the runtime's signal driver) is used
/// ... On the first delivered signal the active phase is
/// cancelled once; later signals are ignored because the task has completed.
```

The parent side re-forwards every signal as SIGINT to the same child forever
(`rust/cli/src/signals.rs:48-58`):

```rust
    std::thread::spawn(move || {
        loop {
            let _sig = set.wait();
            let pid = child_for_thread.get();
            if pid > 0 {
                let _ = kill(Pid::from_raw(pid), Signal::SIGINT);
            } else {
                std::process::exit(130);
            }
        }
    });
```

**Observable user impact:** Because `tokio::signal` installs a handler, SIGINT no
longer carries its default terminating disposition in the child; the listener task
has completed after the first signal, so signals two through N are delivered and
discarded. A drain that hangs (an unresponsive endpoint holding open streams) can
only be stopped by finding the PID from another terminal and sending SIGKILL.
Python users have been taught by the on-screen panel that a second Ctrl-C works.
No message tells them it stopped working.

**Confidence:** High for "the second signal does not cancel or terminate" (both
handler sites read). Medium on whether some other timeout eventually reaps a
pathological drain — see Unverified.

### 3. Three pre-run advisories are silently absent, including the TLS-disabled warning

**Severity:** P1
**Status:** NEW

**Python evidence** — all three fire at controller construction,
`src/aiperf/controller/system_controller.py:97-107`:

```python
        if Environment.DEV.MODE:
            print_developer_mode_warning()

        # EOS may cause server to stop early, producing misleading OSL results
        if self._should_warn_osl_without_ignore_eos():
            warn_osl_without_ignore_eos()

        if self._should_warn_accuracy_temperature():
            warn_accuracy_temperature()
```

with the OSL text at `src/aiperf/cli_utils.py:154-162` and the accuracy-temperature
text at `src/aiperf/cli_utils.py:165-172`. The TLS warning is at
`src/aiperf/controller/system_controller.py:318-321`:

```python
        if not Environment.HTTP.SSL_VERIFY:
            self.warning(
                "SSL certificate verification is DISABLED - this is insecure. This should only be used for testing in a trusted environment."
            )
```

**Rust evidence** — no counterpart for any of the three.
`rg 'ignore_eos' rust/cli/src rust/runtime/src` returns only payload composition
(`rust/runtime/src/config/resolve.rs:1780-1834`, `rust/runtime/src/agentx/wire.rs:97`)
and the post-run OSL panel's fix suggestion
(`rust/runtime/src/export/console_txt.rs:282`). TLS verification is disabled with
no log line at all — `rust/runtime/src/transport/http/client/connection.rs:356-359`:

```rust
    if !ssl_verify {
        cfg.dangerous()
            .set_certificate_verifier(Arc::new(NoCertificateVerification { provider }));
    }
```

**Observable user impact:** The OSL advisory is the "AIPerf will tell me my
benchmark is wrong" case fired *before* wasting the run; Rust's post-run panel only
fires if the server actually short-generated, so a server that silently truncates
in a way the mismatch heuristic misses now produces no signal at either end. The
TLS case is worse in kind than in frequency: `--ssl-verify false` in Rust
downgrades the connection to unauthenticated with zero acknowledgement, so a
copy-pasted flag in a shared config can persist into a production measurement
against an untrusted endpoint without anyone seeing a warning in the log.

**Confidence:** High.

### 4. When every request fails, Rust prints nothing to the terminal

**Severity:** P1
**Status:** partly KNOWN(now-fixed) — P0.6 "A single native run can succeed when
every request fails" is fixed; the console-silence half is NEW

**Python evidence** — `src/aiperf/controller/system_controller.py:1136-1157`
suppresses the metrics tables and prints the error panel plus the log-file path
through `_print_exit_errors_and_log_file()`:

```python
        results = self._profile_results.results
        if results.successful_request_count == 0 and results.error_request_count > 0:
            self.error(
                f"All {results.error_request_count} inference request(s) failed; "
                "no successful responses were collected."
            )
            self._exit_errors.append(...)
            self._print_exit_errors_and_log_file()
            return
```

**Rust evidence** — the exit code is now correct.
`rust/runtime/src/engine/coordinator.rs:357-379` returns `success: false`,
`exit_code: 1`, and an `all_requests_failed` diagnostic. But
`rust/cli/src/profile.rs:513-533` gates *all* terminal output on success:

```rust
    let code = if terminal.success {
        tracing::info!("Native AIPerf run completed");
        if let Some(path) = &terminal.report_path {
            crate::render::print_console_summary(path);
            ...
    } else {
        let detail = terminal.error.as_deref().unwrap_or("native benchmark failed");
        tracing::error!("Native AIPerf run failed: {detail}");
```

**Observable user impact:** The run *did* write a `profile_export_console.txt`
containing the Error Summary table and, when applicable, the
`max_completion_tokens` advisory panel that tells the user exactly which flag to
change (`rust/runtime/src/export/console_txt.rs:135-137`, written before the
terminal envelope is built). The user never sees it — they get one
`tracing::error!` line on stderr and have to know to open the artifact directory.
Python printed the actionable panel inline. Rust also prints no log-file path in
this case, where Python's `_print_exit_errors_and_log_file` does.

**Confidence:** High.

### 5. `--show-trace-timing` no longer shows trace timing in the console

**Severity:** P1
**Status:** NEW

**Python evidence** — `src/aiperf/exporters/http_trace_console_exporter.py:10-25`:

```python
class HttpTraceConsoleExporter(ConsoleMetricsExporter):
    """Console exporter for HTTP trace timing metrics (k6-style breakdown).

    Gated on the `--show-trace-timing` user config flag.
    """
    title = "NVIDIA AIPerf | HTTP Trace Timing"
    require_flags = MetricFlags.HTTP_TRACE_ONLY
    ...
    def _check_enabled(self, exporter_config: ExporterConfig) -> None:
        if not exporter_config.cfg.artifacts.show_trace_timing:
            raise ConsoleExporterDisabled(...)
```

**Rust evidence** — the flag exists (`rust/cli/src/flags.rs:553-555`) and is
lowered, but only into the per-record artifact columns.
`rust/runtime/src/config/resolve.rs:1627`:

```rust
                trace: (inputs.export_trace || inputs.show_trace_timing) && per_record,
```

Nothing else consumes it; there is no console branch. Independently, all 14
`http_req_*` tags in `rust/runtime/resources/metric_metadata.json` carry
`"group": "none"`, e.g. `http_req_waiting → {"display_order": 2040, "group": "none",
"header": "HTTP Waiting (TTFB)"}`, and `rust/runtime/src/export/console_txt.rs:587-592`
drops any row whose group is not in `GROUP_ORDER`:

```rust
        let Some(slot) = grouped
            .iter_mut()
            .find(|(candidate, _)| *candidate == group)
        else {
            continue; // group `none` (hidden) or otherwise not rendered.
        };
```

`GROUP_ORDER` (`console_txt.rs:494-504`) has no `none` entry by design.

**Observable user impact:** A user debugging connection overhead passes
`--show-trace-timing` (a flag whose name is a promise about the console), gets no
warning and no table — the flag's only effect is extra JSONL columns. This is
distinct from the "loudly refused flag" category: `--show-trace-timing` is *not*
in `UNIMPLEMENTED_FLAGS` (`rust/cli/src/profile.rs:383-416`), so it produces no
compatibility warning either.

**Confidence:** High.

### 6. Logs moved from stdout to stderr, and `-v`/`-vv` no longer raise dependency levels

**Severity:** P1
**Status:** NEW (backlog P1.6 covers env-var name splits, not stream destination)

**Python evidence** — both handler paths target stdout.
`src/aiperf/common/logging.py:200-211`:

```python
    if is_tty():
        console_handler = CustomRichHandler(..., console=Console(), ...)
    else:
        console_handler = _create_basic_handler(level)
    logging.root.addHandler(console_handler)
```

and `src/aiperf/common/logging.py:51-61`:

```python
def _create_basic_handler(level: str | int) -> logging.StreamHandler:
    """Create a basic non-rich StreamHandler for non-TTY environments.

    Uses sys.stdout to match CustomRichHandler's Console() which also defaults to stdout.
    """
    handler = logging.StreamHandler(sys.stdout)
```

Python also sets the level on the *root* logger
(`src/aiperf/common/logging.py:197-198`), so `-vv` raises dependency loggers too.

**Rust evidence** — `rust/cli/src/logging.rs:43-47`:

```rust
    let console_layer = tracing_subscriber::fmt::layer()
        .with_writer(io::stderr)
        .with_ansi(false)
```

and `rust/cli/src/logging.rs:104-119` scopes the verbose flags to AIPerf targets
only, pinning dependencies at `warn`:

```rust
const VERBOSE_TARGETS: [&str; 4] = ["aiperf", "aiperf_cli", "aiperf_runtime", "aiperf_e2e_tests"];
const DEP_LEVEL: &str = "warn";
```

**Observable user impact:** `aiperf profile ... > run.log` used to capture the
whole session including logs; in Rust it captures only the final console report
and the logs go to the terminal. Conversely `2>/dev/null` used to hide nothing
and now hides all logs. Any wrapper that greps stdout for a log substring
silently matches nothing. The verbose scoping is defensible on its merits (the
code documents a real dependency-flood problem) but it means `-vv` no longer
surfaces `rustls`/`tokenizers` traces that a Python user could get; `AIPERF_LOG`
is the replacement and is not mentioned in any output. Default level is INFO on
both sides, and the log file is `<artifact_dir>/logs/aiperf.log` on both
(`src/aiperf/config/artifacts.py:46-47` `LOG_FOLDER`/`LOG_FILE` vs
`rust/cli/src/logging.rs:75-80`).

**Confidence:** High.

### 7. The entire end-of-run footer block is dropped

**Severity:** P1
**Status:** NEW

**Python evidence** — `src/aiperf/controller/system_controller.py:1178-1189`:

```python
        console.print()
        self._print_cli_command(console)
        self._print_benchmark_duration(console)
        self._print_exported_file_infos(exporter_manager, console)
        self._print_log_file_info(console)
```

producing four labelled lines: `CLI Command: …`
(`system_controller.py:1212-1217`), `Benchmark Duration: 12.34 sec`
(`:1219-1231`), one `<ExportType>: <path>` line per written artifact
(`:1202-1210`), and `Log File: …` (`:1191-1200`).

**Rust evidence** — `rust/cli/src/render.rs:11-22` prints only the file body:

```rust
pub fn print_console_summary(report_path: &str) {
    ...
    let console = dir.join("profile_export_console.txt");
    if let Ok(text) = std::fs::read_to_string(&console) {
        print!("{text}");
```

and the file body ends at the OSL panel (`console_txt.rs:166-168`). `Benchmark
Duration` is deliberately not a table row on either side — Python sets
`console_group = MetricConsoleGroup.NONE`
(`src/aiperf/metrics/types/benchmark_duration_metric.py:28`) and Rust's metadata
sets `"group": "none"` — so with the footer gone the value appears nowhere in
Rust's console output at all. `rg 'Log File|CLI Command|Benchmark Duration'` over
`rust/cli/src` and `rust/runtime/src` finds no counterpart string.

**Observable user impact:** Four concrete losses. The wall-clock duration of the
benchmark is no longer printed anywhere the user looks. There is no printed path
to the artifact directory or the log file, so a user whose run produced a warning
has nothing to `cat`. And the reproduce-this-run command line is gone, which was
the thing people copied out of pasted results.

### 8. GPU output is reduced to four unattributed rows; per-GPU table and vendor disclaimer are gone

**Severity:** P1
**Status:** KNOWN(still-true) — backlog P2.9 lists "GPU … sections"; severity is
understated there given the vendor-comparability disclaimer

**Python evidence** — 12 NVIDIA and 12 AMD power-efficiency metrics render in two
vendor-attributed tables. `src/aiperf/exporters/console_power_efficiency_exporter.py:8-25`:

```python
class ConsoleNvidiaPowerEfficiencyExporter(ConsoleMetricsExporter):
    title = "GPU Power Efficiency (NVIDIA)"
    console_groups = (MetricConsoleGroup.GPU_POWER_EFFICIENCY_NVIDIA,)
    split_by_group = False
    STAT_COLUMN_KEYS = ["avg"]
```

with tags `nvidia_energy_delay_product`, `nvidia_performance_per_watt`,
`nvidia_output_tps_per_watt`, `nvidia_goodput_per_watt`, `nvidia_average_gpu_power`,
`nvidia_total_gpu_energy`, `nvidia_total_gpu_power`, `nvidia_energy_per_total_token`,
`nvidia_energy_per_output_token`, `nvidia_energy_per_request`,
`nvidia_output_tokens_per_joule`, `nvidia_energy_per_user`
(`src/aiperf/metrics/types/power_efficiency_metrics.py:64-212`, AMD mirror at
`:217-355`). The per-GPU telemetry table is
`src/aiperf/exporters/gpu_telemetry_console_exporter.py:30-45`, and the
comparability disclaimer that precedes both is
`src/aiperf/exporters/console_gpu_vendor_disclaimer_exporter.py` (registered ahead
of the GPU blocks, `src/aiperf/plugin/plugins.yaml:1353-1359`).

**Rust evidence** — `rust/runtime/resources/metric_metadata.json` contains exactly
four GPU/energy console entries, all vendor-neutral and all folded into the main
table:

```
energy_per_user          {"display_order": 903, "group": "default", "header": "Energy per User"}
output_tokens_per_joule  {"display_order": 902, "group": "default", "header": "Output Tokens per Joule"}
total_gpu_energy         {"display_order": 901, "group": "default", "header": "Total GPU Energy"}
total_gpu_power          {"display_order": 900, "group": "default", "header": "Total GPU Power"}
```

(matching `rust/runtime/src/metrics_core/catalog.rs:250-253,757-810`.) There is no
per-GPU section and no disclaimer in `render_console_txt`
(`rust/runtime/src/export/console_txt.rs:145-180`); the telemetry data reaches
`gpu_telemetry_export.jsonl` only.

**Observable user impact:** Eight of twelve efficiency rows disappear
(energy-delay product, performance/watt, output TPS/watt, goodput/watt, average
GPU power, energy per total token, energy per output token, energy per request).
The four survivors lose vendor attribution, so a mixed or unknown-vendor result no
longer tells the reader which vendor's counters produced the number — which is
precisely what the removed disclaimer existed to prevent people from doing
cross-platform. Per-GPU utilization/temperature/power breakdowns and the
"N of M endpoints reachable" status line are terminal-invisible.

**Confidence:** High on the row set and the missing sections.

### 9. OSL/usage thresholds are not configurable, yet the advisory tells you to configure them

**Severity:** P1 (promoted from P2 at baseline correction: this is the only
mechanism by which a user can act on either advisory, and the product prints
instructions for using it)
**Status:** CHANGED — the underlying gap is KNOWN(still-true) from backlog P1.6
("Rust hardcodes discrepancy thresholds that Python reads from environment"); the
self-contradicting advisory text and the severity are new here

**Python evidence** — `src/aiperf/common/environment.py:914-930`, under the
`env_prefix="AIPERF_METRICS_"` declared at `src/aiperf/common/environment.py:899`:

```python
    USAGE_PCT_DIFF_THRESHOLD: float = Field(ge=0.0, le=100.0, default=10.0, ...)
    OSL_MISMATCH_PCT_THRESHOLD: float = Field(ge=0.0, le=100.0, default=5.0, ...)
    OSL_MISMATCH_MAX_TOKEN_THRESHOLD: int = Field(ge=1, default=50, ...)
```

**Rust evidence** — the values are compile-time constants
(`rust/runtime/src/export/console_txt.rs:36` is
`const OSL_MISMATCH_PCT_THRESHOLD: f64 = 5.0;`, with the token and usage
constants at `:38` and `:40`) and the accumulator's configurable
fields (`rust/runtime/src/metrics_core/accumulator.rs:157-162`) are never assigned
anything but their defaults — `rg 'osl_mismatch_threshold_pct|usage_diff_threshold_pct'`
across `rust/` matches only `accumulator.rs` itself. Meanwhile the panel text
instructs the user to set the env var
(`rust/runtime/src/export/console_txt.rs:292-293`):

```rust
        format!("  - Adjust: AIPERF_METRICS_OSL_MISMATCH_PCT_THRESHOLD={pct}"),
        format!("  - Adjust: AIPERF_METRICS_OSL_MISMATCH_MAX_TOKEN_THRESHOLD={max_tokens}"),
```

There is no `env::var` read of any of the three names anywhere under `rust/`: the
string `AIPERF_METRICS_OSL_MISMATCH_PCT_THRESHOLD` occurs only in the advisory it
prints at `console_txt.rs:292` and in two golden fixtures.

**Observable user impact:** A user follows AIPerf's own printed instruction, sets
`AIPERF_METRICS_OSL_MISMATCH_PCT_THRESHOLD=20`, and the warning keeps firing at 5%
with no error and no acknowledgement. This is a knob the product actively
advertises and silently ignores, and it is the only tuning surface either advisory
offers — so a user whose server legitimately short-generates has no way to quiet
the panel and no way to tighten it. Note Rust *does* honor a sibling variable in
the same namespace — `AIPERF_METRICS_SKETCH`
(`rust/runtime/src/config/resolve.rs:2147-2149`) — which makes the namespace look
live and the silence more misleading than an unrecognized prefix would be.

**Confidence:** High.

### 10. `decode_duration` renders as a raw tag in the wrong table position

**Severity:** P2
**Status:** NEW

**Python evidence** — the metric has a header, a display order, and a console
group like any other row; Rust's own catalog agrees
(`rust/runtime/src/metrics_core/catalog.rs`, header "Decode Duration", order 350).

**Rust evidence** — the console does not read the catalog. It reads the frozen
resource, and that resource has no `decode_duration` key:

```
$ python3 -c "import json; print(json.load(open('rust/runtime/resources/metric_metadata.json'))['console_metrics'].get('decode_duration'))"
None
```

`rust/runtime/src/export/console_txt.rs:594-596` then falls back to the raw tag
and a last-place sort:

```rust
        // Unregistered metrics use the raw snake-case tag as their header.
        let header: &str = meta.map_or(tag.as_str(), |meta| meta.header.as_str());
        let display_order = meta.and_then(|meta| meta.display_order).unwrap_or(u32::MAX);
```

**Observable user impact:** The row appears as `decode_duration (ms)` at the
bottom of the default table instead of `Decode Duration (ms)` at order 350. The
same fallback will hit every future metric added to the catalog without a matching
edit to the resource — the resource is documented as intentionally not
regenerated from the catalog, so there is no mechanism keeping them in step.

**Confidence:** High on the rendering; the resource file is the runtime source
(`rust/runtime/src/config/model/export.rs:479`).

### 11. Counter and scalar metrics repeat their value in the min/max columns

**Severity:** P2
**Status:** NEW

**Python evidence** — a single-valued metric gets only `avg`
(`src/aiperf/metrics/accumulator.py:411-418`):

```python
            elif isinstance(value, (int, float)):
                output[tag] = MetricResult(
                    tag=tag, header=mc.header, unit=str(mc.unit), avg=value, count=1,
                )
```

and every unset stat renders as `N/A`
(`src/aiperf/exporters/console_metrics_exporter.py:220-223`).

**Rust evidence** — `rust/runtime/src/export/console_txt.rs:937-944`:

```rust
        ReportStats::Scalar(scalar) => match key {
            "avg" | "min" | "max" => value_f64(&scalar.value),
            _ => None,
        },
        ReportStats::Counter(counter) => match key {
            "avg" | "min" | "max" => value_f64(&counter.total),
            _ => None,
        },
```

The committed golden shows the effect
(`rust/runtime/src/export/console_txt/golden/full_render.regression.txt`):

```
│        request_count (requests) │  10.00 │  10.00 │  10.00 │    N/A │    N/A │    N/A │  N/A │
```

**Observable user impact:** `Request Count` and every other counter now print
`avg 10.00 | min 10.00 | max 10.00` where Python printed `10.00 | N/A | N/A`. A
reader scanning the min/max columns sees what looks like a degenerate
distribution rather than "this is a single number", and a script that treated a
non-`N/A` `min` column as "this row is a distribution" now misclassifies every
counter.

**Confidence:** High.

### 12. Dev-mode console surfaces are unreachable, and `--ui`/dev toggles are inert

**Severity:** P2
**Status:** KNOWN(still-true) — backlog P2.9 and P1.4

**Python evidence** — `src/aiperf/exporters/internal_metrics_console_exporter.py:18-22`
and the experimental sibling gate on `Environment.DEV.MODE` plus
`SHOW_INTERNAL_METRICS` / `SHOW_EXPERIMENTAL_METRICS`; the Developer Mode panel is
`src/aiperf/cli_utils.py:138-151`.

**Rust evidence** — `rust/runtime/src/config/model/export.rs:474-481` hardcodes
`dev: false`, so the internal/experimental filter at
`rust/runtime/src/export/console_txt.rs:576-580` always drops those rows; there is
no reader of `AIPERF_DEV_MODE` in the console path. `--ui`/`--ui-type` and
`--stats-interval` are at least *loudly* refused (`rust/cli/src/profile.rs:410-411`
plus the aggregated `warn_unimplemented_flags` at `:419-432`), so those are noted
here only for completeness rather than as silent changes.

**Observable user impact:** Limited to developers; a dev-mode user gets neither
the mode banner nor the internal/experimental tables and no message explaining it.

**Confidence:** High.

### 13. Aggregate cosmetic bullet

**Severity:** P2
**Status:** NEW (aggregate)

Rust always emits fixed-width (default 140) box-drawn tables with ANSI disabled
(`rust/cli/src/logging.rs:45`, `rust/runtime/src/config/model/export.rs:476`),
while Python renders the artifact at the fixed width *and* re-renders a second
copy at the live terminal width with color when stdout is an interactive
non-`NO_COLOR` terminal (`src/aiperf/exporters/exporter_manager.py:325-378`). The
colors, the Rich vs. hand-rolled box characters, and the panel border styles are
out of scope; the one nuance worth recording is that on a terminal narrower than
140 columns Rust's tables hard-wrap where Python's would have fit, and Rust
ignores `NO_COLOR` because it never emits color. The width knob itself is at
parity: both read `AIPERF_UI_CONSOLE_EXPORT_WIDTH` with default 140
(`src/aiperf/common/environment.py:1550-1553`,
`rust/runtime/src/config/model/export.rs:29-35`).

## Checked and consistent

- **OSL mismatch advisory**: panel title, all 14 body lines, the
  `min(requested x 5%, 50)` threshold expression, the `{count} of {total} requests
  ({pct:.1}%)` phrasing, comma-grouped counts, and the `N/A` average-mismatch
  fallback all match byte-for-byte modulo Rich markup.
  (`console_osl_mismatch_exporter.py:101-117` vs `console_txt.rs:272-295`.)
- **Usage discrepancy advisory**: same, including the numbered investigation steps
  and the `10%` threshold. (`console_usage_discrepancy_exporter.py:75-89` vs
  `console_txt.rs:317-334`.)
- **`max_completion_tokens` API advisory**: identical title, problem statement,
  two causes, two investigation steps, three fixes.
  (`console_api_error_exporter.py:56-75` vs `console_txt.rs:343-352` and the
  committed golden `golden/full_render.regression.txt`.)
- **Error Summary table**: same title string (`NVIDIA AIPerf | Error Summary`),
  same four columns in order (Code, Type, Message, Count), same right/right/left/right
  justification, same `N/A` for absent code/type, same comma-grouped count.
- **Metrics-table columns and numeric formatting**: both use
  `Metric | avg | min | max | p99 | p90 | p50 | std`
  (`console_metrics_exporter.py:39` vs `console_txt.rs:507`), both format numbers
  as `{:,.2f}` (`console_metrics_exporter.py:228` vs `comma_2dp`,
  `console_txt.rs:243-245`), both print `N/A` for absent stats, and both move the
  unit to a second line when the header exceeds 30 characters
  (`console_metrics_exporter.py:217` vs `console_txt.rs:600-605`).
- **Group split and titles**: same group set and render order, same
  `<base>: <Group>` title convention, same endpoint-derived base title
  (`console_metrics_exporter.py:55-64,244-252` vs `console_txt.rs:494-504,900-912`).
- **Metric headers, console groups, display order, and internal/experimental/
  error-only flags**: consistent across all 108 tags present in both Python's
  `MetricRegistry` and Rust's `metric_metadata.json` (programmatic diff, 0
  mismatches). Displayed units consistent across the 106 tags carrying explicit
  unit-conversion rules (0 mismatches).
- **Hidden rows**: `benchmark_duration`, `min_request_timestamp`,
  `max_response_timestamp`, `requested_osl`, and the other `filtered_tags` are
  hidden from the main table on both sides.
- **Default log level** is INFO on both sides, and the log file is
  `<artifact_dir>/logs/aiperf.log` on both.
- **Exit codes** for success (0), all-requests-failed (1 — P0.6 fixed at
  `rust/runtime/src/engine/coordinator.rs:357-379`; Python reaches the same 1
  through `src/aiperf/common/bootstrap.py:320-334` raising `SystemExit(1)` on a
  non-empty `_exit_errors`, propagated at
  `src/aiperf/cli_runner/_single_run.py:111-112,132`), and startup/config error
  (1 — `rust/cli/src/main.rs:49-53` vs `src/aiperf/cli_utils.py:31,58`).
- **Console export width**: same env var, same 140 default.

## Withdrawn after baseline correction

None. Every finding's Python half was located in `bc359bf8fd`, including the two
that the rebase put most at risk: the "cancelled early" console notice
(`src/aiperf/controller/system_controller.py:1185` and the duration tag at
`:1230`) and the second-Ctrl-C force-quit path with its on-screen advertisement
(`:844-853`, `:872`, `:882-903`). `controller/system_controller.py` is
byte-identical between the branch and baseline, so findings 1, 2, 3, 4, and 7
required no re-derivation beyond confirming that fact.

## Unverified / needs runtime check

- **Whether a pathological Rust drain eventually self-terminates.** Finding 2
  establishes that signals after the first are discarded, but I did not trace
  whether phase grace/force escalation (`rust/runtime/src/phase_runtime.rs`)
  guarantees a bounded exit for a stream that never closes. Needs a runtime test:
  point a run at a mock server that holds SSE connections open indefinitely,
  Ctrl-C twice, and measure time-to-exit.
- **Whether the interrupt exit code is 0 in practice.** Code reading says a
  drained cancellation yields `success: true` → exit 0
  (`coordinator.rs:381-394`, `profile.rs:513-521`), matching Python, where a
  cancelled run leaves `_exit_errors` empty so `_exit_if_service_failed` does not
  raise (`src/aiperf/common/bootstrap.py:320-334`) and the runner exits with the
  propagated zero (`src/aiperf/cli_runner/_single_run.py:111-112,132`). But whether
  cancelled in-flight requests land in the error bucket and trip
  `all_requests_failed` for a very early Ctrl-C is not determinable from the
  static path alone. Needs: Ctrl-C during warmup and check `$?`.
- **Whether Rust logs GPU-telemetry endpoint reachability anywhere.** Python
  prints "N/M endpoint(s) reachable" plus an unreachable list to the console
  (`system_controller.py:560-568`). I confirmed no console counterpart but did not
  exhaustively check the Rust telemetry module for an equivalent `info!`. Needs a
  run against an unreachable DCGM endpoint with stderr captured.
- **`profile_export_console.txt` byte parity.** Both sides write the artifact at
  the same fixed width, but I compared render logic rather than diffing two real
  files. Needs one identical mock-server run per engine and a diff.
