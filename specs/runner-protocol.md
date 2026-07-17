<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Runner protocol

## Purpose

Define the boundary between the Config-v2 front end and one execution. The
`aiperf` binary re-executes itself over a protocol-v2 stdio seam: the front end
resolves a run into a `BenchmarkRun` and spawns a fresh `aiperf --execute` (or
`--validate`) child per operation. One vocabulary crosses the boundary — the
canonical `BenchmarkRun` — with no parallel wire dialect and no distribution
pin.

## Built

### Process boundary

Config v2 owns the human-facing CLI, outer loops (sweeps, trials, adaptive
search), and presentation; it fully resolves a run before spawn. The child
process is the sole Rust composition root for one run's hot path and is
protocol-v2 only. One fresh child per operation emits exactly one JSONL response
line on stdout.

### Envelope and operations

The versioned envelope is `EnvelopeV2` (`deny_unknown_fields`):

```json
{ "protocol_version": 2, "operation": "validate" | "execute", "run": { /* BenchmarkRun */ } }
```

`OperationV2` is `validate` (side-effect-free structural and static semantic
validation) or `execute` (revalidate, prepare, execute, commit the report). In
the current stdio path the stdin payload is the bare `BenchmarkRunWireV2` and the
operation is carried by the child's `--execute`/`--validate` mode; the child
reconstructs `EnvelopeV2` in-process with the fixed protocol version before
handing it to the coordinator. `protocol_version` must equal 2; unknown envelope
keys fail closed.

### `run` is exact `BenchmarkRun`

`run` uses the same field names and nesting as the front end's `BenchmarkRun`
(`benchmark_id`, `sweep_id`, `label`, `trial`, `artifact_dir`, `random_seed`,
`variation`, `cli_command`, `variables`), plus:

- `cfg`: the `BenchmarkConfig` dump. Transport is the inline discriminated object
  Config authors (`type` plus variant fields), not a `{type, config}` re-nest.
- `resolved`: the resolution product of `BenchmarkConfig` — concrete facts derived
  once (absolutized paths, tokenizer aliases, telemetry mode, sampling
  strategies, GPU custom-metric bindings). Rust treats `resolved` as authoritative
  and does not re-derive those facts.

There is no `identity` wrapper, `artifact_target`, `resources` bag, `workload`
framing, `backend` dialect, or `replay_mode` field. Presentation-only sections
that must not reach the runner are stripped from the dump.

### Path selection

The execution path is chosen from `run.cfg`, not a workload id:

| Signal in `run.cfg` | Execution path |
|---|---|
| Dataset type `dag_jsonl` / `weka_trace` / `dynamo_trace` | graph |
| Otherwise | scheduled |
| `transport.type: http` | wall-clock HTTP |
| `transport.type: grpc` | wall-clock gRPC |
| `transport.type: dynosim_offline` | virtual-clock in-process co-simulation |
| `transport.type: dynosim_online` | wall-clock in-process co-simulation |

`BenchmarkConfigWireV2` deliberately deserializes only the runner-relevant
Config subset without `deny_unknown_fields`. An authored `cfg.workload` key is
therefore ignored, and the runner derives `scheduled` versus `graph` from the
dataset type. The CLI's typed `cfg.accuracy` value is serialized in
`BenchmarkRun`, but the runner subset likewise ignores it. These keys are not
actively rejected; static accuracy remains off the product execution path
because no consumed runner field selects it.

### Discovery

`aiperf_cli::execute_mode::capabilities_catalog()` is the in-process discovery
API; there is no `--capabilities` argv mode. The function composes the stock
`Application` for the linked distribution and returns its typed `Catalog` in
`category → id → { description?, metadata? }` shape from the same registries
used for validation and execution. Feature gates appear as presence/absence of
ids (for example `dynosim_offline` only in dynosim builds). Callers serialize or
query the returned catalog by component id, not by transport/workload pair.

### Responses and exit codes

- `validate` → one `RunValidationV2` line: `protocol_version`, `event`,
  `benchmark_id`, `success`, `completeness`, `deferred_checks`, and typed `errors`
  with paths under `run.cfg.*` / `run.resolved.*`.
- `execute` → one `RunTerminalV2` line: `protocol_version`, `event`,
  `benchmark_id`, `success`, `report_path` under `run.artifact_dir`, a failed
  `stage` (`preparation` / `execution` / `reporting`) when useful, and typed
  `errors`.

Exit codes: 0 success, 1 validation/execution failure, 2 protocol failure.
Report and terminal provenance use Config names (`transport`, dataset-derived
path), never registry ids.

## Source anchors

- `rust/runtime/src/engine/protocol_v2.rs` (`EnvelopeV2`, `OperationV2`,
  `BenchmarkRunWireV2`, `RunValidationV2`, `RunTerminalV2`).
- `rust/cli/src/{execute.rs,execute_mode.rs,exec_bin.rs}` (re-exec, mode
  selection, stdin decode, and `capabilities_catalog`).
- Runner stdio tests under `rust/cli/tests/*_stdio.rs`.
