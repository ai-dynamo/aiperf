<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# BenchmarkRun wire and runner catalog redesign

**Status:** built (performance wire + catalog; agentic/static_accuracy/evaluation modules remain linked but off the product wire; some dynosim process tests ignored pending fixture migration)  
**Date:** 2026-07-13  
**Supersedes (wire + discovery portions):**  
`2026-07-11-python-orchestrator-rust-single-run-design.md` (authored projection dialect, `expected_distribution_id`, `transport`/`workload` request framing, pair preflight),  
`2026-07-11-aiperf-runner-only-execution-surface-design.md` (`backend`/`workload` pair matrix, `supported_pairs` capabilities shape, distribution pinning).  
Those specs remain historical; this document is authoritative where they conflict on the Python↔runner boundary.

## Problem

The protocol-v2 boundary grew a second dialect beside Config:

- Python projects `run.transport` with IDs like `http` / `grpc` / `dynosim_offline`; Rust still decodes `run.backend` with `online_http` / `online_grpc` / `dynosim` (+ `replay_mode`).
- `rust_wire.py` hand-lowers Config into `{type, config}` frames, renames fields, and stuffs `dataset` / `phases` / `tokenizer` into `workload.config`.
- Adding a transport or endpoint requires coordinated Python projection, Rust DTOs, capabilities pairs, and tests.
- `--capabilities` advertises a runner-invented pair matrix (`supported_pairs`, `backends`, workload IDs) unlike the catalog Python already understands from `plugins.yaml`.
- `expected_distribution_id` / binary BLAKE3 pinning adds envelope complexity without belonging in the Config contract.

## Goals

1. **Least maintenance** — one vocabulary; no parallel wire schema; no hand re-nesting dialect.
2. **Python owns the contract** — `BenchmarkRun` / `BenchmarkConfig` are canonical; Rust strict-decodes them.
3. **Hard cut** — one shape only; no dual decode; no protocol version bump (remain `protocol_version: 2`).
4. **Discovery like `plugins.yaml`** — category → id → metadata, emitted as JSON from the linked binary.
5. **Performance-only product path** for this cut — scheduled and graph execution from Config shape; agentic / static accuracy / evaluation / telemetry-watch leave the product wire as Config sheds them.

## Non-goals

- JSON Schema / codegen dual binding.
- Dynamic plugin loading of Python classes into the Rust runner.
- Keeping `workload` as a registry selection axis.
- Keeping distribution-id pinning on the wire.
- Redesigning native-v2 report metric formulas (only provenance field names that still say `backend` / `workload`).

## Decision summary

| Topic | Decision |
|---|---|
| Request body | Exact `BenchmarkRun` JSON (including `resolved`) |
| Envelope | `{ protocol_version: 2, operation, run }` only |
| Projection | Dump `BenchmarkRun`; no `resources` / `{type,config}` transport re-nest / workload stuffing |
| `resolved` | First-class resolution product of `BenchmarkConfig`; runner must not re-derive the same facts |
| Path selection | From `run.cfg` (dataset kind → graph vs scheduled; `transport.type` → clock/transport) — no `workload.type` |
| Discovery | Linked-inventory JSON catalog in `plugins.yaml` category shape; YAML only if hand-authored |
| Distribution pin | Removed entirely |
| Product modes | Performance only (`http` / `grpc` / `dynosim_offline` / `dynosim_online` × scheduled/graph) |

---

## 1. Process envelope

stdin JSON for validate/execute:

```json
{
  "protocol_version": 2,
  "operation": "validate" | "execute",
  "run": { /* BenchmarkRun */ }
}
```

- No `expected_distribution_id`.
- Unknown envelope keys fail closed (`deny_unknown_fields` on the Rust envelope).
- Still one fresh child per operation; one JSONL response line on stdout.

Discovery (replaces today’s pair-matrix `--capabilities` payload):

- Selected binary prints **one JSON object** (stdout) describing its linked catalog.
- Hand-authored manifests may remain YAML elsewhere; **machine-emitted catalogs are JSON**.

## 2. `run` is exact `BenchmarkRun`

`run` uses the same field names and nesting as `aiperf.config.resolution.plan.BenchmarkRun`:

- `benchmark_id`, `sweep_id`, `label`, `trial`, `artifact_dir`, `random_seed`, `variation`, `cli_command`, `variables`, …
- `cfg`: `BenchmarkConfig` dump
- `resolved`: resolution product (see §3)

**Deleted from the wire dialect (not renamed):**

- `identity` wrapper
- `artifact_target` (use `artifact_dir`)
- `resources` bag
- `workload` framing and stuffing shared schedule inputs into `workload.config`
- `backend` / `online_http` / `online_grpc` / single `dynosim` + `replay_mode`

**Transport** stays as Config already authors it: inline discriminated object (`type` + variant fields), not `{type, config}`.

**Path selection (no workload ID):**

| Signal in `run.cfg` | Execution path |
|---|---|
| Dataset type `dag_jsonl` / `weka_trace` / `dynamo_trace` | graph |
| Otherwise | scheduled |
| `transport.type: http` | wall-clock HTTP |
| `transport.type: grpc` | wall-clock gRPC |
| `transport.type: dynosim_offline` | virtual-clock in-process Dynamo |
| `transport.type: dynosim_online` | wall-clock in-process Dynamo |

As Config drops `workload`, agentic, static `accuracy`, evaluation, and archive-watch sections, the wire shrinks with them because the wire *is* BenchmarkRun.

Python-only presentation/orchestration blocks that must never reach the runner (e.g. `mlflow`, `wandb`, `otel`, `logging`, leftover ZMQ service config) are excluded from the dump **or** removed from `BenchmarkConfig` over time so exclusion is not a permanent hand list. Prefer removing them from the model that crosses the boundary.

## 3. `resolved` is part of the contract

`BenchmarkRun.resolved` is **not** private Python cache. It is the **side-effect resolution of `BenchmarkConfig`**: concrete facts derived once (absolutized paths, tokenizer aliases, telemetry mode, sampling strategies, GPU custom-metric bindings, etc.) so the runner does not re-derive them.

- Authored policy lives in `cfg`.
- Derived bindings live in `resolved`.
- Rust treats `resolved` as authoritative for those facts.
- Removing `resolved` from the wire would force duplicate resolution in Rust — rejected.

Validate may still refuse unsafe side effects (network fetch, artifact creation) while accepting an already-populated `resolved` object whose facts were computed by the orchestrator before spawn.

## 4. Runner catalog (discovery)

### Shape

Category → type id → `{ description?, metadata? }`, matching the **UX of `plugins.yaml`** for categories the runner owns. Example:

```json
{
  "schema_version": "1.0",
  "endpoint": {
    "chat": {
      "description": "OpenAI Chat Completions",
      "metadata": {
        "endpoint_path": "/v1/chat/completions",
        "supports_streaming": true,
        "produces_tokens": true,
        "tokenizes_input": true
      }
    }
  },
  "transport": {
    "http": {
      "metadata": {
        "transport_type": "http",
        "url_schemes": ["http", "https"]
      }
    },
    "grpc": {
      "metadata": {
        "transport_type": "grpc",
        "url_schemes": ["grpc", "grpcs"]
      }
    }
  },
  "custom_dataset_loader": {},
  "public_dataset_loader": {},
  "dataset_sampler": {}
}
```

Include only categories the **linked runner** implements (endpoint, transport, dataset loaders/samplers, and any other Config-facing registries). Do **not** advertise Python-only categories (services, ZMQ, plots, UI, search recipes).

### Compiled binary mechanics

- No dynamic load of Python `class:` paths.
- Catalog is **frozen at link time** from the same registries used for validation/execution.
- Feature gates appear as **presence/absence** of ids (e.g. `dynosim_offline` only in dynosim builds).
- Optional diagnostic factory id may appear for humans; it is not a loadable Python class path.
- Delivery: discovery subcommand/flag on the selected binary prints this JSON object.

### Python preflight

Preflight checks Config identifiers against this catalog the way today’s plugin registry checks `plugins.yaml` (endpoint type exists, transport type exists, dataset format exists, metadata constraints). **Not** `[transport, workload]` pair tuples.

### Deleted discovery fields

`event: runner_capabilities`, `capabilities_schema_version`, `distribution_id`, `backends`, `workloads`, `supported_pairs`, `statically_compatible_pairs`, evaluation-provider matrices tied to the old pair surface.

## 5. Responses and provenance

- Validate → one `run_validation` JSON line: `protocol_version`, `benchmark_id`, `success`, errors with paths under `run.cfg.*` / `run.resolved.*`.
- Execute → one `run_terminal` JSON line: `protocol_version`, `benchmark_id`, `success`, `report_path` under `run.artifact_dir`.
- No `distribution_id` on responses.
- Report / terminal provenance uses Config names (`transport`, dataset-derived path) — never `backend` / `workload` registry ids.

Exit codes unchanged in spirit: 0 success, 1 validation/execution failure, 2 protocol failure.

## 6. Error handling

| Failure | Behavior |
|---|---|
| Malformed envelope / unknown envelope keys | protocol failure (exit 2) |
| `run` fails BenchmarkRun-shaped decode or Config/`resolved` rules | validation failure (exit 1), field paths |
| Catalog discovery unusable | Python fails before spawn; no silent fallback |
| Unknown endpoint/transport/dataset id vs catalog | preflight error in Python or validation error in Rust |
| Execute-time failure | terminal failure with stage when useful (`preparation` / `execution` / `reporting`) |

## 7. Hard-cut migration

Single change set (no dual decode):

**Python**

- Replace `rust_wire` dialect projection with `BenchmarkRun` JSON dump (+ only transforms that cannot live in `resolved` / Config).
- Drop distribution pin verify/stamp path.
- Consume JSON runner catalog for preflight.
- Remove product paths for agentic / static accuracy / evaluation / telemetry-watch as Config sheds them.

**Rust**

- Strict-decode BenchmarkRun-shaped `run`.
- Emit JSON catalog from linked registries.
- Delete `backend` DTOs, pair-matrix capabilities, workload wire stuffing, `replay_mode` clock split (clock is on transport id).
- Align provenance field names.

**Docs**

- This spec is authoritative for the boundary.
- Append addenda on the two superseded specs (done alongside this file).
- On implementation: update agent files, `llms.txt`, tests, and golden fixtures in the same change.

## 8. Testing

- Golden JSON requests: minimal scheduled HTTP BenchmarkRun; graph dataset path; dynosim transport.
- Golden catalog JSON: base image vs dynosim feature image.
- Python preflight tests assert catalog id lookup (plugins.yaml-style), not `supported_pairs`.
- Delete dual-vocabulary fixtures (`backend`, `online_http`, workload-stuffed configs).
- Runner stdio tests speak BenchmarkRun + catalog JSON only.

## 9. Implementation notes (non-normative)

- Prefer shrinking `BenchmarkConfig` so the dumpable model *is* the runner-facing model, rather than maintaining a permanent exclude list.
- Endpoint metadata in the catalog should converge on the same fields Python already reads from `plugins.yaml` / `EndpointMetadata` where those fields still matter for preflight.
- Keep `protocol_version: 2`; the hard cut replaces the body and discovery documents in place.

## Open follow-ups (explicitly out of this decision)

- Exact category list beyond endpoint/transport/dataset\* (add when a Config field needs preflight).
- Whether discovery flag is renamed from `--capabilities` (cosmetic; payload is the contract).
- Timing of deleting leftover Python plugin categories that the runner never owned.

## Addendum — 2026-07-13 (implementation)

Built the hard cut on the product performance path:

- Python serializes a BenchmarkRun-shaped `run` in a thin `{protocol_version, operation, run}` envelope with no `expected_distribution_id`. Presentation sections (`logging` / `wandb` / `otel` / `mlflow`) and null `workload` are stripped; nested factory inputs (`phases`, `datasets`, `tokenizer`, endpoint readiness, artifacts/sidecars) are still lowered into the shapes linked runner factories already decode until those factories accept raw Config dumps directly.
- Python preflight consumes the plugins.yaml-shaped JSON catalog from `--capabilities` (no `supported_pairs` / distribution pin).
- Rust strict-decodes BenchmarkRun, selects scheduled vs graph from dataset format, binds wire transport ids `http` / `grpc` / `dynosim_offline` / `dynosim_online`, and rejects authored `cfg.workload` / `cfg.accuracy` on the product wire.
- Agentic / static-accuracy / evaluation modules remain linked but off the product projection path; some dynosim and dead-mode process fixtures are ignored pending separate fixture migration.

This addendum does not change the target contracts in §§1–5; it records the temporary nested lowering and ignored fixtures that remain until factories and Config converge further.
