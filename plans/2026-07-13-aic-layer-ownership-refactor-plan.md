<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AIC layer ownership refactor plan

Move the generic AIC (aiconfigurator) timing-model logic out of AIPerf and into
its correct owner, leaving AIPerf holding only the consumer-side install glue.

## Problem statement

`rust/aiperf/src/aic_runtime.rs` (~350 lines, feature-gated behind
`dynamo-aic-forward-pass`) is a wholesale port of dynamo's
`lib/bindings/python/rust/llm/aic_callback.rs` plus the KV-block scaling from
`replay.rs:1668-1715`. The module doc says the intent is "keeping the
embedded-Python AIC bridge in the consumer rather than in the pure-Rust
simulator."

Of those ~350 lines, only the tail of `configure_aic_runtime` — the
`MockEngineArgs → args.perf_model` install — is genuinely AIPerf-specific. The
rest is generic AIC-model logic:

- `normalize_quant_modes` / `normalize_quant_mode` — resolves modes against
  `aiconfigurator.sdk.common`.
- `build_engine` — `build_aic_engine` + a process-wide `OnceLock` engine cache.
- `estimate_engine_num_gpu_blocks` — calls `aiconfigurator.sdk.memory`.
- `populate_missing_offload_kv_bytes_per_token` — computes `kv_bytes_per_token`
  from a HuggingFace `AutoConfig`.
- `NativeAicCallback` — implements the sibling-owned `AicCallback` and
  `PrefillLoadEstimator` traits over a compiled `AicEngine`.

### Why the current placement is wrong

- **All dependencies and the trait are already sibling-owned.** `AicCallback`
  and `PerfModel` are defined in `dynamo_mocker::common::perf_model`;
  `PrefillLoadEstimator` in `dynamo_kv_router`; `AicEngine` / `build_aic_engine`
  in `aiconfigurator-core`. AIPerf's type only *implements* traits it does not
  own.
- **It is a fork, and it will drift.** The code cites upstream line numbers it
  was copied from. The same logic now exists twice (dynamo's `aic_callback.rs`
  and AIPerf's `aic_runtime.rs`) and both will be maintained in parallel.
- **The original "foreign vendored fork" justification does not hold.**
  `aiconfigurator` (`github.com/ai-dynamo/aiconfigurator`), the dynamo `mocker`
  (checked out at `../../../../../dynamo-aiperf-native`), and `aiperf` are all
  the same organization (ai-dynamo / NVIDIA). There is no external-vendor rebase
  burden to avoid; the org governs all three trees.
- **The pyo3/embed-python cost is already paid on the sibling side.**
  `aiconfigurator-core` is pulled with `features = ["embed-python"]` and the
  sibling already ships a pyo3 `aic_callback.rs`. Keeping a second pyo3-touching
  copy in AIPerf duplicates the cost rather than avoiding it.

### Correct ownership

The generic logic calls back into aiconfigurator's own Python SDK
(`aiconfigurator.sdk.common`, `aiconfigurator.sdk.memory`). That is
`aiconfigurator`'s domain, not AIPerf's and not the mocker's. Target ownership:

- **`aiconfigurator-core`** owns quant normalization, engine build/cache, KV
  block estimation, and `kv_bytes_per_token`, and exposes a builder that returns
  a value implementing `AicCallback` + `PrefillLoadEstimator`. It already owns
  `AicEngine` and the `embed-python` feature.
- **dynamo `mocker`** stays pure-Rust and continues to own the `AicCallback` /
  `PerfModel` traits (unchanged).
- **AIPerf** keeps only `configure_aic_runtime`'s real body: map
  `MockEngineArgs` → call the aiconfigurator builder → install the result on
  `args.perf_model`. Estimated ~10–30 lines.
- **dynamo's own `aic_callback.rs`** collapses onto the same `aiconfigurator`
  surface, deleting the duplication on both sides — not only AIPerf's.

## Non-goals

- No change to the `dynosim` / `dynamo-full` product-reachability story or the
  Config-v2 `transport.type: dynosim_offline | dynosim_online` surface.
- No change to the `{clock, transport}` seams, the mocker's pure-Rust guarantee,
  or the `AicCallback` / `PerfModel` trait definitions.
- Not upstreaming anything to a truly external project — all three targets are
  same-org.

## Preconditions / open questions (resolve before Phase 1)

1. **Cross-repo change coordination.** This touches three repos
   (`aiconfigurator`, `dynamo-aiperf-native`, `aiperf`). Confirm the merge order
   and who owns the `aiconfigurator-core` API addition. This plan assumes
   aiconfigurator lands first, then the mocker/aiperf consumers bump to it.
2. **Builder input shape.** `aiconfigurator-core` must not depend on
   `dynamo_mocker::MockEngineArgs` (that would invert the dependency direction).
   The builder takes a plain params struct owned by `aiconfigurator-core`;
   AIPerf and dynamo each project their own args into it.
3. **Engine cache location.** The process-wide `OnceLock<Mutex<HashMap>>` cache
   moves into `aiconfigurator-core`. Confirm a shared cache across both consumers
   in one process is acceptable (it is keyed by the full engine signature, so it
   is correct; verify no consumer wants isolation).
4. **Release cadence.** Inter-repo version-bump friction is the one remaining
   real cost. Confirm the teams accept it versus the current duplication.

## Phase 1 — extract the generic AIC surface into `aiconfigurator-core`

Author, in `aiconfigurator` (same org):

- A params struct (`AicEngineParams` or similar) capturing: backend,
  backend_version, system, model_path, tp/attention-dp/moe sizes, the five quant
  modes, nextn, accept rates, and the memory-fraction inputs.
- A `build_callback(params) -> Result<AicCallbackHandle>` returning a value that
  implements `AicCallback` + `PrefillLoadEstimator`, backed by a compiled
  `AicEngine` and the moved `OnceLock` cache.
- `estimate_num_gpu_blocks(params) -> Result<usize>` and
  `kv_bytes_per_token(model_path, kv_dtype) -> Result<Option<usize>>` as public
  helpers (the pyo3 calls into `aiconfigurator.sdk.*` and `transformers` move
  here verbatim, preserving the canonical-`None` behavior).
- Keep the quant-mode normalization and default backend-version resolution here.

Evidence: `aiconfigurator-core` builds with `--features embed-python`; unit
coverage for quant normalization and default-version resolution ported with the
code.

## Phase 2 — collapse AIPerf onto the new surface

In `rust/aiperf/src/aic_runtime.rs`:

- Delete `normalize_quant_mode(s)`, `resolve_backend_version`, `build_engine`,
  `estimate_engine_num_gpu_blocks`, `populate_missing_offload_kv_bytes_per_token`,
  and `NativeAicCallback`.
- Rewrite `configure_aic_runtime` to: (a) call
  `aiconfigurator_core::kv_bytes_per_token` for the offload sizing branch, (b)
  early-return `Ok(None)` when `aic_backend` is absent, (c) call
  `estimate_num_gpu_blocks` when `num_gpu_blocks` is not explicit, (d) call
  `build_callback`, and (e) install `PerfModel::from_aic_callback_with_attention_dp`
  on `args.perf_model`, returning the estimator handle.
- Update the `dynamo-aic-forward-pass` feature: it still needs
  `aiconfigurator-core` and `dynamo-kv-router`; it should no longer need `pyo3`
  or `parking_lot` directly if all pyo3 use moved to `aiconfigurator-core`.
  Verify and drop the now-unused optional deps from `rust/aiperf/Cargo.toml`.

Evidence: `cargo build -p aiperf --features dynamo-full` green; the offline/online
replay stdio tests that exercise AIC still pass byte-for-byte against the Dynamo
parity summary.

## Phase 3 — collapse dynamo's own copy (optional but the point of the exercise)

In `dynamo-aiperf-native`, rewrite
`lib/bindings/python/rust/llm/aic_callback.rs` to consume the same
`aiconfigurator-core` surface, deleting its duplicate engine build / quant /
KV-estimation logic. This is what turns the change from "AIPerf stops forking"
into "the org maintains one AIC bridge."

Evidence: dynamo's own AIC-backed replay tests pass against the shared surface.

## Verification gates

- `cargo build -p aiperf --features dynamo-full` and
  `cargo test -p aiperf --features dynosim --lib`.
- The runner offline/online replay parity: AIC-backed runs still serialize to
  identical bytes against the Dynamo flat summary (the existing offline return
  path already enforces this — it must remain green, unmodified).
- No pyo3 symbol remains reachable from `rust/aiperf` on the AIC path (grep
  `Python::with_gil` under `rust/aiperf/src/` returns nothing after Phase 2).
- Line-count check: `aic_runtime.rs` shrinks from ~350 to under ~50 lines.

## Risk / rollback

- **Primary risk:** parity drift during extraction (quant-mode name resolution,
  the `int4 → int4_wo` rewrite, KV-block attention-DP scaling, the canonical
  `None` preservation in `kv_bytes_per_token`). Mitigation: port the pyo3 bodies
  verbatim and pin them with unit tests before deleting the AIPerf copies.
- **Rollback:** the change is additive in `aiconfigurator-core`; if consumers
  break, revert Phases 2/3 and keep the AIPerf copy until the shared surface is
  fixed. Phase 1 can land independently.

## Docs to update on completion

- `rust/aiperf/CLAUDE.md` (and the three synced agent files) — the
  "Canonical vs aspirational" AIC line now points at `aiconfigurator-core` as the
  owner; sync with `tools/sync_agent_files.py`.
- `docs/module-organization.md` if `aic_runtime` is reduced to install glue.
