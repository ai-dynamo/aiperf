# Native Baseten Columnar Load Progress

## Ancestry and scope

- Isolated worktree: `/mnt/4tb/aiperf-origin-port-039`
- Branch: `port/origin-039-baseten-perf`
- Exact merge: `da917561fbbe375fa08ec88eaa378a00e44ddc23`
- First parent: `3e43143555caac99af69c4f5c4d167a5cc4b2f93`
- Second parent: upstream `1e32a51318af3474dc1672d225fd495471ab45df`
- Design: `docs/specs/2026-08-25-native-baseten-columnar-load.md`
- Plan: `docs/superpowers/plans/2026-08-25-native-baseten-columnar-load.md`

The upstream parent preceding `1e32a51318` was already an ancestor of the
first parent, so the merge imports only the target delta. Tracker #40's
outcome-fidelity fields are outside this port.

## Characterization

The pinned fixture is a deterministic 100,000-row, 177 MiB Parquet trace with
190.8 MB of decoded Arrow data. It contains all replay columns plus three
unused 512-byte strings and unused outcome/model/metadata fields. The release
`dataset_load_bench` adapter measures the real native load -> compose path with
the tokenizer initialized before timing.

Every sample used
`AIPERF_DATASET_BASETEN_SESSION_COLUMN=poor_man_session_id` and produced exactly
100,000 rows, 50,000 conversations, 100,000 turns, and 6,400,000 input tokens.

| Sample | Baseline elapsed | Baseline RSS | Port elapsed | Port RSS |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 3.249376745 s | 562,772 KiB | 2.670634273 s | 420,440 KiB |
| 2 | 3.310308095 s | 564,064 KiB | 2.634085088 s | 421,304 KiB |
| 3 | 3.266392883 s | 562,604 KiB | 2.426971834 s | 422,352 KiB |

Median elapsed fell from 3.266 s to 2.634 s (19.4%); median peak RSS fell from
562,772 KiB to 421,304 KiB (25.1%). This is a measured native performance port,
not a source-only claim.

Parquet decoding is projected and limited to 128-row record batches. Arrow IPC
decoding is projected but `arrow::ipc::reader::FileReader` materializes each
authored record batch; the loader then visits zero-copy slices of at most 128
rows. The IPC decode bound is therefore the largest projected authored batch,
not 128 rows. The specification and Sol plan state this narrower contract.

## Upstream-to-native test mapping

| Upstream surface | Native coverage |
| --- | --- |
| `.parquet`, `.arrow`, and `.ipc` detection | `arrow_ipc_detection_and_composition_match_parquet`; public adapter parity test |
| Required-schema refusal | `can_load_requires_parquet_extension_and_required_columns` |
| Projected decode and 128-row processing slices | `projected_batches_skip_unused_columns_and_validate_late_rows`; the spec records the narrower authored-batch IPC decode bound |
| Ignore unused wide columns | Same projection-observer test; real wide fixture benchmark |
| Validate malformed later required rows | Same test places invalid `input_tokens` at row 130 |
| Configured/preferred/fallback/none/invalid session policy | `session_column_policy_covers_preference_configuration_fallback_and_absence` plus `configured_default_session_column_controls_grouping` |
| No-session sources bypass ratio sampling | `metadata_sampling_is_disabled_without_a_session_column` |
| Deterministic whole-session sampling and max-row ordering | `metadata_sampling_is_deterministic_and_keeps_whole_sessions`; `max_rows_precedes_seeded_session_sampling_for_direct_and_registry_loads` |
| Direct/registry seed parity | `LoadConfig::with_rng_root` and `max_rows_precedes_seeded_session_sampling_for_direct_and_registry_loads` |
| Path, column, and stable row ordinal diagnostics/origins | `projected_batches_skip_unused_columns_and_validate_late_rows`; direct Arrow origin assertions in `arrow_ipc_detection_and_composition_match_parquet` |
| Parquet/Arrow public load -> compose semantics | `baseten_parquet_and_arrow_adapter_samples_match` |
| Baseten `inputs.json` suppression | Existing generic fixed-schedule `phase_filters_dataset` sidecar policy; no duplicate Baseten path added |

## Verification

All Cargo commands used `RUSTC_WRAPPER=/usr/bin/sccache` and
`CARGO_TARGET_DIR=/mnt/4tb/aiperf-origin-port-039-target`.

- `cargo test -p aiperf-runtime --features parquet dataset::loader::baseten --lib`
  - 17 passed, 0 failed.
- `cargo test -p aiperf-runtime --features engine,parquet --test dataset_load_bench baseten`
  - 1 passed, 0 failed.
- `cargo build --release -p aiperf-runtime --example dataset_load_bench`
  - passed.
- `rustfmt --edition 2024 --check` on the two changed Rust loader files and the
  adapter integration test
  - passed.
- `git diff --check`
  - passed.
- `cargo clippy -p aiperf-runtime --features parquet --lib --tests --no-deps`
  - blocked by the unrelated existing
    `agentx_production_transport_e2e.rs:130` missing-field compile error; no
    Baseten diagnostic preceded it. A library-only Clippy result is recorded
    separately after this evidence draft.
- `cargo clippy -p aiperf-runtime --features parquet --lib --no-deps`
  - passed; existing workspace warnings remain, with no Baseten diagnostic.

The workspace-wide format check also reports unrelated existing formatting
drift in `rust/cli/src/yaml.rs` and `rust/runtime/src/endpoints/mod.rs`; scoped
formatting for every changed Rust file is clean.

## Commits

- `554f8a7e16` — design and Sol implementation plan.
- `e7d57c2618` — Arrow IPC feature, source detection, and parity seam.
- `d93e07e5a3` — projected bounded scans, configured grouping, metadata-first
  deterministic sampling, and focused tests.
- `fec6ed8e96` — public adapter Parquet/Arrow integration parity and exact
  environment literal handling.
- `90ffc47336` — registry detection probes columnar schema without JSON-reading
  binary files.
- `8d8a17b474` — checked typed-array decode, pre-sampling row limit, explicit
  direct-load RNG, stable origins/errors, and complete session-policy tests.
- `59ae696f21` — narrowed and documented Arrow IPC authored-batch decode bound.

The first independent Graham review returned Important findings. Every finding
has a RED-to-GREEN repair in the commits above. An independent fresh review then
made two focused passes over `da917561fb..8e6a45372b` and reported
`GRAHAM APPROVED` with no findings and no unresolved Critical or Important
finding. The full review scope and verdict are recorded in `graham-review.md`.
