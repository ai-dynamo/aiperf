# SDD ledger — plan: docs/superpowers/plans/2026-08-25-native-verbatim-system-prompts.md

## Controller preflight

Design base: `1d1978c22e00786ccf8739a599fd5b70d0d1b191`; implementation
base: target-only exact merge `9eeeac98f944b5fdc425b3b2c1f8f65231447ac8`.

| Task | Native seam | Status |
|---|---|---|
| 1 | CLI/YAML source selection and one-time secure acquisition | completed (`e50421f0b4`) |
| 2 | Common composition, BLAKE3 identity, validation, synthetic accounting | completed (`8da4c4f3f4`) |
| 3 | OpenAI and Anthropic production request construction | completed (`15b384ac06`) |
| 4 | Real-binary E2E and current-truth docs | completed (`f16aff6909`, `8180bf629f`) |
| 5 | Upstream mapping, full verification, independent Graham review | completed (`337430c252`, `b4f2376319`, closure commit) |

## Ancestry and scope receipts

- `9eeeac98f944b5fdc425b3b2c1f8f65231447ac8` is a real two-parent
  merge, never a cherry-pick.
- First parent:
  `17c69ce049be7aa5c7532a72efb35c617d2a858f`.
- Second parent: exact upstream
  `88242293b552db96b90b2e3999bbfa93488c994f`.
- Upstream changes 20 paths. The target merge changes the same 19 authored
  paths; the generated Python schema is unchanged because the target generator
  already emits the current schema.
- The cumulative range adds only the native config/composition/endpoint/E2E
  implementation, design/plan/review artifacts, and synchronized current-truth
  documentation required for #49.

## Test mapping and verification

- The tracker accounts for all 40 upstream-added test functions. Thirty-nine
  map to native behavioral tests; `test_injected_without_a_tokenizer` is
  inapplicable because native text segments require token-bearing identity.
- CLI library: 259 passed, 0 failed.
- OpenAI endpoint integration: 21 passed, 0 failed.
- Anthropic endpoint integration: 14 passed, 0 failed.
- Real native binary plus in-process Rust mock server: 14 passed, 0 failed.
- Focused cache-isolation and equal/changed BLAKE3 identity tests passed.
- Full default runtime: 1,811 passed, 7 ignored, one unchanged failure in
  `metrics_core::report::tests::v2_uses_type_specific_series_and_null_for_non_finite_tail`
  because its golden says `0.0.0` while the package emits `0.12.0`.
- Engine and all-target Clippy stop at unchanged
  `runtime/tests/agentx_online_e2e.rs:116,254`, whose initializers omit the
  inherited `cache_bust_first_user_turn` field. No #49 path has that field or
  diagnostic.
- Full CLI integration stops at unchanged `cli/tests/graph_tools.rs:1300`,
  whose expected built-in inventory omits `tracelab`; CLI library remains
  259/259 green.
- `check-ruff-baselined` reports inherited violations. Its one report in an
  upstream-touched file is the old `_reject_unknown_envelope_keys` body, whose
  line moved after the imported validator; blame remains pre-#49.
- `cargo fmt --check`, range `git diff --check`, `check_docs_current.py`, and
  `check_agent_files_sync.py` passed.

## Independent review

The independent two-pass Graham review covered the complete range
`1d1978c22e00786ccf8739a599fd5b70d0d1b191..b4f2376319cfe5bbac94e9d79dfaf5b6e585242c`.
It found no Critical or Important issues in file acquisition, composition,
identity, endpoint rendering, capability validation, async/error/allocation
behavior, test rigor, comments, or scope.

GRAHAM APPROVED
