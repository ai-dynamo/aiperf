# Commit 062 — `d8d49e8c2a`

## Upstream intent

Upstream makes CLI adaptive-search dimensions shape-aware before its Python configuration model is built. Direct profiling dimensions named `rate`, `rate_ramp`, `smoothness`, and `users` select or validate compatible rate, gamma, and user-centric phase shapes; a searchable scalar can seed the base phase from its lower bound. It rejects unsearchable `rate_series`, incompatible shape combinations, invalid scalar bounds, and non-integer user dimensions. The Optuna planner also rewrites only phase-shape validation failures into an actionable error while preserving ordinary validation failures.

## Source inventory

The exact upstream commit changes three Python implementation files and adds the following test coverage:

- `src/aiperf/config/flags/_converter_profiling.py`: lightweight CLI dimension parsing, shape selection, seeding, and early validation.
- `src/aiperf/orchestrator/search_planner/optuna_planner.py`: phase-scoped `ValidationError` reframing at Optuna trial construction.
- `tests/unit/config/test_converter_profiling_phase_routes.py`: the phase selection, seed, conflict, malformed-dimension, and bounds matrix.
- `tests/unit/orchestrator/search_planner/test_optuna_planner.py`: shape-error reframing and preservation of ordinary/root validation errors.

The accompanying upstream documentation describes the same `--search-space` adaptive-search CLI contract. It is not a generic static sweep contract.

## Native comparison and disposition

The native CLI declares `ProfileFlags::search_space`, but there is no native consumer of that field: `rust/cli/src/profile.rs` drives only named recipes and fixed grid/zip sweeps, while `rust/cli/src/search.rs` has a closed `AxisKind` with only concurrency, ISL, and OSL overrides. `rust/cli/src/bayes.rs` is the recipe-specific BO loop, not an arbitrary search-space parser or typed ask/tell planner. `rust/cli/src/load.rs` consequently has no `SearchSpaceDimension`, dotted-path resolver, or phase-overlay boundary to which upstream inference or error framing could apply.

Native phase lowering does already own explicit `--request-rate`, `--arrival-pattern`, `--arrival-smoothness`, `--user-centric-rate`, and `--num-users` semantics. Those flags do not make a parsed `--search-space` dimension exist, and attempting to infer a phase from an otherwise inert raw string would create a benchmark whose requested search never runs. That would be materially less correct than the current native feature boundary.

The upstream defect therefore cannot occur in the native product: no native Optuna trial overlays `rate`, `users`, `smoothness`, `rate_ramp`, or `rate_series` onto a base phase. The applicable unit and planner tests require the absent parser and generic planner, so no Rust test is ported solely to simulate them. This is a true not-applicable closure, not a claim that native implements upstream adaptive search.

## Future port boundary

When native generic `--search-space` support is introduced, it must deliver the complete contract atomically: strict repeatable dimension grammar and aliases; typed dimension kinds and finite/positive bounds; direct profiling field classification; phase selection and base-field seeding before config validation; `rate_series` refusal; conflicts with explicit user/gamma modes; and phase-only error framing at arbitrary trial overlay. The upstream matrix is the source test inventory for that feature, supplemented by native end-to-end ask/tell execution.

## Graham review outcome

The closure-only diff has no Rust production change. Graham review therefore has no hot-path, async, allocation, tracing, synchronization, or error-path implementation to approve or reject. The review receipt at `.superpowers/sdd/2026-08-26-native-search-space-shape-inference/graham-review.md` records the examined scope and no findings.
