# Native search-space shape inference

## Purpose

Record the native boundary for upstream commit `d8d49e8c2a`: Python adaptive search infers a profiling phase shape from parsed `--search-space` dimensions, but the native CLI does not offer generic `--search-space` execution.

## Decision

Do not add Rust shape inference, seeds, or a compatibility test fixture. A native implementation would need a typed, repeatable search-space grammar and an arbitrary dimension planner. Without those consumers, modifying phase lowering from `ProfileFlags::search_space` would alter a normal run while silently failing to run the requested adaptive search.

Named native recipes remain unchanged. Their closed axes are explicit, validated, and do not accept arbitrary profiling paths. Explicit native phase flags remain governed by their existing lowering and validation rules.

## Requirements for a future generic planner

A future implementation must parse dimensions before phase construction and apply the complete upstream behavior in one feature:

- resolve only direct `phases.profiling.<field>` scalar aliases for inference;
- infer rate, gamma, or user-centric shapes from the supported fields, with deterministic interaction with explicit arrival/user flags;
- seed required `rate` and `users` base values from valid dimension bounds;
- reject `rate_series`, shape conflicts, non-finite/non-positive bounds, and non-`:int` user dimensions before launching a trial; and
- preserve ordinary trial validation failures while reframing only profiling phase shape mismatches.

The feature must include native unit coverage for all classification and validation paths plus an end-to-end planner run. Until then, Python owns the upstream behavior and this commit is not applicable to native Rust.
