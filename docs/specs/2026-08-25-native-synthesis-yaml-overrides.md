# Native synthesis CLI-over-YAML precedence

## Goal

Make native `aiperf profile --config ... --synthesis-*` apply the explicitly
authored synthesis flags over a YAML-authored `dataset.synthesis` block, while
preserving every unoverridden YAML value exactly.

## Problem

The CLI-only path already lowers `--synthesis-*` through
`rust/cli/src/load.rs::build_synthesis`, but the YAML path resolves through
`rust/cli/src/yaml.rs::resolve_expanded_inputs` and never overlays those flags
onto `Inputs.synthesis`. As a result, `--config foo.yaml --synthesis-max-osl N`
keeps the YAML `maxOsl` value instead of honoring the CLI override.

This is a real behavioral gap relative to upstream commit `6480e5467f`. The
exact upstream diff adds only Python unit coverage; there is no upstream
integration or E2E test to port.

## Contract

When a YAML config produces a recorded-trace `Inputs.synthesis` object and the
user also passes one or more `--synthesis-*` flags, the native loader must:

1. override only the fields explicitly authored by the CLI;
2. preserve every unoverridden YAML synthesis value, including non-identity
   values such as `speedupRatio: 2.0`;
3. continue to stamp the required identity defaults when the CLI is the only
   synthesis author;
4. preserve the existing precedence of `--trace-idle-gap-cap-seconds` over
   `--synthesis-idle-gap-cap`; and
5. leave non-synthesis YAML resolution and runtime validation unchanged.

For `baseten_trace`, the existing native validation remains the source of truth:
`max_isl`/`max_osl`-only synthesis stays valid, while reshaping fields still
fail with the current loader-specific error.

## Implementation boundary

Implement the overlay in `rust/cli/src/yaml.rs`, reusing the same synthesis
field semantics as `load.rs` without replacing the whole YAML-authored object.
The change should operate on `Inputs.synthesis` after YAML normalization and
before runtime resolution.

Add focused Rust tests in `rust/cli/src/yaml.rs` that prove:

- an explicit CLI synthesis value overrides the same YAML key;
- YAML-only synthesis keys survive the overlay; and
- explicit `--cache-bust none` removes an authored cache-bust target.
