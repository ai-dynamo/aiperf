# E09: Reject lossy multi-dataset YAML before lowering

## Problem

The native YAML adapter models both the single-entry `dataset:` shorthand and
the expanded `datasets:` list.  Its single-run lowering currently chooses the
shorthand when present, otherwise consumes `datasets.into_iter().next()`.  A
second authored dataset is therefore accepted and discarded without a
diagnostic, despite the protocol-v2 execution model having one
`Dataset`/factory per run.

## Contract

- The native YAML single-run adapter accepts exactly one explicitly authored
  dataset when either dataset form is present.
- An expanded list with any length other than one fails before prompt
  extraction, dataset-factory selection, or protocol-v2 projection.
- The failure names `datasets` and states that exactly one dataset is
  supported; it does not imply that multi-dataset composition was performed.
- A valid `dataset:` shorthand continues to lower byte-for-byte as today.
- A valid one-entry `datasets:` form continues to lower to the same inputs as
  the shorthand.
- Supplying both forms is rejected rather than assigning precedence and
  discarding one authored value.
- Omitting both forms retains the established synthetic-default behavior; it
  is not an authored zero-dataset request.

## Non-goals

This task does not add multi-dataset composition, per-phase dataset selection,
multiple protocol-v2 dataset factories, or change the typed runtime
`BenchmarkConfig` contract.  It is a truthful parser boundary: unsupported
authoring fails before adaptation.

## Acceptance evidence

The implementation must first demonstrate RED coverage for an explicitly empty
expanded list, a two-entry expanded list, and simultaneous shorthand plus
expanded forms.  GREEN coverage must demonstrate the stable one-entry
expanded, shorthand, and omitted-dataset default paths.  The tests must call
the public YAML resolver so that the pre-adaptation boundary is exercised.
Focused tests use a distinct
`CARGO_TARGET_DIR` under `/mnt/4tb`; the resulting change requires an
independent Graham review before integration.
