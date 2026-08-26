# Native Baseten Columnar Load Graham Review

## First review

The first independent Graham review did not approve the implementation. It
reported eight Important findings:

1. registry auto-detection attempted JSON probing before columnar detection;
2. the Arrow IPC memory contract overstated the dependency's decode bound;
3. selected Arrow values used scalar rendering and generic JSON reparsing;
4. sources without a session column were still ratio-sampled;
5. `max_rows` was applied after session sampling;
6. direct loader and registry seed authority was not explicit or tested;
7. columnar diagnostics/origins lacked path, column, and stable row ordinal;
8. session-column policy and `.ipc` composition parity coverage was incomplete.

## Repairs

- `90ffc47336` makes registry probing inspect columnar schema without reading a
  binary file as JSON and adds a registry-level regression.
- `8d8a17b474` adds checked typed-array decoding, direct `BasetenRow`
  construction, allocation sized to the retained mask, no-session sampling
  bypass, pre-sampling `max_rows`, an explicit direct-load RNG builder and
  registry parity test, path/column/ordinal diagnostics and origins, complete
  session policy coverage, and actual `.ipc` composition parity.
- `59ae696f21` narrows the Arrow IPC contract: Parquet decode batches are at
  most 128 projected rows; IPC decode is bounded by the largest authored
  projected record batch and downstream processing uses zero-copy 128-row
  slices.

The repaired focused suite passes 17 tests, public adapter integration passes,
scoped formatting and library Clippy pass with existing warnings, and the
three-sample release benchmark preserves all semantic counts while improving
median elapsed time by 19.4% and median RSS by 25.1% from baseline.

## Fresh review

Pending independent review of `da917561fb..HEAD`. Closure requires an explicit
no-unresolved-Critical-or-Important verdict; this document does not self-approve.
