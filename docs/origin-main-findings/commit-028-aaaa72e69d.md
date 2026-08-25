# Origin commit aaaa72e69d: arrival tutorial output flag

## Finding and audit

This documentation-only commit corrects three examples in
`docs/tutorials/arrival-patterns.md` from the obsolete `--output-dir` spelling
to the native `--output-artifact-dir` flag. It changes no runtime code and
adds no upstream tests, integration scenarios, or E2E coverage.

The native CLI exposes `--output-artifact-dir`, so the correction is directly
applicable as documentation parity. No Rust implementation or test port is
needed.

## Closure

The exact non-fast-forward merge is retained. Documentation diff checks and
the repository documentation validation hook pass. Graham review has no native
code scope and found no findings.
