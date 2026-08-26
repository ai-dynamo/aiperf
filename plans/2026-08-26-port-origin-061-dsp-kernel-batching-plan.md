# Port origin/main #61: DSP kernel batching

1. Add the upstream regression tests for batched DSP kernels and constrained
   qLogNEI candidate fitting; run them before production edits to record the
   missing-batch-shape failure.
2. Add optional batch-shape propagation to the DSP kernel factory and derive
   the augmented GP shape at the qLogNEI call site.
3. Run the focused BoTorch tests and the feature-gated Rust CLI library tests
   with `sccache` and an isolated Cargo target. Record exact results in the
   finding and campaign ledger, then perform a Graham review of the range.
