# Native DSP kernel batching

## Scope

The native `aiperf --search-style bo` execution path embeds CPython through
`rust/cli/src/pyopt.rs`, then loads AIPerf's qLogNEI candidate factory. A
constrained probe passes a training matrix whose output columns represent the
objective and constraints. The qLogNEI helper must construct a batch-compatible
DSP kernel before fitting the `SingleTaskGP`.

## Design

Extend `make_dsp_kernel` with an optional `torch.Size` batch shape and pass it
to both `MaternKernel` and its enclosing `ScaleKernel`. In the qLogNEI factory,
derive the shape through `SingleTaskGP.get_batch_dimensions(train_X, train_Y)`
after concatenating the objective and constraint columns, and pass the returned
augmented batch shape to the kernel factory.

Leaving the argument absent preserves the unbatched single-output behavior.
The factory must not calculate `Size([m])` manually because BoTorch owns any
leading input-batch semantics.

## Verification

Focused BoTorch tests must prove kernel parameter batching, unbatched defaults,
and successful qLogNEI candidate fitting with one and two SLA filters. Existing
Rust CLI tests remain the proof that the feature-gated Pyo3 bridge compiles and
owns the selected sampler path.
