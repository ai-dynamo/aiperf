# Graham review — actual upstream ancestry merge

## Scope

Reviewed the real merge of upstream `d8d49e8c2adc76072625c0789ac1029967b639a5`
into shared tip `0b0482b27400d3f903e9a63df304602ed312a6c2`.

## Conflict resolution

The converter retains upstream's parsed search-space shape inference, seeds,
validation, and Optuna phase-error framing. Shared #61's DSP kernel behavior
is retained: its qLogNEI call site derives the BoTorch batch shape and its
regressions continue to exercise constrained output counts. The only manual
repair after combining the test variants was restoring the class docstring
terminator; it changes no test or product behavior.

## Findings

No Critical or Important findings. The combined Python paths do not introduce
new synchronization, async work, allocation strategy, logging, or public API
changes beyond the upstream contract. The merged #61 test uses BoTorch's own
`SingleTaskGP.get_batch_dimensions`, which is more robust than reconstructing
the batch dimension manually.

## Verification

With `PYTHONPATH` rooted at this worktree, the focused converter, Optuna,
BoTorch-kernel, and constrained-qLogNEI suites reported 122 passed and 4
deselected. This explicit path is required because the shared virtualenv's
editable install otherwise imports the original shared checkout.
