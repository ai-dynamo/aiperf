# Native LiveCodeBench Codegen Worker

## Problem

The native Rust accuracy path launches `aiperf.accuracy.worker` and asks it to
grade `lcb-codegeneration` batches. That worker currently calls
`codegen_metrics` from `asyncio.to_thread`. The metric forks its own sandboxed
process pool, so it can fork from a non-main thread. Upstream commit
`817a8d84ddb9` fixes the equivalent Python record-processor path by delegating
the operation to the already-available, single-threaded
`CodegenGradingWorker` subprocess.

## Decision

Use one lazily created `CodegenGradingWorker` per `AccuracyWorker` only when
the loaded benchmark selects the canonical LiveCodeBench batch grader. In
`_grade_lcb_batch`, retain all existing payload decoding, code extraction,
batch ordering, score validation, and result construction; replace only the
direct `asyncio.to_thread(_run_codegen_metrics, ...)` call with one framed
`grade_codegen` request to that child.

The worker request receives the entire existing batch to preserve the
canonical aggregate/detail mapping. Its timeout is derived from the batch's
total test-case count using the same bounded policy as the upstream grader.
`CodegenWorkerError` becomes a contextual evaluator error, so the native
Rust caller receives the existing structured worker-operation failure rather
than silently treating a faulty grade as a score.

`AccuracyWorker.close()` closes the owned codegen worker, then clears the
reference. Replacing a loaded benchmark first closes an existing worker, so
LiveCodeBench resources cannot survive a later load. Non-LiveCodeBench
benchmarks never create this extra subprocess.

## Interfaces

- `AccuracyWorker._codegen_worker: CodegenGradingWorker | None` is owned by
  the evaluator process, not Rust hot-path workers.
- `AccuracyWorker._grade_lcb_batch(...)` uses
  `CodegenGradingWorker.grade_codegen(evaluation_sample, generated_code,
  timeout)` once per submitted batch.
- `AccuracyWorker.close()` remains idempotent and awaits the child close.

No Rust JSONL protocol, Rust public API, score schema, prompt, or metric field
changes are allowed.

## Tests

Add focused `test_accuracy_worker.py` coverage with a fake codegen worker to
prove batch delegation, bounded timeout input, preservation of batch order,
and close/reload cleanup. The tests must fail before production changes and
use the existing child-worker tests as the subprocess/reaping regression
coverage. Run the focused Python accuracy-worker and codegen-worker suites,
then the Rust accuracy worker tests using the configured `sccache` and an
`/mnt/4tb` target directory.

## Acceptance criteria

1. Native LiveCodeBench grading never invokes `codegen_metrics` through
   `asyncio.to_thread`.
2. Exactly one child-worker request represents each nonempty native LCB batch.
3. Existing score and result ordering semantics remain unchanged.
4. Close and replacement-load reap the delegated grader process.
5. Focused Python and Rust tests pass, and Graham review finds no unresolved
   issue in the port diff.
