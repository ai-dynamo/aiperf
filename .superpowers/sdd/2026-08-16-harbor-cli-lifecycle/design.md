# Harbor CLI lifecycle record design

## Problem

`aiperf eval` currently imports a package, invokes a backend, and emits only
task/artifact/reward JSON. `HarborEvaluationCoordinator::execute_local` can
create a trial, verifier result, immutable score, regrade, and evidence, but
it requires inputs the CLI neither accepts nor persists: agent variant, model,
seed, policy, runtime, paired budgets, attempt id, initial metric/rationale,
and regrade metric/rationale. Supplying invented defaults would make the
record look reproducible when it is not.

The local backend now rejects a separate verifier before provisioning. Docker
remains the only product backend for a separately provisioned verifier.

## Proposed contract

Add a versioned `--lifecycle-request <path>` JSON DTO, parsed strictly in
`aiperf-cli` and passed unchanged to runtime composition. It contains:

- `agent_variant`, `model`, `seed`, `policy`, `runtime`, and `attempt`;
- `budget.execution_seconds` and `budget.verifier_seconds`;
- `initial_score { metric, rationale }` and `regrade { metric, rationale }`;
- `agent_contract` (`installed`, `external`, or `native_graph`) and the
  selected command provenance.

The CLI derives the immutable task, environment, verifier, package plan, and
actual backend from the imported snapshot; those fields are not user supplied.
It rejects a request whose external/installed contract disagrees with
`--agent-command`, whose budget disagrees with effective phase deadlines, or
whose requested verifier topology cannot be enforced by the selected backend.

## Runtime shape

Split coordinator execution into three explicit operations:

1. `resolve_trial(imported, lifecycle_request)` validates all identity inputs
   and constructs `TrialSpec` before provisioning.
2. `complete_attempt(imported, trial, command, execution, lifecycle_request)`
   creates `VerifierResult`, initial `ScoreVersion`, append-only regrade, and
   ordered evidence from the actual backend result.
3. Keep `execute_local` as a test/compatibility convenience implemented from
   those operations; add an executor-neutral product path used by both Docker
   and local backends.

Neither operation owns Docker nor duplicates artifact collection. The Docker
executor returns its existing immutable execution result; completion consumes
only that result and the resolved lifecycle request.

## Output and persistence

Preserve the existing three-key eval JSON when no lifecycle request is given.
With one, emit a versioned additive `lifecycle` object containing serialized
`trial`, `verifier_result`, `initial_score`, `regraded_score`, and `evidence`.
Write the same canonical object atomically to a caller-selected artifact path
or a documented default under the evaluation output root. This turns product
execution into a durable P0 record without changing legacy scripts.

## Test-first slices

1. CLI DTO rejects missing/nonfinite identities and conflicting contracts
   before Docker build or local process start.
2. A fake executor proves `resolve_trial` is constructed before provision and
   `complete_attempt` preserves reward-json precedence and exact artifact
   manifest evidence.
3. Daemon-backed standard-task E2E executes a separate verifier through Docker
   and asserts the emitted lifecycle chain, no undeclared file transfer, and
   immutable pinned-Git identity after origin mutation.
4. Local shared-verifier E2E emits a lifecycle record; local separate remains
   a pre-provision typed refusal.
