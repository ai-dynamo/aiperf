# Kubernetes-vs-Local Correctness Audit Suite

**Date:** 2026-04-26
**Status:** Spec, pending plan
**Branch:** `ajc/k8s`
**Related:** `tests/kubernetes/`, `tests/integration/`, `src/aiperf/cli_commands/kube/results.py`, `docs/kubernetes/direct-mode.md`

## Problem

AIPerf has two production-relevant ways to run a benchmark:

1. **Operator-managed** — `aiperf kube profile/sweep` submits an `AIPerfJob` (or `AIPerfSweep`) CR; the kopf operator schedules a JobSet (one controller pod + N worker pods); results are downloaded by the user via `aiperf kube results <id>` against the operator's PVC.
2. **Single-process local** — `aiperf profile <args>` runs in one process and writes artifacts to a local directory.

The two paths share the request-payload code, dataset code, exporters, and metric formulas, but diverge sharply in orchestration: controller↔worker credit distribution, multi-pod scheduling, results-ready marker gating, sidecar serving, results-API → CLI download serialization. A bug in any orchestration-layer component can silently produce artifacts that look superficially correct but disagree with what a single-process run would have produced for the same workflow.

There is no test today that proves the artifacts shipped by `aiperf kube results` match (within tolerance) what `aiperf profile` would produce for the same workflow. The integration tests (`tests/integration/`) cover the local subprocess path against host endpoints; the kubernetes tests (`tests/kubernetes/`) cover deployment, RBAC, scaling, and per-feature smoke checks of the operator path. Neither cross-checks the two against each other.

## Approach

A new opt-in pytest module — `tests/kubernetes/audit/` — runs each of five tightly-scoped workflow cases twice against the same in-cluster mock server: once via the full operator path (download via `aiperf kube results`), once via a bare `batch/v1.Job` running `aiperf profile` directly. The two artifact trees are diffed through a three-bucket comparator (exact, tolerance, structural) and any divergence fails the test with a structured report.

The bare-pod side is the oracle: same image, same in-cluster network, but no controller, no workers, no operator, no CRD, no results sidecar, no PVC, no download CLI. If the operator side disagrees with the bare side beyond bucket tolerances, the operator side is wrong. The audit explicitly exercises `aiperf kube results` (the user-facing download path) on the operator side — proving that path produces correct, complete artifacts is part of the audit's purpose.

## Goals

- Catch divergence between the operator-managed run and a single-process run for the same workflow, as observed in the artifact tree the user actually downloads.
- Exercise the user-facing `aiperf kube results <id>` download path end-to-end (operator → PVC → CLI → local disk).
- Provide a tight, opt-in test surface (~16 min/suite) suitable for nightly CI.
- Produce structured, machine-readable diff reports (`audit-report.json`) and human-readable summaries (`report.md`) on failure.

## Non-goals

- Mirror the integration-test endpoint zoo. Endpoint-payload code is shared between modes; adding embeddings/images/video/multimodal cases gives diminishing returns and inflates CI time. The suite can grow case-by-case once the harness exists.
- Compare absolute latencies exactly. Latency stats are inherently noisy under k8s scheduling; the tolerance bucket handles them with looser bands on tails.
- Replace the existing `tests/kubernetes/` deployment / scaling / chaos coverage. The audit is additive.
- Audit the no-operator direct mode (`aiperf kube profile --no-operator`). That is a third path; if it ever needs auditing it can reuse the same harness with a different deployer.

## Architecture

### Layout

```
tests/kubernetes/audit/
  __init__.py
  conftest.py            # marker registration, --audit-repeats option, fixtures
  cases.py               # AuditCase dataclasses + parametrize list
  bare_pod.py            # BarePodDeployer: raw Job + kubectl cp
  operator_runner.py     # OperatorAuditRunner: AIPerfJob deploy + `aiperf kube results`
  diff.py                # three-bucket comparator + AuditFindings
  report.py              # markdown + JSON renderers
  test_audit.py          # one parametrized test per case
```

### Pytest marker

`@pytest.mark.k8s_audit`, registered in `pyproject.toml` under `[tool.pytest.ini_options].markers`. The audit suite is **not** collected by `pytest tests/kubernetes/` runs by default; it is invoked explicitly:

```bash
uv run pytest -m k8s_audit tests/kubernetes/audit/ -n auto
```

### Reused fixtures

The audit conftest does not stand up its own cluster. It depends on:

- `local_cluster` — kind cluster from `tests/kubernetes/conftest.py`.
- `kubectl` — `KubectlClient` bound to the test cluster.
- `mock_server` — in-cluster deterministic mock LLM server.
- `aiperf_image` — `aiperf:local` built and loaded by the existing harness.
- `helm_deployer`, `operator_deployer` — for operator-mode runs (already present).

### New components

**`AuditCase`** (`cases.py`):

```python
@dataclass(frozen=True)
class AuditCase:
    case_id: str
    profile_args: list[str]          # passed verbatim to both modes
    epochs: int = 1
    sweep: dict[str, list] | None = None  # for the sweep case only
    seed: int = 42
    metric_tolerance_overrides: dict[str, float] = field(default_factory=dict)
    expected_artifacts: frozenset[str] = frozenset()  # required filenames
```

**`BarePodDeployer`** (`bare_pod.py`):

- Constructs a `batch/v1.Job` running `aiperf profile <args> --output /aiperf-output --random-seed <seed>` against the in-cluster mock-server URL, on the same image.
- One pod per Job. No replicas, no JobSet.
- Waits for pod terminal phase.
- `kubectl cp <pod>:/aiperf-output <local-dir>`.
- Deletes the Job (cascades pod).
- Returns the local artifact path.

**`OperatorAuditRunner`** (`operator_runner.py`):

- Wraps the existing `OperatorDeployer.deploy()` to submit an `AIPerfJob` CR with the case's profile args.
- For the `multi-epoch` and `small-sweep` cases, submits the appropriate CR shape (`AIPerfJob` with `epochs: N`, or `AIPerfSweep` per the existing sweep design).
- Waits for terminal phase.
- Subprocess-shells `aiperf kube results <job_id> --output <local-dir> --all` against the cluster.
- On success, deletes the CR (or relies on `ttlSecondsAfterFinished`).
- Returns the local artifact path.

**`AuditFindings`** (`diff.py`):

```python
@dataclass(frozen=True)
class Finding:
    bucket: Literal["exact", "tolerance", "structural"]
    field: str           # dotted path or filename
    expected: object     # bare-pod (oracle) value
    actual: object       # operator-side value
    reason: str          # human-readable

@dataclass(frozen=True)
class AuditFindings:
    case_id: str
    findings: list[Finding]
    @property
    def empty(self) -> bool: return not self.findings
```

### Three-bucket diff

Implemented in `diff.py` as three pure functions over the two artifact trees:

- **`diff_exact(operator_dir, bare_dir, case) -> list[Finding]`**
  - `inputs.json`: configured-args echo (concurrency, request-count, num-conversations, model, endpoint, seed) must match exactly.
  - Request count, conversation count, completion-record count: extracted from per-record exporter output (CSV/parquet); must match exactly.
  - Error count = 0 on both sides; status = success on both sides.
  - Dataset hash: deterministic-seed runs produce identical dataset payload; SHA-256 of the dataset export must match.
  - Exporter file *set* (filenames in the artifact root) must match exactly. File contents are checked by tolerance/structural buckets.
- **`diff_tolerance(operator_dir, bare_dir, case) -> list[Finding]`**
  - Loads `profile_export_genai_perf.json` (canonical metrics summary) from both sides.
  - For every numeric field, computes `|a − b| / max(|a|, |b|, eps)`.
  - Default bands:
    - means / medians: ≤ 10%
    - p90 / p95 / p99: ≤ 25%
    - throughput (RPS, output-tokens-per-sec, etc.): ≤ 10%
  - `case.metric_tolerance_overrides` overrides per-metric for cases where contention is structurally higher.
- **`diff_structural(operator_dir, bare_dir, case) -> list[Finding]`**
  - Required filenames in `case.expected_artifacts` are present in **both** trees.
  - For each shared file: CSV header equality; parquet column-set equality; JSON top-level key set + depth-1 keys equality. Values are not compared here (handled by tolerance/exact buckets).

The test calls all three, concatenates findings, writes the report, asserts `findings.empty`.

### Reporting

`report.py` renders an `AuditFindings` object to:

- `tests/_artifacts/<case-id>/audit-report.json` — `{case_id, findings: [...]}` (machine-readable, the schema matches the dataclass).
- `tests/_artifacts/<case-id>/report.md` — markdown with three sections (Exact / Tolerance / Structural), each listing the violating fields, expected vs. actual values, and computed deltas. Headed by a one-line PASS/FAIL summary and the case args.

On failure, the test prints the markdown to stdout and attaches both raw artifact trees to the pytest failure (already preserved on disk).

### Determinism

- Per-case fixed `--random-seed`; both sides receive the same seed.
- Same `aiperf:local` image (already pinned by the existing harness via `ImageManager`).
- Sequential runs by default — operator side completes and tears down before the bare side starts. Eliminates cross-pod CPU contention so tolerance bands stay tight enough to be meaningful.
- Single repeat per side by default. Opt-in `--audit-repeats N` (registered in `audit/conftest.py::pytest_addoption`); when N > 1, each side runs N times, the per-metric median across repeats is taken per side, and tolerance bands are computed against medians. Used locally to investigate a specific divergence; not enabled in CI.
- Mock-server determinism is inherited from the existing `mock_server` fixture; no new flags.

## Workflow cases (tight: 5)

1. **`baseline-chat`** — `--endpoint-type chat`, `--concurrency 4`, `--num-conversations 32`, default request shape. Exercises the most common path; minimum baseline.
2. **`baseline-completions`** — same as above with `--endpoint-type completions`. Catches per-endpoint exporter divergence.
3. **`concurrency-scale`** — `--endpoint-type chat`, `--concurrency 16`, `--num-conversations 64`. Operator side exercises multi-worker credit distribution; bare side runs the same load from a single process. Tolerance bands stay loose on tails because the bare process is single-CPU-bound vs. multi-pod parallel.
4. **`multi-epoch`** — `--endpoint-type chat`, `--concurrency 4`, `--num-conversations 16`, `--epochs 3`. Operator submits one `AIPerfJob` with `epochs: 3`; bare runs `aiperf profile` three times sequentially with `--epoch-index k` (or the equivalent local epoch flag) into the same output dir. Asserts the per-epoch artifact set on each side matches the other.
5. **`small-sweep`** — `--endpoint-type chat`, sweep on `concurrency: [4, 16]`, `--num-conversations 32`. Operator submits one `AIPerfSweep` with two variations; bare runs `aiperf profile` twice sequentially with the swept arg substituted. Asserts the per-variation artifact set matches.

## What this catches

- `aiperf kube results` dropping or corrupting files during download.
- Exporter parity drift between the controller-pod orchestration path and the single-process path.
- Results-ready marker race surfacing as missing files only on the operator side.
- CRD spec → controller arg translation losing a flag (silent default-fallback) vs. the direct `aiperf profile` invocation.
- Sweep planner producing artifacts whose layout, filenames, or content differ from N independent sequential profile runs.
- Multi-worker credit distribution diverging from single-process request totals.
- Schema drift in any exporter where the operator path wraps or unwraps a layer the local path does not.

## What this does not catch

- Bugs that exist identically in both modes (shared exporter / metric-formula bugs). Those need a separate cross-check against an independent oracle.
- Latency regressions. The tolerance bands are too loose to flag a 10–20% real-world latency degradation.
- Endpoint-payload bugs in unexercised endpoints (embeddings, images, video, multimodal). Add a case if/when one is suspected.

## Runtime budget

5 cases × (~90 s operator run + ~90 s bare run + ~5–10 s diff + teardown) ≈ **~16 min/suite** on the existing kind harness with `-n auto`. Cases are independent and parallelize cleanly across pytest workers (each worker gets its own namespace; the cluster and mock server are shared).

## Testing strategy

The audit suite *is* the test. There is no test-of-the-test beyond:

- Unit tests for `diff.py` (`tests/unit/kubernetes/audit/test_diff.py`): synthetic artifact trees with known divergences in each bucket, asserting the right `Finding`s are produced. No cluster required.
- Unit tests for `report.py`: golden-file rendering of a synthetic `AuditFindings` to markdown and JSON.

The deployers (`bare_pod.py`, `operator_runner.py`) are exercised end-to-end by the audit cases themselves; mocking them out would defeat the suite's purpose.

## Open questions

None blocking. Two adjacency notes for the implementation plan:

- The `multi-epoch` and `small-sweep` cases depend on the local CLI's epoch flag and sweep flag being argument-compatible with the operator-submitted shape. If they are not (e.g. local CLI takes `--epochs N` while the operator translates `epochs: N` from the CR into a different shape), the bare-pod runner needs a small per-case translation hook in `cases.py`. This is an implementation detail, not a design risk.
- `aiperf kube sweep` writes per-variation results into the epoch-keyed layout (`<base>/<ns>/<name>/<epoch>/<variation>/`); the structural diff for the sweep case must compare the **per-variation** subtree against each bare run's flat output, not the wrapping epoch directory.
