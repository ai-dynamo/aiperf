# Codegen Grade Concurrency — Design Spec

**Linear:** [AIP-1094](https://linear.app/nvidia/issue/AIP-1094/restore-codegen-grade-concurrency)
**Related:** AIP-1089 (PR #1175 — introduced the out-of-process worker)
**GitHub issue:** https://github.com/ai-dynamo/aiperf/issues/1189

---

## Problem

`CodegenGradingWorker` holds `asyncio.Lock` during every `grade_codegen()` call, fully
serializing all concurrent graders inside a record-processor. N simultaneous grades
therefore take ~sum(individual) time. `lighteval`'s `ProcessPoolExecutor(max_workers=8)`
sits mostly idle — it only gets one problem at a time and can't fan out.

### Root cause (two components)

**Client** (`_codegen_worker_client.py`): `asyncio.Lock` wraps the full request lifecycle
(write → drain → readline), so a second caller can't even send its request until the
first caller's response has been received.

**Worker** (`_codegen_worker.py`): sequential `for line in stdin` loop processes one
request at a time, so even without the lock the pool would only see one problem per cycle.

---

## Chosen approach: batch drain + single batched `codegen_metrics` call

No new worker processes, no lighteval internals, no transport change.

**Client:** drop the `asyncio.Lock`; add an `id → asyncio.Future` demux table and a
persistent reader task. All concurrent callers can write their requests immediately and
then await their individual futures.

**Worker:** after reading the first blocking request, non-blocking drain any
already-queued lines, then call `codegen_metrics` once with all N batched
samples/generations. `codegen_metrics` returns `(metrics, results)` where
`results: dict[int, list]` is keyed by batch index — so per-problem demux is possible
by calling `compute_metrics_from_results({0: results[i]})` for each `i`.

### Why this works

`lighteval`'s `evaluate_generations()` (called inside `codegen_metrics`) submits all N
problems to a `ProcessPoolExecutor` at once via `executor.submit`. With N > 1 problems in
a batch, the pool actually has work for multiple slots simultaneously, so N grades
complete in ~max(individual) rather than ~sum.

### Rejected alternatives

| Approach | Why rejected |
|---|---|
| `evaluate_generations_by_problem` streaming | Uses lighteval internal API; more complex worker state (persistent pool). Mentioned in issue as the non-preferred alternative. |
| Multi-worker pool (one worker process per grade) | Issue explicitly rules out "worker pool"; multiplies process/memory overhead. |

---

## Known tradeoff

Top-level batching couples latency across problems: if one problem in a batch is slow,
all batchmates wait. A per-caller `timeout` fires after the deadline and kills the
worker, canceling all currently-pending futures with `CodegenWorkerError`. Other callers
whose records were in that batch get a grading failure — same behavior as today's
per-grade timeout, just applied to more records at once. Accepted per the issue.

---

## Implementation

### Files changed

| File | Change |
|---|---|
| `src/aiperf/accuracy/graders/_codegen_worker_client.py` | Drop lock, add demux table + reader task |
| `src/aiperf/accuracy/graders/_codegen_worker.py` | Batch-drain loop + per-problem demux |
| `tests/unit/accuracy/test_codegen_worker_client.py` | New concurrency tests |
| `tests/unit/accuracy/test_codegen_worker.py` | Batch-loop tests |
| `tests/component_integration/test_lcb_codegen_worker_e2e.py` | Multi-problem concurrent test |

---

### Client changes (`_codegen_worker_client.py`)

#### Remove

```python
self._lock = asyncio.Lock()
```

and its `async with self._lock:` usage in `grade_codegen()` and `aclose()`.

#### Add to `__init__`

```python
self._pending: dict[int, asyncio.Future[dict]] = {}
self._reader_task: asyncio.Task[None] | None = None
```

#### `grade_codegen()` new flow

1. Check `_start_failures` cap (no lock needed — only written from `_handle_fault`).
2. `await _ensure_worker()` — lazy spawn; also starts `_reader_task` (see below).
3. Allocate `id = (self._next_id := self._next_id + 1)`.
4. Create future: `fut = asyncio.get_event_loop().create_future()`, store in `_pending[id]`.
5. Write JSONL to stdin immediately (no await on response).
6. `return await asyncio.wait_for(fut, timeout)`.
7. On `TimeoutError`: pop `_pending[id]`, `await _handle_fault(count_start_failure=False)`, raise `CodegenWorkerError`.
8. On `CancelledError`: pop `_pending[id]`, `await _handle_fault(count_start_failure=False)`, re-raise.

#### Reader task (`_run_reader`)

Runs as a persistent `asyncio.Task` for the lifetime of the worker process.

```
loop:
    line = await proc.stdout.readline()
    if not line:                              # EOF — worker died
        _handle_fault()                       # cancels all pending futures
        return
    try:
        resp = orjson.loads(line)
    except JSONDecodeError:
        _handle_fault()
        return
    fut = _pending.pop(resp["id"], None)
    if fut is None:
        continue                              # stale / already-timed-out id
    if resp["ok"]:
        fut.set_result(resp["metrics"])
        _mark_proven()
    else:
        fut.set_exception(CodegenWorkerError(resp.get("error", "unknown")))
        _mark_proven()                        # clean error response = worker works
```

#### `_handle_fault()` additions

After the existing kill logic, cancel and clear all pending futures:

```python
for fut in self._pending.values():
    if not fut.done():
        fut.set_exception(CodegenWorkerError("grading worker fault"))
self._pending.clear()
```

Also cancel `_reader_task` when faulting (it will exit on the EOF that follows the kill).

#### `_ensure_worker()` additions

Start `_reader_task = asyncio.create_task(self._run_reader(self._proc.stdout))`
alongside the existing `_stderr_task`.

#### `aclose()` (simplified — no lock needed)

```python
async def aclose(self) -> None:
    if self._reader_task is not None:
        self._reader_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._reader_task
    for fut in self._pending.values():
        if not fut.done():
            fut.cancel()
    self._pending.clear()
    await self._kill()
```

---

### Worker changes (`_codegen_worker.py`)

#### Replace `run_worker_loop` with a batch-drain loop

```python
import select

def run_worker_loop(stdin, out, codegen_fn):
    stdin_fd = stdin.fileno()
    while True:
        # Block until at least one request arrives (or EOF)
        first = stdin.readline()
        if not first:
            break  # EOF: client closed stdin, clean exit
        first = first.strip()
        if not first:
            continue
        batch = [first]

        # Non-blocking drain: collect any requests already queued in the pipe
        while True:
            ready, _, _ = select.select([stdin_fd], [], [], 0)
            if not ready:
                break
            line = stdin.readline()
            if not line:
                break  # EOF mid-drain; will be caught on next outer iteration
            line = line.strip()
            if line:
                batch.append(line)

        # Decode all requests
        reqs = []
        for raw in batch:
            try:
                reqs.append(orjson.loads(raw))
            except orjson.JSONDecodeError as exc:
                reqs.append({"_raw_err": str(exc)})  # sentinel for error response

        # Grade the batch and send responses
        for resp in handle_batch(reqs, codegen_fn):
            out.write(orjson.dumps(resp) + b"\n")
        out.flush()
```

#### Add `handle_batch()`

```python
def handle_batch(reqs, codegen_fn):
    # Validate and collect well-formed requests
    all_samples, all_generations, id_map = [], [], []
    error_resps = {}

    for i, req in enumerate(reqs):
        if "_raw_err" in req:
            error_resps[i] = {"id": None, "ok": False, "error": f"bad json: {req['_raw_err']}"}
            continue
        if not isinstance(req, dict):
            error_resps[i] = {"id": req.get("id") if isinstance(req, dict) else None, "ok": False, "error": "malformed request: expected object"}
            continue
        req_id = req.get("id")
        try:
            all_samples.append(req["evaluation_sample"])
            all_generations.append(req["generated_code"])
            id_map.append((i, req_id))
        except (KeyError, TypeError) as exc:
            error_resps[i] = {"id": req_id, "ok": False, "error": f"malformed request: {exc!r}"}

    # Single batched call to lighteval
    batch_results: dict[int, Any] = {}
    batch_error: str | None = None
    if all_samples:
        try:
            _, raw_results = codegen_fn(
                all_samples,
                all_generations,
                k_list=list(_LCB_PASS_AT_K),
                num_process_evaluate=_LCB_NUM_PROCESSES,
            )
            batch_results = raw_results
        except Exception as exc:
            batch_error = _truncate_error(f"{type(exc).__name__}: {exc}")

    # Build per-problem responses preserving original request order
    responses = [None] * len(reqs)
    for pos, (req_idx, req_id) in enumerate(id_map):
        if batch_error is not None:
            responses[req_idx] = {"id": req_id, "ok": False, "error": batch_error}
        else:
            per_problem_results = {0: batch_results[pos]}
            try:
                from lighteval.tasks.tasks.lcb.codegen_metrics import compute_metrics_from_results
                metrics, _ = compute_metrics_from_results(per_problem_results, k_list=list(_LCB_PASS_AT_K))
                responses[req_idx] = {"id": req_id, "ok": True, "metrics": _coerce_metrics(metrics)}
            except Exception as exc:
                responses[req_idx] = {"id": req_id, "ok": False, "error": _truncate_error(f"{type(exc).__name__}: {exc}")}

    for i, req in enumerate(reqs):
        if responses[i] is None:  # was in error_resps
            responses[i] = error_resps[i]

    return responses
```

> **Note on `compute_metrics_from_results`:** this is already imported at module level
> from lighteval (the worker has already completed its heavy import by the time
> `run_worker_loop` is called). The import inside `handle_batch` above is for clarity;
> in the actual implementation it should be captured at startup the same way
> `codegen_metrics` is, e.g. `from lighteval.tasks.tasks.lcb.codegen_metrics import
> codegen_metrics, compute_metrics_from_results`.

#### Keep `handle_request()` as a thin wrapper (for existing unit tests)

Or remove it if the unit tests are updated to test `handle_batch` directly. The
component integration test is more valuable here.

---

## Tests

### Unit — client (`test_codegen_worker_client.py`)

- **Concurrency**: N `grade_codegen()` tasks launched together; mock worker echoes
  responses in reverse order; assert all futures resolve to correct results (verifies
  id-based demux, not position-based).
- **Timeout doesn't serialize**: one slow grade times out; concurrent fast grades should
  not be blocked by the slow one (with lock they would be).
- **Fault cancels pending**: simulate EOF from mock worker; assert all in-flight futures
  raise `CodegenWorkerError`.
- **Stale id ignored**: reader receives a response for an id that already timed out;
  assert no exception, no crash.
- **aclose with pending**: call `aclose()` while futures are pending; assert they are
  cancelled cleanly.

### Unit — worker (`test_codegen_worker.py`)

- **Batch of one**: single request → `codegen_fn` called with 1-element lists →
  correct single response.
- **Batch of N**: pre-queue N requests before loop reads; assert `codegen_fn` called
  exactly once with N-element lists; assert N responses with correct ids in order.
- **Malformed request in batch**: one bad-JSON line in a batch of N; assert N responses,
  bad one gets `ok: false`, others get their metrics.
- **Batch-level exception**: `codegen_fn` raises; assert all N requests get `ok: false`
  error responses.

### Component integration (`test_lcb_codegen_worker_e2e.py`)

Add a test that launches the real worker and fires N concurrent `grade_codegen()` calls
using `asyncio.gather`. Assert all return `pass@1 == 1.0` and that wall-clock time is
closer to max than to sum of per-problem times.

---

## Acceptance criteria (from the issue)

- N concurrent grades in one record-processor complete in ~max(individual), not ~sum.
- Worker stays single-threaded at fork (no new threads introduced).
- Fault/timeout/restart paths still pass (existing tests green).
- New concurrency tests added and passing.

---

## Open questions

None — the issue's proposal is fully specified and the lighteval API confirms batching
is possible via `results: dict[int, list]` from `evaluate_generations`.
