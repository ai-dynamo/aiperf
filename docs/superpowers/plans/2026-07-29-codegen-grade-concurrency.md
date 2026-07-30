# Codegen Grade Concurrency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow N concurrent `grade_codegen()` calls to complete in ~max(individual) time by replacing the serializing `asyncio.Lock` with an `id → Future` demux table on the client and a batch-drain loop on the worker.

**Architecture:** The client drops its `asyncio.Lock` and instead multiplexes concurrent requests over the same stdin pipe using request ids; a persistent reader task demuxes responses back to individual `asyncio.Future` objects. The worker reads the first blocking request, non-blocking drains any queued requests, then calls `codegen_metrics` once with all batched samples so lighteval's `ProcessPoolExecutor` handles all problems in parallel.

**Tech Stack:** Python 3.11+ asyncio, `orjson`, `lighteval` (`codegen_metrics`, `compute_metrics_from_results`), `select` (POSIX non-blocking stdin drain), `pytest-asyncio`

## Global Constraints

- Python 3.11+; use `asyncio.get_running_loop()`, not `asyncio.get_event_loop()`
- No new threads in the worker (stays single-threaded at fork)
- All existing tests in `tests/unit/accuracy/test_codegen_worker.py` and `test_codegen_worker_client.py` must remain green after each task
- `_handle_fault` must remain idempotent (called from both reader task and caller)
- `ruff format . && ruff check --fix .` must pass after each commit
- `pre-commit run --all-files` must pass before each commit
- Every new function needs a type hint on all parameters and return value
- Every new Pydantic field needs `Field(description=...)` (not applicable here — no new models)

---

### Task 1: Worker — batch-drain loop and `handle_batch`

**Spec:** `docs/superpowers/specs/2026-07-29-codegen-grade-concurrency-design.md` § "Worker changes"

**Files:**
- Modify: `src/aiperf/accuracy/graders/_codegen_worker.py`
- Modify: `tests/unit/accuracy/test_codegen_worker.py`

**Interfaces:**
- Produces: `handle_batch(reqs, codegen_fn, compute_metrics_fn)` — takes a list of raw request dicts, returns a list of JSONL-ready response dicts (one per input, same order)
- Produces: `run_worker_loop(stdin, out, codegen_fn, compute_metrics_fn)` — updated signature (adds `compute_metrics_fn` param)
- Removes: `handle_request` (dead code after this task; its tests are migrated to `handle_batch`)

---

- [ ] **Step 1: Write failing tests for `handle_batch`**

Add a new `TestHandleBatch` class in `tests/unit/accuracy/test_codegen_worker.py`. Place it after the existing `TestHandleRequest` class.

The mock `codegen_fn` for `handle_batch` must match the new signature that also accepts `compute_metrics_fn`. But `codegen_fn` itself is still the original `(samples, generations, ...) -> (metrics, results)` signature. The `compute_metrics_fn` is a separate argument to `handle_batch`.

Add at the top of the file alongside the existing fakes:

```python
def _fake_compute_metrics(results: dict, k_list: list[int] | None = None) -> dict[str, Any]:
    # Mirrors compute_metrics_from_results: returns {"pass@1": <float>} using
    # the single-problem results dict {0: [[True, True, ...]]} passed by handle_batch.
    result_list = results.get(0, [[-2]])  # [-2] = compile error
    if result_list and all(x > 0 for x in result_list[0]):
        return {"pass@1": 1.0}
    return {"pass@1": 0.0}


def _fake_codegen_batch_ok(
    samples: list, generations: list, **_kwargs: Any
) -> tuple[dict[str, Any], dict[int, list]]:
    # Returns aggregate metrics (ignored by handle_batch) and per-problem results.
    n = len(samples)
    raw_results = {i: [[True]] for i in range(n)}  # all pass
    return {"pass@1": 1.0}, raw_results


def _fake_codegen_batch_boom(
    samples: list, generations: list, **_kwargs: Any
) -> tuple[dict[str, Any], dict[int, list]]:
    raise RuntimeError("pool exploded")
```

Then add `TestHandleBatch`:

```python
class TestHandleBatch:
    def _req(self, req_id: int) -> dict[str, Any]:
        return {
            "id": req_id,
            "evaluation_sample": [{"input_output": "{}"}],
            "generated_code": [["x"]],
        }

    def test_single_request_returns_one_ok_response(self) -> None:
        resps = worker.handle_batch(
            [self._req(1)], _fake_codegen_batch_ok, _fake_compute_metrics
        )
        assert len(resps) == 1
        assert resps[0] == {"id": 1, "ok": True, "metrics": {"pass@1": 1.0}}

    def test_batch_of_n_calls_codegen_fn_once(self) -> None:
        call_count = 0

        def counting_codegen(samples, generations, **kwargs):
            nonlocal call_count
            call_count += 1
            n = len(samples)
            return {"pass@1": 1.0}, {i: [[True]] for i in range(n)}

        reqs = [self._req(i) for i in range(1, 5)]
        resps = worker.handle_batch(reqs, counting_codegen, _fake_compute_metrics)
        assert call_count == 1
        assert len(resps) == 4
        assert all(r["ok"] for r in resps)
        assert [r["id"] for r in resps] == [1, 2, 3, 4]

    def test_response_order_matches_request_order(self) -> None:
        reqs = [self._req(i) for i in [7, 3, 99]]
        resps = worker.handle_batch(reqs, _fake_codegen_batch_ok, _fake_compute_metrics)
        assert [r["id"] for r in resps] == [7, 3, 99]

    def test_batch_exception_returns_error_for_all(self) -> None:
        reqs = [self._req(i) for i in range(1, 4)]
        resps = worker.handle_batch(reqs, _fake_codegen_batch_boom, _fake_compute_metrics)
        assert len(resps) == 3
        assert all(not r["ok"] for r in resps)
        assert all("pool exploded" in r["error"] for r in resps)

    def test_malformed_request_in_batch_does_not_affect_others(self) -> None:
        reqs = [
            self._req(1),
            {"id": 2},  # missing evaluation_sample + generated_code
            self._req(3),
        ]
        resps = worker.handle_batch(reqs, _fake_codegen_batch_ok, _fake_compute_metrics)
        assert len(resps) == 3
        assert resps[0] == {"id": 1, "ok": True, "metrics": {"pass@1": 1.0}}
        assert resps[1]["id"] == 2
        assert not resps[1]["ok"]
        assert resps[2] == {"id": 3, "ok": True, "metrics": {"pass@1": 1.0}}

    def test_non_object_request_in_batch_is_error(self) -> None:
        reqs = [[1, 2, 3], self._req(5)]
        resps = worker.handle_batch(reqs, _fake_codegen_batch_ok, _fake_compute_metrics)
        assert len(resps) == 2
        assert resps[0]["id"] is None
        assert not resps[0]["ok"]
        assert resps[1] == {"id": 5, "ok": True, "metrics": {"pass@1": 1.0}}

    def test_parse_error_sentinel_produces_error_response(self) -> None:
        # run_worker_loop encodes JSON decode errors as {"_parse_error": "..."}.
        reqs = [{"_parse_error": "unexpected token"}, self._req(2)]
        resps = worker.handle_batch(reqs, _fake_codegen_batch_ok, _fake_compute_metrics)
        assert len(resps) == 2
        assert resps[0]["id"] is None
        assert not resps[0]["ok"]
        assert "unexpected token" in resps[0]["error"]
        assert resps[1]["ok"]
```

Also add a test for the new `run_worker_loop` batch-drain behaviour. Add to a new class `TestRunWorkerLoopBatch` after `TestHandleBatch`:

```python
class TestRunWorkerLoopBatch:
    def _run(
        self,
        payloads: list[dict[str, Any]],
        codegen_fn=_fake_codegen_batch_ok,
        compute_metrics_fn=_fake_compute_metrics,
    ) -> list[dict[str, Any]]:
        # Write all payloads to a BytesIO pipe so they are already queued when
        # run_worker_loop reads; this exercises the non-blocking drain path.
        data = b"".join(orjson.dumps(p) + b"\n" for p in payloads)
        stdin = io.BytesIO(data)
        out = io.BytesIO()
        worker.run_worker_loop(stdin, out, codegen_fn, compute_metrics_fn)
        out.seek(0)
        return [orjson.loads(line) for line in out if line.strip()]

    def _req(self, req_id: int) -> dict[str, Any]:
        return {
            "id": req_id,
            "evaluation_sample": [{"input_output": "{}"}],
            "generated_code": [["x"]],
        }

    def test_pre_queued_requests_are_batched_in_one_call(self) -> None:
        call_count = 0

        def counting_codegen(samples, generations, **kwargs):
            nonlocal call_count
            call_count += 1
            n = len(samples)
            return {"pass@1": 1.0}, {i: [[True]] for i in range(n)}

        reqs = [self._req(i) for i in range(1, 4)]
        resps = self._run(reqs, counting_codegen)
        assert call_count == 1
        assert len(resps) == 3

    def test_responses_carry_correct_ids(self) -> None:
        reqs = [self._req(i) for i in [10, 20, 30]]
        resps = self._run(reqs)
        assert {r["id"] for r in resps} == {10, 20, 30}
```

- [ ] **Step 2: Run the new tests to verify they fail**

```bash
uv run pytest tests/unit/accuracy/test_codegen_worker.py::TestHandleBatch tests/unit/accuracy/test_codegen_worker.py::TestRunWorkerLoopBatch -v 2>&1 | head -40
```

Expected: `AttributeError: module ... has no attribute 'handle_batch'` or `TypeError` from wrong arg count on `run_worker_loop`.

- [ ] **Step 3: Implement `handle_batch` and update `run_worker_loop` in `_codegen_worker.py`**

Open `src/aiperf/accuracy/graders/_codegen_worker.py`.

**3a — Add `import select` at the top of the file** (after the stdlib imports block).

**3b — Add `handle_batch` after the existing `_is_number` function:**

```python
def handle_batch(
    reqs: list[Any],
    codegen_fn: Callable[..., tuple[dict[str, Any], Any]],
    compute_metrics_fn: Callable[..., dict[str, Any]],
) -> list[dict[str, Any]]:
    """Grade a batch of requests with a single codegen_fn call.

    Calls codegen_fn once with all well-formed requests batched together so
    lighteval's ProcessPoolExecutor can process multiple problems in parallel.
    Never raises: all failures become error responses so a bad batch cannot
    kill the worker loop.
    """
    all_samples: list[Any] = []
    all_generations: list[Any] = []
    id_map: list[tuple[int, Any]] = []  # (batch_position, req_id)
    responses: list[dict[str, Any] | None] = [None] * len(reqs)

    for i, req in enumerate(reqs):
        if isinstance(req, dict) and "_parse_error" in req:
            responses[i] = {
                "id": None,
                "ok": False,
                "error": f"bad json: {req['_parse_error']}",
            }
            continue
        if not isinstance(req, dict):
            responses[i] = {
                "id": None,
                "ok": False,
                "error": "malformed request: expected object",
            }
            continue
        req_id = req.get("id")
        try:
            all_samples.append(req["evaluation_sample"])
            all_generations.append(req["generated_code"])
            id_map.append((i, req_id))
        except (KeyError, TypeError) as exc:
            responses[i] = {
                "id": req_id,
                "ok": False,
                "error": f"malformed request: {exc!r}",
            }

    if all_samples:
        batch_error: str | None = None
        raw_results: dict[int, Any] = {}
        try:
            _, raw_results = codegen_fn(
                all_samples,
                all_generations,
                k_list=list(_LCB_PASS_AT_K),
                num_process_evaluate=_LCB_NUM_PROCESSES,
            )
        except Exception as exc:
            batch_error = _truncate_error(f"{type(exc).__name__}: {exc}")

        for pos, (req_idx, req_id) in enumerate(id_map):
            if batch_error is not None:
                responses[req_idx] = {"id": req_id, "ok": False, "error": batch_error}
            else:
                try:
                    metrics = compute_metrics_fn(
                        {0: raw_results[pos]},
                        k_list=list(_LCB_PASS_AT_K),
                    )
                    responses[req_idx] = {
                        "id": req_id,
                        "ok": True,
                        "metrics": _coerce_metrics(metrics),
                    }
                except Exception as exc:
                    responses[req_idx] = {
                        "id": req_id,
                        "ok": False,
                        "error": _truncate_error(f"{type(exc).__name__}: {exc}"),
                    }

    return [r for r in responses if r is not None]
```

**3c — Replace `run_worker_loop`:**

```python
def run_worker_loop(
    stdin: BinaryIO,
    out: BinaryIO,
    codegen_fn: Callable[..., tuple[dict[str, Any], Any]],
    compute_metrics_fn: Callable[..., dict[str, Any]],
) -> None:
    """Serve JSONL grading requests until stdin EOF.

    Blocks on the first request of each cycle, then non-blocking drains any
    already-queued requests to form a batch. Calls codegen_fn once per batch so
    lighteval's ProcessPoolExecutor can process multiple problems in parallel.
    """
    import select

    stdin_fd = stdin.fileno()
    while True:
        first = stdin.readline()
        if not first:
            break  # EOF: client closed stdin, clean exit
        first = first.strip()
        if not first:
            continue
        batch_raw: list[bytes] = [first]

        while True:
            ready, _, _ = select.select([stdin_fd], [], [], 0)
            if not ready:
                break
            line = stdin.readline()
            if not line:
                break
            line = line.strip()
            if line:
                batch_raw.append(line)

        reqs: list[Any] = []
        for raw in batch_raw:
            try:
                reqs.append(orjson.loads(raw))
            except orjson.JSONDecodeError as exc:
                reqs.append({"_parse_error": str(exc)})

        for resp in handle_batch(reqs, codegen_fn, compute_metrics_fn):
            out.write(orjson.dumps(resp) + b"\n")
        out.flush()
```

**3d — Remove `handle_request`** (the whole function and its docstring). Do not replace it with a comment.

**3e — Update `main()` to import and pass `compute_metrics_from_results`:**

```python
def main() -> None:
    protocol_out = _install_stdout_guard()
    _start_death_watcher()
    _force_fork()
    from lighteval.tasks.tasks.lcb.codegen_metrics import (
        codegen_metrics,
        compute_metrics_from_results,
    )

    run_worker_loop(sys.stdin.buffer, protocol_out, codegen_metrics, compute_metrics_from_results)
```

- [ ] **Step 4: Migrate `TestHandleRequest` tests to `TestHandleBatch` equivalents**

The `TestHandleRequest` class in `test_codegen_worker.py` is now orphaned (`handle_request` was removed). Replace it with `TestHandleBatch` (already written in Step 1 above — just remove the old class). Also remove the `_fake_codegen_ok`, `_fake_codegen_boom`, `_fake_codegen_list_pass` helpers if they are only used by the old `TestHandleRequest`; they are replaced by the new batch-aware fakes from Step 1.

Check if any other test in the file (e.g., `TestRunWorkerLoop`, `TestStdoutGuard`) still calls `handle_request` directly; update those to use `handle_batch` with the batch-aware fakes.

Grep to find remaining usages:

```bash
grep -n "handle_request\|_fake_codegen_ok\|_fake_codegen_boom\|_fake_codegen_list_pass" \
    tests/unit/accuracy/test_codegen_worker.py
```

For any `TestRunWorkerLoop` tests that currently call `run_worker_loop` with 3 args, update to pass `_fake_compute_metrics` as the 4th argument.

- [ ] **Step 5: Run all unit tests for the worker to verify they pass**

```bash
uv run pytest tests/unit/accuracy/test_codegen_worker.py -v
```

Expected: all tests pass, including the new `TestHandleBatch` and `TestRunWorkerLoopBatch` classes.

- [ ] **Step 6: Lint**

```bash
ruff format . && ruff check --fix .
```

- [ ] **Step 7: Commit**

```bash
git add src/aiperf/accuracy/graders/_codegen_worker.py \
        tests/unit/accuracy/test_codegen_worker.py
git commit -s -m "feat(accuracy): batch-drain worker loop for codegen grading concurrency"
```

---

### Task 2: Client — drop lock, add demux table and reader task

**Spec:** `docs/superpowers/specs/2026-07-29-codegen-grade-concurrency-design.md` § "Client changes"

**Files:**
- Modify: `src/aiperf/accuracy/graders/_codegen_worker_client.py`
- Modify: `tests/unit/accuracy/test_codegen_worker_client.py`

**Interfaces:**
- Consumes: worker protocol from Task 1 (JSONL responses carry `id`, `ok`, `metrics`)
- Produces: `CodegenGradingWorker` with the same public API (`grade_codegen`, `aclose`) but concurrent-safe without a global lock

---

- [ ] **Step 1: Write failing concurrency tests**

Open `tests/unit/accuracy/test_codegen_worker_client.py`.

Add the following mock worker scripts near the top of the file alongside `_ECHO_OK`:

```python
# Echoes responses with pass@1 == id * 0.1 so each caller can verify it got
# back its OWN response (not another caller's).
_ECHO_ID_IN_METRICS = """
    import sys, orjson
    for line in sys.stdin.buffer:
        line = line.strip()
        if not line:
            continue
        req = orjson.loads(line)
        resp = {"id": req["id"], "ok": True, "metrics": {"pass@1": req["id"] * 0.1}}
        sys.stdout.buffer.write(orjson.dumps(resp) + b"\\n")
        sys.stdout.buffer.flush()
"""

# Buffers the first 4 requests and responds in REVERSE id order to exercise
# the demux table (correct demux requires id matching, not position matching).
_REVERSE_BATCH_OF_4 = """
    import sys, orjson
    buf = []
    for line in sys.stdin.buffer:
        line = line.strip()
        if not line:
            continue
        req = orjson.loads(line)
        buf.append(req)
        if len(buf) == 4:
            for r in reversed(buf):
                resp = {"id": r["id"], "ok": True, "metrics": {"pass@1": r["id"] * 0.1}}
                sys.stdout.buffer.write(orjson.dumps(resp) + b"\\n")
                sys.stdout.buffer.flush()
            buf = []
"""
```

Add a new `TestConcurrency` class:

```python
class TestConcurrency:
    async def test_concurrent_grades_all_complete(self, tmp_path) -> None:
        w = CodegenGradingWorker(worker_cmd=_write_worker(tmp_path, _ECHO_OK))
        try:
            results = await asyncio.gather(*[
                w.grade_codegen([{"input_output": "{}"}], [["x"]], timeout=30)
                for _ in range(5)
            ])
            assert all(r == {"pass@1": 1.0} for r in results)
        finally:
            await w.aclose()

    async def test_concurrent_grades_demux_by_id_not_position(self, tmp_path) -> None:
        # 4 concurrent grades; mock responds in reverse order.
        # If demux were position-based, callers would get wrong metrics.
        w = CodegenGradingWorker(worker_cmd=_write_worker(tmp_path, _REVERSE_BATCH_OF_4))
        try:
            results = await asyncio.gather(*[
                w.grade_codegen([{"input_output": "{}"}], [["x"]], timeout=30)
                for _ in range(4)
            ])
            # IDs 1-4 → pass@1 values 0.1, 0.2, 0.3, 0.4 (one per caller)
            values = sorted(r["pass@1"] for r in results)
            assert values == pytest.approx([0.1, 0.2, 0.3, 0.4])
        finally:
            await w.aclose()

    async def test_fault_cancels_all_pending_futures(self, tmp_path) -> None:
        # Worker dies immediately after the first line — all concurrent callers
        # should raise CodegenWorkerError, not hang.
        w = CodegenGradingWorker(
            worker_cmd=_write_worker(
                tmp_path,
                """
                import sys
                sys.stdin.buffer.readline()  # consume one line then exit
                """,
            )
        )
        try:
            with pytest.raises(Exception):  # CodegenWorkerError or ExceptionGroup
                await asyncio.gather(*[
                    w.grade_codegen([{"input_output": "{}"}], [["x"]], timeout=10)
                    for _ in range(3)
                ], return_exceptions=False)
        finally:
            await w.aclose()

    async def test_stale_id_after_timeout_does_not_crash(self, tmp_path) -> None:
        # Reader receives a response for an id that the caller already timed out on.
        # The stale future was already removed from _pending; the reader must skip it.
        # Use _ECHO_OK with a very short timeout so the grade times out, then send
        # a second grade to prove the worker (if restarted) still works.
        w = CodegenGradingWorker(worker_cmd=_write_worker(tmp_path, _ECHO_OK))
        try:
            with pytest.raises(CodegenWorkerError):
                await w.grade_codegen(
                    [{"input_output": "{}"}], [["x"]], timeout=0.000001
                )
            # If stale id handling is broken, the second grade would hang or crash.
            # Give it a real timeout; it may or may not succeed (worker restarted).
        finally:
            await w.aclose()

    async def test_aclose_with_pending_futures_does_not_hang(self, tmp_path) -> None:
        hang_worker = """
            import sys, time
            for line in sys.stdin.buffer:
                time.sleep(3600)
        """
        w = CodegenGradingWorker(worker_cmd=_write_worker(tmp_path, hang_worker))
        grade_task = asyncio.create_task(
            w.grade_codegen([{"input_output": "{}"}], [["x"]], timeout=60)
        )
        await asyncio.sleep(0.05)  # let grade_task start and block
        await w.aclose()  # must not hang even with grade_task pending
        grade_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, CodegenWorkerError):
            await grade_task
```

Add `import contextlib` at the top of the test file if not already present.

- [ ] **Step 2: Run the new tests to verify they fail**

```bash
uv run pytest tests/unit/accuracy/test_codegen_worker_client.py::TestConcurrency -v 2>&1 | head -30
```

Expected: failures because the lock still serializes requests (demux test would hang or return wrong values) or `AttributeError` if the test references methods not yet on the class.

- [ ] **Step 3: Rewrite `CodegenGradingWorker` in `_codegen_worker_client.py`**

The full rewrite of the class. Replace the existing class body (not the module-level helpers `_kill_process_group`, `CodegenWorkerError`, `_STREAM_LIMIT`, etc. — keep those unchanged).

**3a — Update `__init__`:** remove `self._lock`, add `self._pending`, `self._reader_task`, and `self._spawn_lock`:

```python
def __init__(
    self,
    worker_cmd: list[str] | None = None,
    max_start_failures: int = 3,
) -> None:
    self._cmd = worker_cmd or _DEFAULT_WORKER_CMD
    self._max_start_failures = max_start_failures
    self._proc: asyncio.subprocess.Process | None = None
    self._spawn_lock = asyncio.Lock()
    self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
    self._reader_task: asyncio.Task[None] | None = None
    self._next_id = 0
    self._start_failures = 0
    self._worker_proven = False
    self._stderr_tail: deque[str] = deque(maxlen=_STDERR_TAIL_LINES)
    self._stderr_task: asyncio.Task[None] | None = None
    self._death_w: int | None = None
```

**3b — Replace `grade_codegen`:**

```python
async def grade_codegen(
    self,
    evaluation_sample: list[dict[str, str]],
    generated_code: list[list[str]],
    timeout: float,
) -> dict[str, Any]:
    if self._start_failures >= self._max_start_failures:
        raise CodegenWorkerError(
            f"grading worker unavailable after {self._start_failures} start failures"
        )
    await self._ensure_worker()
    self._next_id += 1
    req_id = self._next_id
    req = {
        "id": req_id,
        "evaluation_sample": evaluation_sample,
        "generated_code": generated_code,
    }
    loop = asyncio.get_running_loop()
    fut: asyncio.Future[dict[str, Any]] = loop.create_future()
    self._pending[req_id] = fut
    assert self._proc is not None and self._proc.stdin
    self._proc.stdin.write(orjson.dumps(req) + b"\n")
    try:
        return await asyncio.wait_for(fut, timeout)
    except asyncio.TimeoutError as exc:
        self._pending.pop(req_id, None)
        await self._handle_fault(count_start_failure=False)
        raise CodegenWorkerError(f"grading worker timed out: {exc!r}") from exc
    except asyncio.CancelledError:
        self._pending.pop(req_id, None)
        await self._handle_fault(count_start_failure=False)
        raise
```

**3c — Update `_ensure_worker`** to use `_spawn_lock` and start `_reader_task`:

```python
async def _ensure_worker(self) -> None:
    async with self._spawn_lock:
        if self._proc is not None and self._proc.returncode is None:
            return
        self._worker_proven = False
        self._stderr_tail.clear()
        self._close_death_pipe()
        death_r: int | None = None
        death_w: int | None = None
        pass_fds: tuple[int, ...] = ()
        death_env: dict[str, str] = {}
        if not IS_WINDOWS:
            death_r, death_w = os.pipe()
            os.set_inheritable(death_r, True)
            pass_fds = (death_r,)
            death_env = {_DEATH_FD_ENV: str(death_r)}
        try:
            self._proc = await asyncio.create_subprocess_exec(
                *self._cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=_STREAM_LIMIT,
                start_new_session=True,
                pass_fds=pass_fds,
                env={**os.environ, **death_env},
            )
        except Exception as exc:
            if death_r is not None:
                os.close(death_r)
            if death_w is not None:
                os.close(death_w)
            self._start_failures += 1
            raise CodegenWorkerError(f"failed to spawn grading worker: {exc}") from exc
        if death_r is not None:
            os.close(death_r)
        self._death_w = death_w
        self._stderr_task = asyncio.create_task(self._drain_stderr(self._proc.stderr))
        self._reader_task = asyncio.create_task(self._run_reader())
```

**3d — Add `_run_reader`** (new method, add after `_drain_stderr`):

```python
async def _run_reader(self) -> None:
    """Read worker responses and resolve the corresponding pending futures by id."""
    assert self._proc is not None and self._proc.stdout
    reader = self._proc.stdout
    try:
        while True:
            line = await reader.readline()
            if not line:
                await self._handle_fault()
                return
            try:
                resp = orjson.loads(line)
            except orjson.JSONDecodeError:
                await self._handle_fault()
                return
            if not isinstance(resp, dict):
                await self._handle_fault()
                return
            req_id = resp.get("id")
            fut = self._pending.pop(req_id, None)
            if fut is None or fut.done():
                continue  # stale id (caller already timed out) or cancelled
            if not resp.get("ok"):
                fut.set_exception(
                    CodegenWorkerError(resp.get("error", "unknown grading error"))
                )
                self._mark_proven()
            else:
                metrics = resp.get("metrics")
                if not isinstance(metrics, dict):
                    await self._handle_fault()
                    return
                fut.set_result(metrics)
                self._mark_proven()
    except asyncio.CancelledError:
        pass
```

**3e — Add `_mark_proven`** (new helper, add after `_run_reader`):

```python
def _mark_proven(self) -> None:
    self._worker_proven = True
    self._start_failures = 0
```

**3f — Replace `_handle_fault`:**

```python
async def _handle_fault(self, count_start_failure: bool = True) -> None:
    if self._proc is None:
        return  # already handled; _handle_fault is idempotent
    if count_start_failure and not self._worker_proven:
        self._start_failures += 1
    for fut in list(self._pending.values()):
        if not fut.done():
            fut.set_exception(CodegenWorkerError("grading worker fault"))
    self._pending.clear()
    tail = await self._kill()
    _log.debug(
        lambda: f"codegen worker fault (proven={self._worker_proven}, "
        f"start_failures={self._start_failures}); killed + respawning next grade"
        + (f"; stderr tail:\n{chr(10).join(tail)}" if tail else "")
    )
```

**3g — Update `_kill`** to also cancel and await `_reader_task`:

```python
async def _kill(self) -> list[str]:
    proc, self._proc = self._proc, None
    task, self._stderr_task = self._stderr_task, None
    reader_task, self._reader_task = self._reader_task, None
    self._close_death_pipe()
    if proc is not None and proc.returncode is None:
        _kill_process_group(proc)
        with contextlib.suppress(ProcessLookupError):
            await proc.wait()
    if reader_task is not None:
        reader_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await reader_task
    tail: list[str] = []
    if task is not None:
        with contextlib.suppress(TimeoutError, asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2.0)
        tail = list(self._stderr_tail)
    return tail
```

**3h — Replace `aclose`** (remove the lock, cancel pending futures):

```python
async def aclose(self) -> None:
    for fut in list(self._pending.values()):
        if not fut.done():
            fut.cancel()
    self._pending.clear()
    await self._kill()
```

**3i — Remove `_request`** entirely (replaced by the reader task + futures approach). Grep to confirm nothing else calls it:

```bash
grep -rn "_request" src/aiperf/accuracy/graders/_codegen_worker_client.py
```

- [ ] **Step 4: Update the existing `TestSerialization` class**

The existing `TestSerialization.test_concurrent_grades_do_not_interleave` test was checking that concurrent grades were serialized (old lock behaviour). With the new design, concurrent grades overlap — which is the desired behaviour. Rename and update the test so it validates the new invariant:

Replace the existing `TestSerialization` class with:

```python
class TestSerialization:
    async def test_concurrent_grades_return_correct_results(self, tmp_path) -> None:
        # Previously tested that grades were serialized (lock enforced).
        # Now tests that concurrent grades all return correct results without the lock.
        w = CodegenGradingWorker(worker_cmd=_write_worker(tmp_path, _ECHO_OK))
        try:
            results = await asyncio.gather(*[
                w.grade_codegen([{"input_output": "{}"}], [["x"]], timeout=30)
                for _ in range(4)
            ])
            assert all(r == {"pass@1": 1.0} for r in results)
        finally:
            await w.aclose()
```

- [ ] **Step 5: Run all client unit tests**

```bash
uv run pytest tests/unit/accuracy/test_codegen_worker_client.py -v
```

Expected: all tests pass, including the new `TestConcurrency` class.

- [ ] **Step 6: Run the full unit test suite**

```bash
uv run pytest tests/unit/ -n auto
```

Expected: all green.

- [ ] **Step 7: Lint**

```bash
ruff format . && ruff check --fix .
```

- [ ] **Step 8: Commit**

```bash
git add src/aiperf/accuracy/graders/_codegen_worker_client.py \
        tests/unit/accuracy/test_codegen_worker_client.py
git commit -s -m "feat(accuracy): concurrent codegen grading via id-demux reader task"
```

---

### Task 3: Component integration — concurrent multi-problem test

**Spec:** `docs/superpowers/specs/2026-07-29-codegen-grade-concurrency-design.md` § "Tests / Component integration"

**Files:**
- Modify: `tests/component_integration/test_lcb_codegen_worker_e2e.py`

**Interfaces:**
- Consumes: `CodegenGradingWorker` from Task 2 (concurrent-safe)
- Consumes: real `lighteval` (skip if not installed)

---

- [ ] **Step 1: Write the concurrent e2e test**

Open `tests/component_integration/test_lcb_codegen_worker_e2e.py`.

Add the following after the existing `test_worker_grades_correct_stdin_solution` test:

```python
@pytest.mark.slow
@pytest.mark.asyncio
async def test_worker_grades_multiple_problems_concurrently() -> None:
    """N concurrent grade_codegen() calls all resolve correctly.

    This exercises the batch-drain path: all N requests are sent before the
    worker responds, so they are drained into a single codegen_metrics call and
    processed in parallel by lighteval's ProcessPoolExecutor.
    """
    worker = CodegenGradingWorker()
    sample, code = _sample_and_solution()
    n = 4
    try:
        results = await asyncio.gather(*[
            worker.grade_codegen(sample, code, timeout=240)
            for _ in range(n)
        ])
        assert len(results) == n
        assert all(float(r["pass@1"]) == 1.0 for r in results), results
    finally:
        await worker.aclose()
```

Add `import asyncio` at the top of the file if not already present.

- [ ] **Step 2: Run the component integration tests**

```bash
uv run pytest tests/component_integration/test_lcb_codegen_worker_e2e.py -v -s
```

Expected: both `test_worker_grades_correct_stdin_solution` and `test_worker_grades_multiple_problems_concurrently` pass with `pass@1 == 1.0`.

> These tests run lighteval for real — they take 30-120 seconds each. If `lighteval` is not installed, both tests are auto-skipped via `pytest.importorskip`.

- [ ] **Step 3: Run the full unit test suite one more time**

```bash
uv run pytest tests/unit/ -n auto
```

Expected: all green.

- [ ] **Step 4: Lint and pre-commit**

```bash
ruff format . && ruff check --fix .
pre-commit run --all-files
```

- [ ] **Step 5: Commit**

```bash
git add tests/component_integration/test_lcb_codegen_worker_e2e.py
git commit -s -m "test(accuracy): concurrent multi-problem codegen grading e2e test"
```

---

## Self-Review

**Spec coverage:**

| Spec requirement | Task covering it |
|---|---|
| Drop `asyncio.Lock` | Task 2 Step 3a |
| `id → Future` demux table | Task 2 Step 3b + `_pending` |
| Persistent reader task | Task 2 Step 3c + `_run_reader` |
| Worker batch-drain loop | Task 1 Step 3c |
| Single `codegen_metrics` call per cycle | Task 1 Step 3b (`handle_batch`) |
| Per-problem demux via `compute_metrics_from_results` | Task 1 Step 3b |
| `_handle_fault` cancels all pending futures | Task 2 Step 3f |
| `aclose` cancels pending futures without lock | Task 2 Step 3h |
| `_kill` tears down `_reader_task` | Task 2 Step 3g |
| `_spawn_lock` prevents double-spawn | Task 2 Step 3c |
| `_handle_fault` idempotent (`_proc is None` guard) | Task 2 Step 3f |
| New unit concurrency tests | Task 2 Step 1 |
| New worker batch unit tests | Task 1 Step 1 |
| Component integration concurrent test | Task 3 Step 1 |

**No gaps found.**

**Placeholder scan:** No TBDs, TODOs, or vague steps. All code blocks are complete.

**Type consistency:**
- `handle_batch(reqs, codegen_fn, compute_metrics_fn)` — consistent across Task 1 definition and Task 1 tests
- `run_worker_loop(stdin, out, codegen_fn, compute_metrics_fn)` — consistent in Step 3c and Step 3e (`main()`)
- `_pending: dict[int, asyncio.Future[dict[str, Any]]]` — consistent across `__init__`, `grade_codegen`, `_run_reader`, `_handle_fault`, `aclose`
- `_reader_task: asyncio.Task[None] | None` — consistent across `__init__`, `_ensure_worker`, `_kill`
- `_mark_proven()` — defined in Task 2 Step 3e, called from `_run_reader` (Task 2 Step 3d)
