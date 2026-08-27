# Contest ledger

> Do not edit — generated from the contest tables after each round.

- **kind:** diff_off
- **status:** running
- **round:** 1 / 8
- **artifact:** `.superpowers/sdd/pr-1325-address-review/review-fixes.diff`
- **low-friction:** yes — zero retractions and zero contested objections; unanimity is as consistent with correlated bias as with correctness, so this exchange is flagged UNVALIDATED rather than clean
- **persuasion-override rate (CW-POR):** 0.00 — the fraction of author retractions that answered an UNPROVEN objection; a high value means the author yielded to confident assertion rather than to evidence

## Seats

- **author** — `opus` (family `claude-trace`, lens —)
- **skeptic** — `gpt-5.6-sol` (family `claude-codex`, lens correctness — especially concurrency/ordering hazards, silent behavior changes, and whether each fix actually resolves the reviewer concern it claims to)

## Objections

## O1 — `src/aiperf/api/routers/results.py:85-91` lets `get_results` return `BenchmarkStatus.COMPLETE` whenever `component._final_results is not None`, even while `component._benchmark_complete` is `False`. This reverses the documented safety gate at `src/aiperf/api/routers/results.py:73-76`: `BENCHMARK_COMPLETE` is the signal that controller-side artifact export is finished. `ProcessAllResultsMessage` can therefore make `/api/results` terminal before artifacts are safe to fetch.
severity: high   raised: r1   status: standing
proven: yes
evidence:
  Executed:
  uv run python (construct component with `_final_results=make_process_records_result(...)`, `_benchmark_complete=False`, then `await get_results(component)`).
  Output:
  benchmark_complete= False
  returned_status= complete
  returned_results= True
  
  Exact code: `src/aiperf/api/routers/results.py:85-91` checks `_final_results` first and returns `BenchmarkStatus.COMPLETE`; only `src/aiperf/api/routers/results.py:93-94` checks `_benchmark_complete`. Exact handler comments at `src/aiperf/api/routers/results.py:73-76` say: `Only now do we report "complete" to external consumers so they can safely fetch all result files.`

## O2 — `ResultJoinCoordinator.evict_service` returns `was_required` at `src/aiperf/controller/result_join_coordinator.py:85`, while recording degradation only for `was_pending` at `src/aiperf/controller/result_join_coordinator.py:78-84`. `SystemController._on_service_reaped` branches directly on that return at `src/aiperf/controller/system_controller.py:617` and appends `ProducerReaped` at `src/aiperf/controller/system_controller.py:622-635`. A producer reaped after `complete_domain` is therefore still reported as missing/fatal even though `coord.evicted == {}`.
severity: high   raised: r1   status: standing
proven: yes
evidence:
  Executed:
  coord.register('telemetry', 'tm-1'); coord.complete_domain('telemetry'); returned = coord.evict_service('tm-1', 'pod OOMKilled')
  Output:
  evict_service_returned= True
  evicted= {}
  SystemController branch condition= True
  
  The exact caller at `src/aiperf/controller/system_controller.py:617-620` logs `results for this producer will be missing from the run`, and `src/aiperf/controller/system_controller.py:622-635` appends the fatal `ProducerReaped` error, despite the new completed-producer contract.

## O3 — `_compute_best_trials` filters only `_primary(h)` at `src/aiperf/exporters/search_history.py:154-170`; `_dominates` skips any missing/non-finite secondary objective at `src/aiperf/exporters/search_history.py:229-241`. A trial with `[100.0, NaN]` is emitted as the sole Pareto-best trial over a fully scored `[50.0, 8.0]`, serialized as `[100.0, null]`. The NaN fix does not resolve multi-objective non-primary NaN contamination.
severity: high   raised: r1   status: standing
proven: yes
evidence:
  Executed `write_search_history` with `_two_obj_cfg()` (maximize throughput, minimize latency) and trials `objective_values=[100.0, float('nan')]` and `[50.0, 8.0]`.
  Output:
  best_trial_indices= [0]
  best_trial_objectives= [[100.0, None]]
  
  Exact cause: `src/aiperf/exporters/search_history.py:170` applies `is_finite_value(_primary(h))` only; `src/aiperf/exporters/search_history.py:233-234` says `if candidate_value is None or other_value is None: continue`.

## O4 — `_Environment.validate_worker_stale_time_vs_heartbeat` at `src/aiperf/common/environment.py:2071-2098` treats `SERVICE.HEARTBEAT_MISSED_THRESHOLD` as part of the worker heartbeat cadence, but heartbeats are emitted solely at `Environment.SERVICE.HEARTBEAT_INTERVAL` by `src/aiperf/common/base_component_service.py:66-75`; `HEARTBEAT_MISSED_THRESHOLD` is only the controller watchdog cutoff at `src/aiperf/controller/base_service_manager.py:235-237`. The import-time singleton rejects safe configurations based on an unrelated watchdog policy.
severity: high   raised: r1   status: standing
proven: yes
evidence:
  Executed:
  AIPERF_WORKER_STALE_TIME=10 AIPERF_SERVICE_HEARTBEAT_INTERVAL=5 AIPERF_SERVICE_HEARTBEAT_MISSED_THRESHOLD=100 uv run python -c 'from aiperf.common.environment import Environment'
  Output: import fails with `ValidationError`: `STALE_TIME * 3 = 30.0` must be greater than `5.0 * 100 = 500.0`.
  
  Yet exact emitter `src/aiperf/common/base_component_service.py:66` is `@background_task(interval=Environment.SERVICE.HEARTBEAT_INTERVAL, immediate=False)`; `src/aiperf/common/environment.py:1385-1388` describes `HEARTBEAT_MISSED_THRESHOLD` as watchdog suspicion/detection, not emission cadence. A heartbeat every 5s is safely inside the router's 30s cutoff regardless of watchdog threshold 100.

## O5 — `SystemController.failure_shutdown_timeout` returns `None` at `src/aiperf/controller/system_controller.py:105-111`, so `_fail` executes bare `await self.stop()` at `src/aiperf/common/mixins/aiperf_lifecycle_mixin.py:323-325`. `_stop_system_controller` has multiple unbounded awaits beginning with `await self.ui.stop()` at `src/aiperf/controller/system_controller.py:1370`. Any wedged hook now prevents `_set_state(LifecycleState.FAILED)` and the claimed `os._exit()` path forever.
severity: high   raised: r1   status: standing
proven: yes
evidence:
  Executed `SystemController._stop_system_controller(fake)` with `_set_system_state=AsyncMock()` and `ui.stop` waiting forever, under an external 0.02s probe.
  Output:
  TIMED_OUT at await self.ui.stop()
  SystemController.failure_shutdown_timeout= None
  
  Exact contradiction: `src/aiperf/common/mixins/aiperf_lifecycle_mixin.py:317-322` states the bound prevents `a silent zombie that keeps its container alive`; the override removes that bound from the one component whose final `os._exit()` is reachable only after every await at `src/aiperf/controller/system_controller.py:1364-1433` completes.

## O6 — The `_write_error` gate at `src/aiperf/common/mixins/buffered_jsonl_writer_mixin.py:153` does not bound flush retries under a burst: `buffered_write` schedules background flushes before the first `_flush_buffer` task can set `_write_error` at `src/aiperf/common/mixins/buffered_jsonl_writer_mixin.py:246-250`. The added test's `await asyncio.sleep(0)` at `tests/unit/common/mixins/test_buffered_jsonl_writer_mixin.py:1637-1641` artificially serializes this race. 300 immediate appends still schedule 30 doomed flushes, so the claimed write-amplification fix is incomplete.
severity: medium   raised: r1   status: standing
proven: yes
evidence:
  Executed the added test's setup with `batch_size=10`, `300` `await writer.buffered_write(...)` calls, but without its per-record `await asyncio.sleep(0)`, then gathered tracked flush tasks.
  Output:
  flush_calls= 30
  scheduled_tasks= 30
  buffered_records= 300
  
  Exact gate: `src/aiperf/common/mixins/buffered_jsonl_writer_mixin.py:153` is `if self._write_error is None and len(self._buffer) >= self._batch_size:`. Exact hidden yield: `tests/unit/common/mixins/test_buffered_jsonl_writer_mixin.py:1637-1641` explicitly yields so `_write_error` becomes visible before the next append.

## O7 — `ServerMetricsAccumulator._resolve_sample_phase_index` at `src/aiperf/server_metrics/accumulator.py:229-241` keys synthetic instances by last arrival contiguity, but scrapes intentionally overlap at `src/aiperf/common/mixins/base_metrics_collector_mixin.py:505-516` and retain the phase snapshot from scrape start at `src/aiperf/server_metrics/manager.py:801-810`. A slow record from the first warmup arriving after profiling data is assigned a fresh synthetic index and then pooled with the next warmup instance, corrupting phase windows.
severity: high   raised: r1   status: standing
proven: yes
evidence:
  Executed accumulator input in arrival order: warmup timestamp 10, profiling timestamp 30, delayed warmup timestamp 20, next warmup timestamp 40. Output:
  warmup_count= 2
  warmup_windows= [(None, 10, 10), (None, 20, 40)]
  
  The delayed timestamp-20 sample belongs with timestamp 10, but `src/aiperf/server_metrics/accumulator.py:233-240` minted a new index after the intervening profiling signature, causing it to pool with timestamp 40. Production permits this order because `src/aiperf/common/mixins/base_metrics_collector_mixin.py:505-510` explicitly allows multiple scrapes in flight and `src/aiperf/server_metrics/manager.py:806-810` snapshots `_active_phase` for each scrape.

## O8 — `ExporterManager._export_one_phase` eagerly creates all four coroutine objects in `artifact_writers` at `src/aiperf/exporters/exporter_manager.py:314-361`, then catches only `Exception` around each await at `src/aiperf/exporters/exporter_manager.py:363-366`. If any writer is cancelled, `asyncio.CancelledError` escapes and every later already-created coroutine is abandoned unawaited, producing runtime warnings and skipping cleanup/work.
severity: medium   raised: r1   status: standing
proven: yes
evidence:
  Exact deterministic control flow: `src/aiperf/exporters/exporter_manager.py:314-361` calls `_write_phase_export(...)` and `_write_phase_observability_export(...)` four times to instantiate coroutine objects before iteration. `src/aiperf/exporters/exporter_manager.py:365-366` catches `Exception`; `asyncio.CancelledError` derives from `BaseException`, so cancellation exits the loop and leaves later coroutine objects unawaited. This is not hypothetical input: lifecycle/task cancellation is a normal shutdown path in this async service.

## Unresolved risks

- O1 — `src/aiperf/api/routers/results.py:85-91` lets `get_results` return `BenchmarkStatus.COMPLETE` whenever `component._final_results is not None`, even while `component._benchmark_complete` is `False`. This reverses the documented safety gate at `src/aiperf/api/routers/results.py:73-76`: `BENCHMARK_COMPLETE` is the signal that controller-side artifact export is finished. `ProcessAllResultsMessage` can therefore make `/api/results` terminal before artifacts are safe to fetch.
- O2 — `ResultJoinCoordinator.evict_service` returns `was_required` at `src/aiperf/controller/result_join_coordinator.py:85`, while recording degradation only for `was_pending` at `src/aiperf/controller/result_join_coordinator.py:78-84`. `SystemController._on_service_reaped` branches directly on that return at `src/aiperf/controller/system_controller.py:617` and appends `ProducerReaped` at `src/aiperf/controller/system_controller.py:622-635`. A producer reaped after `complete_domain` is therefore still reported as missing/fatal even though `coord.evicted == {}`.
- O3 — `_compute_best_trials` filters only `_primary(h)` at `src/aiperf/exporters/search_history.py:154-170`; `_dominates` skips any missing/non-finite secondary objective at `src/aiperf/exporters/search_history.py:229-241`. A trial with `[100.0, NaN]` is emitted as the sole Pareto-best trial over a fully scored `[50.0, 8.0]`, serialized as `[100.0, null]`. The NaN fix does not resolve multi-objective non-primary NaN contamination.
- O4 — `_Environment.validate_worker_stale_time_vs_heartbeat` at `src/aiperf/common/environment.py:2071-2098` treats `SERVICE.HEARTBEAT_MISSED_THRESHOLD` as part of the worker heartbeat cadence, but heartbeats are emitted solely at `Environment.SERVICE.HEARTBEAT_INTERVAL` by `src/aiperf/common/base_component_service.py:66-75`; `HEARTBEAT_MISSED_THRESHOLD` is only the controller watchdog cutoff at `src/aiperf/controller/base_service_manager.py:235-237`. The import-time singleton rejects safe configurations based on an unrelated watchdog policy.
- O5 — `SystemController.failure_shutdown_timeout` returns `None` at `src/aiperf/controller/system_controller.py:105-111`, so `_fail` executes bare `await self.stop()` at `src/aiperf/common/mixins/aiperf_lifecycle_mixin.py:323-325`. `_stop_system_controller` has multiple unbounded awaits beginning with `await self.ui.stop()` at `src/aiperf/controller/system_controller.py:1370`. Any wedged hook now prevents `_set_state(LifecycleState.FAILED)` and the claimed `os._exit()` path forever.
- O6 — The `_write_error` gate at `src/aiperf/common/mixins/buffered_jsonl_writer_mixin.py:153` does not bound flush retries under a burst: `buffered_write` schedules background flushes before the first `_flush_buffer` task can set `_write_error` at `src/aiperf/common/mixins/buffered_jsonl_writer_mixin.py:246-250`. The added test's `await asyncio.sleep(0)` at `tests/unit/common/mixins/test_buffered_jsonl_writer_mixin.py:1637-1641` artificially serializes this race. 300 immediate appends still schedule 30 doomed flushes, so the claimed write-amplification fix is incomplete.
- O7 — `ServerMetricsAccumulator._resolve_sample_phase_index` at `src/aiperf/server_metrics/accumulator.py:229-241` keys synthetic instances by last arrival contiguity, but scrapes intentionally overlap at `src/aiperf/common/mixins/base_metrics_collector_mixin.py:505-516` and retain the phase snapshot from scrape start at `src/aiperf/server_metrics/manager.py:801-810`. A slow record from the first warmup arriving after profiling data is assigned a fresh synthetic index and then pooled with the next warmup instance, corrupting phase windows.
- O8 — `ExporterManager._export_one_phase` eagerly creates all four coroutine objects in `artifact_writers` at `src/aiperf/exporters/exporter_manager.py:314-361`, then catches only `Exception` around each await at `src/aiperf/exporters/exporter_manager.py:363-366`. If any writer is cancelled, `asyncio.CancelledError` escapes and every later already-created coroutine is abandoned unawaited, producing runtime warnings and skipping cleanup/work.
