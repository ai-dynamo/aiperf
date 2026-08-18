# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Round-2 adversarial coverage for operator improvements.

Covers:
    * Regression for the pod-restarts dedup-state memory leak (orphan
      jobset-name key + sweep-owned leak): ``handle_pod_restart`` must
      keep ``_warned_pod_restarts`` keyed exclusively by job_id (the same
      key ``client_cache._close_unlocked`` evicts).
    * Hypothesis property tests: ``_extract_reason``,
      ``_has_completed_condition``, ``handle_jobset_conditions`` survive
      arbitrary inputs without raising.
    * Higher-fanout concurrent-fire stress (50x) for both watch handlers.
    * ``track_handler`` distinguishes ``kopf.PermanentError`` (fatal),
      ``kopf.TemporaryError`` (retry), other exceptions (error).
    * ``COMPLETION_CLAIM_RACES`` counter actually increments at the
      ``try_claim_completion`` race-loss code paths.
    * Negative-path ``observedGeneration``: a failure in a preceding
      sub-call leaves observedGeneration unwritten.
    * Metrics endpoint smoke test: prometheus_client exposition really
      renders our four metric names.
"""

from __future__ import annotations

import asyncio
import socket
import time
import urllib.request
from collections.abc import Iterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from unittest.mock import patch as mock_patch

import kopf
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from prometheus_client import REGISTRY

from aiperf.kubernetes.constants import Annotations
from aiperf.operator.client_cache import (
    _reset_for_testing,
    _warned_pod_restarts,
    job_key,
)
from aiperf.operator.handlers.jobset_terminal import (
    _has_completed_condition,
    handle_jobset_conditions,
)
from aiperf.operator.handlers.pod_restarts import (
    _extract_reason,
    handle_pod_restart,
)
from aiperf.operator.metrics import (
    COMPLETION_CLAIM_RACES,
    HANDLER_DURATION,
    HANDLER_TOTAL,
    track_handler,
)


@pytest.fixture(autouse=True)
def _reset_state() -> Iterator[None]:
    """Reset module-level state between tests so cross-pollution can't hide bugs."""
    _reset_for_testing()
    HANDLER_TOTAL.clear()
    HANDLER_DURATION.clear()
    yield
    _reset_for_testing()
    HANDLER_TOTAL.clear()
    HANDLER_DURATION.clear()


# =============================================================================
# Bug A + B regression: pod-restart dedup state must NOT orphan a jobset-name key
# =============================================================================


class TestPodRestartDedupStateLeakRegression:
    """Regression tests for the pre-claim dedup-state leak.

    Before round-2 the pre-claim happened BEFORE the AIPerfJob lookup,
    keyed by jobset-name. After a successful lookup, state was migrated
    to the canonical job-id key — but the original jobset-name entry
    was never popped (Bug A) — and for sweep-owned JobSets (lookup
    returns None), the early-exit ran AFTER pre-claim had already
    populated the dict (Bug B). Either path leaked a set per CR for the
    operator-process lifetime.

    The fix moves the lookup BEFORE any pre-claim, so dedup state only
    materializes under the canonical job_id key that
    ``client_cache._close_unlocked`` already evicts.
    """

    @pytest.mark.asyncio
    async def test_no_jobset_name_key_after_successful_lookup(self) -> None:
        """After a successful lookup, no entry should exist under
        ``job_key(ns, jobset_name)`` — only under ``job_key(ns, job_id)``."""
        meta = {
            "name": "p",
            "labels": {"jobset.sigs.k8s.io/jobset-name": "aiperf-bench"},
        }
        body = {
            "metadata": {"name": "bench"},
            "status": {"jobId": "real-job-id"},
        }
        with (
            mock_patch("aiperf.operator.handlers.pod_restarts.events.pod_restarts"),
            mock_patch(
                "aiperf.operator.handlers.pod_restarts._lookup_aiperfjob_body",
                return_value=body,
            ),
        ):
            await handle_pod_restart(
                old=[],
                new=[{"name": "c", "restartCount": 5}],
                body={"metadata": meta},
                meta=meta,
                namespace="ns",
                name="p",
                threshold=3,
            )
        assert job_key("ns", "aiperf-bench") not in _warned_pod_restarts, (
            "Bug A regression: jobset-name-keyed dedup entry leaked "
            "after migration to canonical job-id key"
        )
        assert job_key("ns", "real-job-id") in _warned_pod_restarts

    @pytest.mark.asyncio
    async def test_no_dedup_entry_when_lookup_returns_none(self) -> None:
        """Bug B: sweep-owned JobSets (lookup→None) must NOT leave any
        pre-claim entry behind.

        Pre-fix path: pre-claim populated ``_warned_pod_restarts[ns/jobset-name]``
        BEFORE the lookup; the lookup returned None for sweep-owned JobSets
        and the handler returned, leaving the entry forever (since
        ``client_cache._close_unlocked`` evicts by job_id, not jobset name,
        and sweep-owned JobSets have no AIPerfJob).
        """
        meta = {
            "name": "ctrl-0",
            "labels": {"jobset.sigs.k8s.io/jobset-name": "aiperf-some-sweep"},
        }
        with (
            mock_patch("aiperf.operator.handlers.pod_restarts.events.pod_restarts"),
            mock_patch(
                "aiperf.operator.handlers.pod_restarts._lookup_aiperfjob_body",
                return_value=None,
            ),
        ):
            await handle_pod_restart(
                old=[],
                new=[{"name": "c", "restartCount": 5}],
                body={"metadata": meta},
                meta=meta,
                namespace="ns",
                name="ctrl-0",
                threshold=3,
            )
        assert _warned_pod_restarts == {}, (
            "Bug B regression: sweep-owned JobSet left dedup state behind; "
            "it would persist for the operator-process lifetime since "
            "sweep-owned JobSets have no matching AIPerfJob to evict via "
            "client_cache._close_unlocked"
        )

    @pytest.mark.asyncio
    async def test_below_threshold_does_not_create_dedup_entry(self) -> None:
        """Below-threshold calls must NOT pre-create a dict entry — that
        was the cheapest leak path (every monitor tick on a healthy pod
        added a key)."""
        meta = {
            "name": "ctrl-0",
            "labels": {"jobset.sigs.k8s.io/jobset-name": "aiperf-bench"},
        }
        with mock_patch(
            "aiperf.operator.handlers.pod_restarts._lookup_aiperfjob_body",
            return_value={"metadata": {"name": "bench"}, "status": {}},
        ):
            await handle_pod_restart(
                old=[],
                new=[{"name": "c", "restartCount": 0}],
                body={"metadata": meta},
                meta=meta,
                namespace="ns",
                name="ctrl-0",
                threshold=3,
            )
        assert _warned_pod_restarts == {}


# =============================================================================
# Hypothesis property tests: defensive parsing
# =============================================================================


# Strategy that emits arbitrary JSON-like values, including the malformed
# shapes a misbehaving apiserver / kopf delivery could hand us.
_arbitrary_value: st.SearchStrategy[Any] = st.recursive(
    st.one_of(
        st.none(),
        st.booleans(),
        st.integers(),
        st.text(),
        st.floats(allow_nan=True, allow_infinity=True),
    ),
    lambda children: st.one_of(
        st.lists(children, max_size=4),
        st.dictionaries(st.text(max_size=8), children, max_size=4),
    ),
    max_leaves=5,
)


class TestHypothesisPropertyTests:
    """Pure-function defensive parsing must accept arbitrary kopf payloads.

    ``settings(deadline=None)`` because we don't need timing guarantees;
    the body of the test calls a sync function with a value, no async.
    ``suppress_health_check=[HealthCheck.too_slow]`` defends against
    xdist scheduling jitter on slow shards.
    """

    @given(cs=st.dictionaries(st.text(max_size=8), _arbitrary_value, max_size=4))
    @settings(
        deadline=None,
        suppress_health_check=[
            HealthCheck.too_slow,
            HealthCheck.function_scoped_fixture,
        ],
        max_examples=200,
    )
    def test_extract_reason_never_raises(self, cs: dict[str, Any]) -> None:
        """``_extract_reason`` is fed arbitrary nested junk; must not raise."""
        # The function is typed dict[str,Any] but real-world inputs include
        # arbitrary nested shapes from kopf field-watch deliveries. We only
        # require: doesn't raise and returns a string.
        result = _extract_reason(cs)
        assert isinstance(result, str)

    @given(conditions=st.one_of(st.none(), st.lists(_arbitrary_value, max_size=6)))
    @settings(
        deadline=None,
        suppress_health_check=[
            HealthCheck.too_slow,
            HealthCheck.function_scoped_fixture,
        ],
        max_examples=200,
    )
    def test_has_completed_condition_never_raises(self, conditions: Any) -> None:
        """``_has_completed_condition`` must accept arbitrary list-of-anything
        (the round-1 fix added ``isinstance(cond, dict)`` to skip non-dicts)."""
        result = _has_completed_condition(conditions)
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    @given(
        old=st.one_of(st.none(), st.lists(_arbitrary_value, max_size=4)),
        new=st.one_of(st.none(), st.lists(_arbitrary_value, max_size=4)),
    )
    @settings(
        deadline=None,
        suppress_health_check=[
            HealthCheck.too_slow,
            HealthCheck.function_scoped_fixture,
        ],
        max_examples=80,
    )
    async def test_handle_jobset_conditions_never_raises(
        self, old: Any, new: Any
    ) -> None:
        """The full handler must not raise for any (old, new) shapes.

        The lookup + setter are mocked to no-op so we isolate the handler
        body's defensive parsing. If ``new`` does not contain a true
        Completed/True dict, no apiserver path runs; if it does, the
        mocked lookup returns None and the early-exit triggers.
        """
        with (
            mock_patch(
                "aiperf.operator.handlers.jobset_terminal._lookup_aiperfjob_body",
                new=AsyncMock(return_value=None),
            ),
            mock_patch(
                "aiperf.operator.handlers.jobset_terminal._set_benchmark_complete_annotation",
                new=AsyncMock(),
            ),
        ):
            await handle_jobset_conditions(
                old=old, new=new, namespace="ns", jobset_name="aiperf-x"
            )


# =============================================================================
# Higher-fanout concurrent-stress for watch handlers
# =============================================================================


class TestConcurrentFanoutStress:
    """Pinning the dedup invariant under 50-way concurrent fanout.

    Round-1 used 2 fires; round-2 cranks to 50 to defend against any
    future change in pre-claim ordering that introduces sub-step
    interleaving (e.g. someone adds an ``await`` between the membership
    check and the add).
    """

    @pytest.mark.asyncio
    async def test_pod_restart_50_concurrent_emit_one_event(self) -> None:
        """50 concurrent fires for the same (pod, count) → exactly 1 event."""
        meta = {
            "name": "p",
            "labels": {"jobset.sigs.k8s.io/jobset-name": "aiperf-x"},
        }
        body = {"metadata": {"name": "x"}, "status": {"jobId": "j"}}

        # Force every concurrent caller to yield at the lookup so the
        # event-loop interleaves them rather than running serially.
        proceed = asyncio.Event()

        async def slow_lookup(*_args: Any, **_kwargs: Any) -> Any:
            await proceed.wait()
            return body

        with (
            mock_patch(
                "aiperf.operator.handlers.pod_restarts.events.pod_restarts"
            ) as evt,
            mock_patch(
                "aiperf.operator.handlers.pod_restarts._lookup_aiperfjob_body",
                side_effect=slow_lookup,
            ),
        ):
            tasks = [
                asyncio.create_task(
                    handle_pod_restart(
                        old=[],
                        new=[{"name": "c", "restartCount": 5}],
                        body={"metadata": meta},
                        meta=meta,
                        namespace="ns",
                        name="p",
                        threshold=3,
                    )
                )
                for _ in range(50)
            ]
            # Ensure every task has entered slow_lookup before releasing.
            for _ in range(20):
                await asyncio.sleep(0)
            proceed.set()
            await asyncio.gather(*tasks)

        assert evt.call_count == 1, (
            "50-way concurrent fanout for the same (pod, count) MUST emit "
            "exactly one event; pre-claim atomicity is the only guard."
        )

    @pytest.mark.asyncio
    async def test_jobset_terminal_50_concurrent_handle_idempotent(self) -> None:
        """Concurrent Completed events never substitute for controller proof."""
        from aiperf.kubernetes.constants import AIPerfLabels
        from aiperf.kubernetes.cr_refs import AIPERF_JOB_API_VERSION

        new = [{"type": "Completed", "status": "True"}]
        body = {
            "metadata": {
                "name": "ajob",
                "uid": "uid-ajob",
                "resourceVersion": "42",
                "annotations": {},
            },
            "status": {},
        }
        trusted_jobset_body = {
            "metadata": {
                "name": "aiperf-ajob",
                "labels": {
                    AIPerfLabels.APP_KEY: AIPerfLabels.APP_VALUE,
                    AIPerfLabels.JOB_ID: "ajob",
                },
                "ownerReferences": [
                    {
                        "apiVersion": AIPERF_JOB_API_VERSION,
                        "kind": "AIPerfJob",
                        "name": "ajob",
                        "uid": "uid-ajob",
                        "controller": True,
                    }
                ],
            }
        }
        with (
            mock_patch(
                "aiperf.operator.handlers.jobset_terminal._lookup_aiperfjob_body",
                new=AsyncMock(return_value=body),
            ),
            mock_patch(
                "aiperf.operator.handlers.jobset_terminal._set_benchmark_complete_annotation",
                new=AsyncMock(),
            ) as setter,
        ):
            await asyncio.gather(
                *(
                    handle_jobset_conditions(
                        old=[],
                        new=new,
                        namespace="ns",
                        jobset_name="aiperf-ajob",
                        jobset_body=trusted_jobset_body,
                    )
                    for _ in range(50)
                )
            )
        setter.assert_not_awaited()


# =============================================================================
# Bug D: kopf control-flow exceptions distinguished in handler outcome metric
# =============================================================================


def _outcome_counter(handler: str, outcome: str) -> float:
    total = 0.0
    for m in REGISTRY.collect():
        for s in m.samples:
            if (
                s.name == "aiperf_operator_handler_total"
                and s.labels.get("handler") == handler
                and s.labels.get("outcome") == outcome
            ):
                total += s.value
    return total


class TestKopfControlFlowOutcomeClassification:
    """``track_handler`` must classify kopf control-flow exceptions distinctly.

    kopf treats ``PermanentError`` as "stop retrying" and ``TemporaryError``
    as "retry after delay" — these are CONTROL FLOW, not failures. Lumping
    them under ``outcome=error`` makes the metric useless for alerting on
    "stuck CR" because every retry of a flaky handler shows up as error.
    """

    @pytest.mark.asyncio
    async def test_permanent_error_classified_as_fatal(self) -> None:
        @track_handler("perm_handler")
        async def fake_handler() -> None:
            raise kopf.PermanentError("config invalid")

        with pytest.raises(kopf.PermanentError):
            await fake_handler()

        assert _outcome_counter("perm_handler", "fatal") >= 1
        assert _outcome_counter("perm_handler", "error") == 0
        assert _outcome_counter("perm_handler", "retry") == 0
        assert _outcome_counter("perm_handler", "success") == 0

    @pytest.mark.asyncio
    async def test_temporary_error_classified_as_retry(self) -> None:
        @track_handler("tmp_handler")
        async def fake_handler() -> None:
            raise kopf.TemporaryError("apiserver flaky", delay=10)

        with pytest.raises(kopf.TemporaryError):
            await fake_handler()

        assert _outcome_counter("tmp_handler", "retry") >= 1
        assert _outcome_counter("tmp_handler", "error") == 0
        assert _outcome_counter("tmp_handler", "fatal") == 0

    @pytest.mark.asyncio
    async def test_runtime_error_still_classified_as_error(self) -> None:
        """Generic exceptions (not kopf control-flow) keep ``outcome=error``."""

        @track_handler("rte_handler")
        async def fake_handler() -> None:
            raise RuntimeError("oops")

        with pytest.raises(RuntimeError):
            await fake_handler()

        assert _outcome_counter("rte_handler", "error") >= 1
        assert _outcome_counter("rte_handler", "retry") == 0
        assert _outcome_counter("rte_handler", "fatal") == 0

    @pytest.mark.asyncio
    async def test_success_path_still_classified_as_success(self) -> None:
        @track_handler("ok_handler")
        async def fake_handler() -> None:
            return None

        await fake_handler()
        assert _outcome_counter("ok_handler", "success") >= 1


# =============================================================================
# Bug C: COMPLETION_CLAIM_RACES counter wired
# =============================================================================


def _race_counter() -> float:
    total = 0.0
    for m in REGISTRY.collect():
        for s in m.samples:
            if s.name == "aiperf_operator_completion_claim_races_total":
                total += s.value
    return total


@asynccontextmanager
async def _fake_k8s_client() -> Any:
    yield MagicMock()


class TestCompletionClaimRacesCounter:
    """Bug C regression: ``COMPLETION_CLAIM_RACES`` is incremented at every
    race-loss code path of ``try_claim_completion`` (annotation already
    present on the body, or apiserver returns 409/422 on the test-and-add
    JSON patch).

    Not incremented on:
      * In-process ``_shutdown_sent`` short-circuit (we already won; this
        is a re-fire of the same handler in the same process, not a race).
      * Successful claim.
      * Unexpected error (returns False but ``claimed is None`` branch).
    """

    @pytest.mark.asyncio
    async def test_annotation_already_present_increments_counter(self) -> None:
        from aiperf.operator.client_cache import try_claim_completion

        body = {
            "metadata": {
                "annotations": {Annotations.COMPLETION_CLAIMED: "2026-01-01T00:00:00Z"},
            },
        }

        # The body annotation no longer short-circuits (it is user-writable);
        # the atomic patch is attempted and rejected with a 422 because the
        # annotation is genuinely already set, which we simulate here. The lost
        # race still increments the counter.
        async def fake_submit(*_a: Any, **_kw: Any) -> bool:
            return False

        before = _race_counter()
        with mock_patch(
            "aiperf.operator.client_cache._submit_claim_patch",
            side_effect=fake_submit,
        ):
            result = await try_claim_completion("ns", "j", body)
        after = _race_counter()

        assert result is False
        assert after - before >= 1, (
            "annotation-present path must increment race counter"
        )

    @pytest.mark.asyncio
    async def test_apiserver_409_increments_counter(self) -> None:
        """Lost-the-test-op race (concurrent peer claimed first) must increment."""
        from aiperf.operator.client_cache import try_claim_completion

        body = {"metadata": {"annotations": {}}}

        async def fake_submit(*_a: Any, **_kw: Any) -> bool:
            return False  # 409 / 422 path

        with mock_patch(
            "aiperf.operator.client_cache._submit_claim_patch",
            side_effect=fake_submit,
        ):
            before = _race_counter()
            result = await try_claim_completion("ns", "j", body)
            after = _race_counter()

        assert result is False
        assert after - before >= 1

    @pytest.mark.asyncio
    async def test_successful_claim_does_not_increment_counter(self) -> None:
        from aiperf.operator.client_cache import try_claim_completion

        body = {"metadata": {"annotations": {}}}

        async def fake_submit(*_a: Any, **_kw: Any) -> bool:
            return True

        async def fake_refresh() -> None:
            return None

        with (
            mock_patch(
                "aiperf.operator.client_cache._submit_claim_patch",
                side_effect=fake_submit,
            ),
            mock_patch(
                "aiperf.operator.client_cache._post_dashboard_refresh",
                side_effect=fake_refresh,
            ),
        ):
            before = _race_counter()
            result = await try_claim_completion("ns", "j", body)
            after = _race_counter()

        assert result is True
        assert after == before, "successful claim must NOT count as a race"

    @pytest.mark.asyncio
    async def test_in_process_shortcircuit_does_not_increment_counter(self) -> None:
        """Re-firing the same handler in the same process after we already
        won must NOT count as a race — this is just a redundant call we
        skip cheaply."""
        from aiperf.operator.client_cache import _shutdown_sent, try_claim_completion

        # Pretend we already claimed in this process.
        _shutdown_sent.add(job_key("ns", "j"))
        body = {"metadata": {"annotations": {}}}

        before = _race_counter()
        result = await try_claim_completion("ns", "j", body)
        after = _race_counter()

        assert result is False
        assert after == before, (
            "in-process short-circuit is not a race-loss; it's the same "
            "handler re-firing after we already won"
        )

    @pytest.mark.asyncio
    async def test_unexpected_error_does_not_increment_counter(self) -> None:
        """``claimed is None`` (transient apiserver error, not a 409/422)
        must NOT count as a race — the call may yet succeed on retry."""
        from aiperf.operator.client_cache import try_claim_completion

        body = {"metadata": {"annotations": {}}}

        async def fake_submit(*_a: Any, **_kw: Any) -> Any:
            return None  # unexpected error path

        with mock_patch(
            "aiperf.operator.client_cache._submit_claim_patch",
            side_effect=fake_submit,
        ):
            before = _race_counter()
            result = await try_claim_completion("ns", "j", body)
            after = _race_counter()

        assert result is False
        assert after == before


# =============================================================================
# observedGeneration negative-path & missing-metadata coverage
# =============================================================================


class TestObservedGenerationCallSiteContract:
    """Pin the "stamp only on success" invariant at every call site.

    All five sites follow the same pattern:

        generation = body.get("metadata", {}).get("generation")
        if generation is not None:
            sb.set_observed_generation(int(generation))

    These tests pin the call-site shape (defensive against missing
    ``metadata`` entirely) and add negative-path tests where the prior
    sub-call raised — the stamp must not have been applied.
    """

    def test_call_site_shape_handles_missing_metadata_top_key(self) -> None:
        """``body`` with no metadata at all → call-site guard prevents
        any write to patch.status.observedGeneration."""
        from aiperf.operator.status import StatusBuilder

        patch = MagicMock()
        patch.status = {}
        sb = StatusBuilder(patch, {})

        # Production call-site shape (verbatim from create.py / monitor.py / lifecycle.py):
        body: dict[str, Any] = {}
        generation = body.get("metadata", {}).get("generation")
        if generation is not None:
            sb.set_observed_generation(int(generation))

        assert "observedGeneration" not in patch.status

    def test_call_site_shape_handles_metadata_without_generation(self) -> None:
        from aiperf.operator.status import StatusBuilder

        patch = MagicMock()
        patch.status = {}
        sb = StatusBuilder(patch, {})

        body = {"metadata": {"name": "j"}}
        generation = body.get("metadata", {}).get("generation")
        if generation is not None:
            sb.set_observed_generation(int(generation))

        assert "observedGeneration" not in patch.status

    @pytest.mark.asyncio
    async def test_monitor_progress_does_not_stamp_when_monitor_tick_raises(
        self,
    ) -> None:
        """In ``monitor_progress`` the stamp lives AFTER ``_monitor_tick``
        returns successfully. If ``_monitor_tick`` raises, the stamp must
        NOT have been applied — the except branches don't call it.

        REGRESSION GUARD: if someone moves the stamp earlier (say, into a
        ``finally`` block) it would lie about acknowledged generation
        when reconcile actually failed.

        After the timer-rearm fix, transient API errors raise kopf.TemporaryError
        instead of swallowing; the guard still verifies that observedGeneration
        was not stamped before the raise.
        """
        from kubernetes_asyncio.client.exceptions import ApiException

        from aiperf.operator.handlers.monitor import monitor_progress

        patch = kopf.Patch()
        body = {"metadata": {"generation": 7}}
        status = {"phase": "Running", "jobSetName": "aiperf-x", "jobId": "x"}

        async def raise_apiexception(*_a: Any, **_kw: Any) -> None:
            raise ApiException(status=503, reason="ServiceUnavailable")

        with (
            mock_patch(
                "aiperf.operator.handlers.monitor._monitor_tick",
                side_effect=raise_apiexception,
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor.k8s_client",
                new=lambda: _fake_k8s_client(),
            ),
            pytest.raises(kopf.TemporaryError),
        ):
            await monitor_progress(
                body=body,
                status=status,
                spec={},
                name="x",
                namespace="ns",
                patch=patch,
            )

        assert patch.status.get("observedGeneration") is None, (
            "monitor_progress stamped observedGeneration on a failed reconcile; "
            "this lies to kubectl-wait and GitOps tooling about acknowledged "
            "spec generation. The stamp must remain after the successful "
            "_monitor_tick path only."
        )


# =============================================================================
# Metrics endpoint smoke test: prometheus_client exposition really renders.
# =============================================================================


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.timeout(10)
def test_metrics_endpoint_serves_prometheus_exposition() -> None:
    """End-to-end smoke: actually start the server, scrape /metrics, parse,
    verify our four metric names appear.

    NOTE: prometheus_client.start_http_server uses the global REGISTRY and
    a process-wide port. Run via ``-n auto`` is fine because each xdist
    worker is its own process; the OS-assigned port keeps shards isolated.
    """
    from aiperf.operator.metrics import HANDLER_TOTAL, start_metrics_server

    port = _free_port()
    start_metrics_server(port)

    # Trigger samples for each metric so the counters/histograms appear in
    # the exposition (Counter/Histogram series are lazily materialized).
    HANDLER_TOTAL.labels(handler="smoke", outcome="success").inc()
    COMPLETION_CLAIM_RACES.inc()

    # Daemon thread takes a moment to bind.
    time.sleep(0.05)

    deadline = time.monotonic() + 3.0
    last_err: Exception | None = None
    # Bypass any HTTP_PROXY env in the sandbox: localhost requests via the
    # configured proxy returned 405 from the proxy itself, masking the
    # actual /metrics exposition. ProxyHandler({}) disables proxying for
    # this opener.
    no_proxy_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    while time.monotonic() < deadline:
        try:
            with no_proxy_opener.open(
                f"http://127.0.0.1:{port}/metrics", timeout=2
            ) as resp:
                body = resp.read().decode("utf-8")
                content_type = resp.headers.get("Content-Type", "")
            break
        except Exception as e:  # noqa: BLE001 - retrying transient bind delay
            last_err = e
            time.sleep(0.05)
    else:
        raise AssertionError(f"could not scrape /metrics: {last_err}")

    assert "text/plain" in content_type, content_type
    assert "aiperf_operator_handler_duration_seconds" in body
    assert "aiperf_operator_handler_total" in body
    assert "aiperf_operator_completion_claim_races" in body
    assert 'aiperf_operator_handler_total{handler="smoke",outcome="success"}' in body
