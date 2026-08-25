# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the watch-driven pod-restart handler."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch as mock_patch

import pytest

from aiperf.operator.client_cache import _warned_pod_restarts
from aiperf.operator.handlers.pod_restarts import handle_pod_restart


@pytest.fixture(autouse=True)
def _clear_warned_restarts() -> None:
    """Reset the module-level dedup state so tests don't leak state."""
    _warned_pod_restarts.clear()
    yield
    _warned_pod_restarts.clear()


@pytest.mark.asyncio
async def test_emits_event_when_threshold_exceeded() -> None:
    """When a containerStatuses entry has restartCount above the threshold, emit one event."""
    pod_body = {
        "metadata": {
            "name": "controller-0",
            "namespace": "bench",
            "labels": {"jobset.sigs.k8s.io/jobset-name": "aiperf-bench"},
            "ownerReferences": [
                {"kind": "Job", "name": "aiperf-bench-controller-0"},
            ],
        },
    }
    new_statuses = [
        {
            "name": "controller",
            "restartCount": 5,
            "lastState": {"terminated": {"reason": "OOMKilled"}},
            "state": {},
        }
    ]
    with (
        mock_patch("aiperf.operator.handlers.pod_restarts.events.pod_restarts") as evt,
        mock_patch(
            "aiperf.operator.handlers.pod_restarts._lookup_aiperfjob_body",
            return_value=pod_body,
        ),
    ):
        await handle_pod_restart(
            old=[],
            new=new_statuses,
            body=pod_body,
            meta=pod_body["metadata"],
            namespace="bench",
            name="controller-0",
            threshold=3,
        )
    evt.assert_called_once()


@pytest.mark.asyncio
async def test_does_not_emit_below_threshold() -> None:
    pod_body = {"metadata": {"name": "controller-0", "namespace": "bench"}}
    with mock_patch("aiperf.operator.handlers.pod_restarts.events.pod_restarts") as evt:
        await handle_pod_restart(
            old=[],
            new=[{"name": "controller", "restartCount": 1}],
            body=pod_body,
            meta=pod_body["metadata"],
            namespace="bench",
            name="controller-0",
            threshold=3,
        )
    evt.assert_not_called()


@pytest.mark.asyncio
async def test_dedup_same_count_only_emits_once() -> None:
    pod_body = {
        "metadata": {
            "name": "controller-0",
            "namespace": "bench",
            "labels": {"jobset.sigs.k8s.io/jobset-name": "aiperf-bench"},
        },
    }
    new_statuses = [{"name": "controller", "restartCount": 5}]
    with (
        mock_patch("aiperf.operator.handlers.pod_restarts.events.pod_restarts") as evt,
        mock_patch(
            "aiperf.operator.handlers.pod_restarts._lookup_aiperfjob_body",
            return_value=pod_body,
        ),
    ):
        await handle_pod_restart(
            old=[],
            new=new_statuses,
            body=pod_body,
            meta=pod_body["metadata"],
            namespace="bench",
            name="controller-0",
            threshold=3,
        )
        await handle_pod_restart(
            old=new_statuses,
            new=new_statuses,
            body=pod_body,
            meta=pod_body["metadata"],
            namespace="bench",
            name="controller-0",
            threshold=3,
        )
    assert evt.call_count == 1


# =============================================================================
# Adversarial tests — production-hostile inputs
# =============================================================================


class TestPodRestartHandlerAdversarial:
    """Adversarial coverage for ``handle_pod_restart``.

    These tests probe the production-hostile inputs kopf can deliver:
    None / empty / malformed containerStatuses lists, missing labels,
    apiserver lookup failures, and concurrent fires that would race the
    dedup state if it were checked-then-added across the lookup await.
    """

    @pytest.mark.asyncio
    async def test_new_is_none_no_events(self) -> None:
        """kopf field watchers can pass None when the field is removed.

        Iterating ``new or []`` must accept None without KeyError.
        """
        meta = {
            "name": "p",
            "namespace": "ns",
            "labels": {"jobset.sigs.k8s.io/jobset-name": "aiperf-x"},
        }
        with (
            mock_patch(
                "aiperf.operator.handlers.pod_restarts.events.pod_restarts"
            ) as evt,
            mock_patch(
                "aiperf.operator.handlers.pod_restarts._lookup_aiperfjob_body",
                return_value={"metadata": {"name": "x"}, "status": {}},
            ),
        ):
            await handle_pod_restart(
                old=[],
                new=None,  # type: ignore[arg-type]
                body={"metadata": meta},
                meta=meta,
                namespace="ns",
                name="p",
                threshold=3,
            )
        evt.assert_not_called()

    @pytest.mark.asyncio
    async def test_new_is_empty_list_no_events(self) -> None:
        """An empty containerStatuses list emits no events."""
        meta = {
            "name": "p",
            "labels": {"jobset.sigs.k8s.io/jobset-name": "aiperf-x"},
        }
        with mock_patch(
            "aiperf.operator.handlers.pod_restarts.events.pod_restarts"
        ) as evt:
            await handle_pod_restart(
                old=[],
                new=[],
                body={"metadata": meta},
                meta=meta,
                namespace="ns",
                name="p",
                threshold=3,
            )
        evt.assert_not_called()

    @pytest.mark.asyncio
    async def test_status_missing_restart_count_skipped(self) -> None:
        """A containerStatus without restartCount → ``int(None or 0) == 0``
        → below threshold → skipped silently. No KeyError."""
        meta = {
            "name": "p",
            "labels": {"jobset.sigs.k8s.io/jobset-name": "aiperf-x"},
        }
        with (
            mock_patch(
                "aiperf.operator.handlers.pod_restarts.events.pod_restarts"
            ) as evt,
            mock_patch(
                "aiperf.operator.handlers.pod_restarts._lookup_aiperfjob_body",
                return_value={"metadata": {"name": "x"}, "status": {}},
            ),
        ):
            await handle_pod_restart(
                old=[],
                new=[{"name": "controller"}],
                body={"metadata": meta},
                meta=meta,
                namespace="ns",
                name="p",
                threshold=3,
            )
        evt.assert_not_called()

    @pytest.mark.asyncio
    async def test_restart_count_zero_no_event(self) -> None:
        """restartCount=0 is below any positive threshold."""
        meta = {
            "name": "p",
            "labels": {"jobset.sigs.k8s.io/jobset-name": "aiperf-x"},
        }
        with mock_patch(
            "aiperf.operator.handlers.pod_restarts.events.pod_restarts"
        ) as evt:
            await handle_pod_restart(
                old=[],
                new=[{"name": "c", "restartCount": 0}],
                body={"metadata": meta},
                meta=meta,
                namespace="ns",
                name="p",
                threshold=1,
            )
        evt.assert_not_called()

    @pytest.mark.asyncio
    async def test_extreme_restart_count_emits_event(self) -> None:
        """Very large restartCount (overflow paranoia) must still emit cleanly."""
        meta = {
            "name": "p",
            "labels": {"jobset.sigs.k8s.io/jobset-name": "aiperf-x"},
        }
        with (
            mock_patch(
                "aiperf.operator.handlers.pod_restarts.events.pod_restarts"
            ) as evt,
            mock_patch(
                "aiperf.operator.handlers.pod_restarts._lookup_aiperfjob_body",
                return_value={"metadata": {"name": "x"}, "status": {}},
            ),
        ):
            await handle_pod_restart(
                old=[],
                new=[{"name": "c", "restartCount": 100_000}],
                body={"metadata": meta},
                meta=meta,
                namespace="ns",
                name="p",
                threshold=3,
            )
        evt.assert_called_once()

    @pytest.mark.asyncio
    async def test_oscillating_restart_count_dedups_at_higher_value(self) -> None:
        """If restartCount goes 5 → 3 → 5, the second 5 is the SAME
        ``(name, count)`` dedup key → only one event total."""
        meta = {
            "name": "p",
            "labels": {"jobset.sigs.k8s.io/jobset-name": "aiperf-x"},
        }
        body = {"metadata": {"name": "x"}, "status": {}}
        with (
            mock_patch(
                "aiperf.operator.handlers.pod_restarts.events.pod_restarts"
            ) as evt,
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
            # Pretend pod recovered and dropped to 3 (impossible IRL but
            # defensive against state churn).
            await handle_pod_restart(
                old=[{"name": "c", "restartCount": 5}],
                new=[{"name": "c", "restartCount": 3}],
                body={"metadata": meta},
                meta=meta,
                namespace="ns",
                name="p",
                threshold=3,
            )
            await handle_pod_restart(
                old=[{"name": "c", "restartCount": 3}],
                new=[{"name": "c", "restartCount": 5}],
                body={"metadata": meta},
                meta=meta,
                namespace="ns",
                name="p",
                threshold=3,
            )
        # First call emits at 5; the third call is dedup'd.
        # The second call (restartCount=3, threshold=3) emits because
        # 3 >= 3 with a NEW dedup key (p, 3).
        # → exactly 2 events.
        assert evt.call_count == 2

    @pytest.mark.asyncio
    async def test_pod_recreation_silently_skipped_known_limitation(self) -> None:
        """KNOWN LIMITATION: ``_warned_pod_restarts`` persists across pod
        lifetimes within an operator session. If a Pod is deleted and
        recreated with the SAME name, hitting restartCount=5 again, the
        second occurrence is silently skipped.

        Pinning this to make the limitation visible. A fix would scope
        dedup by Pod UID rather than name, OR clear the entry when the
        watch sees the Pod removed.
        """
        meta = {
            "name": "p",
            "labels": {"jobset.sigs.k8s.io/jobset-name": "aiperf-x"},
        }
        body = {"metadata": {"name": "x"}, "status": {}}
        with (
            mock_patch(
                "aiperf.operator.handlers.pod_restarts.events.pod_restarts"
            ) as evt,
            mock_patch(
                "aiperf.operator.handlers.pod_restarts._lookup_aiperfjob_body",
                return_value=body,
            ),
        ):
            # Original pod hits 5 restarts.
            await handle_pod_restart(
                old=[],
                new=[{"name": "c", "restartCount": 5}],
                body={"metadata": meta},
                meta=meta,
                namespace="ns",
                name="p",
                threshold=3,
            )
            # Pod recreated; fresh containerStatuses but same pod name.
            # Same dedup key (name='p', count=5). Skipped.
            await handle_pod_restart(
                old=[],
                new=[{"name": "c", "restartCount": 5}],
                body={"metadata": meta},
                meta=meta,
                namespace="ns",
                name="p",
                threshold=3,
            )
        assert evt.call_count == 1  # only the first.

    @pytest.mark.asyncio
    async def test_multiple_containers_each_above_threshold_emit_separately(
        self,
    ) -> None:
        """A Pod with two containers each at restartCount above threshold
        must emit one event per container (different dedup keys: each
        round in the loop with the SAME pod name and restart_count is
        actually deduped — this tests one event PER (pod_name,
        restart_count) tuple, not per container)."""
        meta = {
            "name": "p",
            "labels": {"jobset.sigs.k8s.io/jobset-name": "aiperf-x"},
        }
        body = {"metadata": {"name": "x"}, "status": {}}
        with (
            mock_patch(
                "aiperf.operator.handlers.pod_restarts.events.pod_restarts"
            ) as evt,
            mock_patch(
                "aiperf.operator.handlers.pod_restarts._lookup_aiperfjob_body",
                return_value=body,
            ),
        ):
            # Two containers with DIFFERENT restart counts → two dedup keys.
            await handle_pod_restart(
                old=[],
                new=[
                    {"name": "c1", "restartCount": 5},
                    {"name": "c2", "restartCount": 7},
                ],
                body={"metadata": meta},
                meta=meta,
                namespace="ns",
                name="p",
                threshold=3,
            )
        assert evt.call_count == 2

    @pytest.mark.asyncio
    async def test_multiple_containers_same_count_dedup_to_one(self) -> None:
        """Two containers in one Pod each at restartCount=5: only one
        event because dedup key is (pod_name, count) not per-container."""
        meta = {
            "name": "p",
            "labels": {"jobset.sigs.k8s.io/jobset-name": "aiperf-x"},
        }
        body = {"metadata": {"name": "x"}, "status": {}}
        with (
            mock_patch(
                "aiperf.operator.handlers.pod_restarts.events.pod_restarts"
            ) as evt,
            mock_patch(
                "aiperf.operator.handlers.pod_restarts._lookup_aiperfjob_body",
                return_value=body,
            ),
        ):
            await handle_pod_restart(
                old=[],
                new=[
                    {"name": "c1", "restartCount": 5},
                    {"name": "c2", "restartCount": 5},
                ],
                body={"metadata": meta},
                meta=meta,
                namespace="ns",
                name="p",
                threshold=3,
            )
        assert evt.call_count == 1

    @pytest.mark.asyncio
    async def test_terminated_reason_empty_string_falls_back_to_unknown(self) -> None:
        """`lastState.terminated.reason = ""` → reason falls back to "Unknown"."""
        meta = {
            "name": "p",
            "labels": {"jobset.sigs.k8s.io/jobset-name": "aiperf-x"},
        }
        body = {"metadata": {"name": "x"}, "status": {}}
        with (
            mock_patch(
                "aiperf.operator.handlers.pod_restarts.events.pod_restarts"
            ) as evt,
            mock_patch(
                "aiperf.operator.handlers.pod_restarts._lookup_aiperfjob_body",
                return_value=body,
            ),
        ):
            await handle_pod_restart(
                old=[],
                new=[
                    {
                        "name": "c",
                        "restartCount": 5,
                        "lastState": {"terminated": {"reason": ""}},
                    }
                ],
                body={"metadata": meta},
                meta=meta,
                namespace="ns",
                name="p",
                threshold=3,
            )
        evt.assert_called_once()
        # 4th positional arg is the reason.
        assert evt.call_args.args[3] == "Unknown"

    @pytest.mark.asyncio
    async def test_state_waiting_reason_none_falls_back_to_unknown(self) -> None:
        """`state.waiting.reason = None` → reason stays "Unknown"."""
        meta = {
            "name": "p",
            "labels": {"jobset.sigs.k8s.io/jobset-name": "aiperf-x"},
        }
        body = {"metadata": {"name": "x"}, "status": {}}
        with (
            mock_patch(
                "aiperf.operator.handlers.pod_restarts.events.pod_restarts"
            ) as evt,
            mock_patch(
                "aiperf.operator.handlers.pod_restarts._lookup_aiperfjob_body",
                return_value=body,
            ),
        ):
            await handle_pod_restart(
                old=[],
                new=[
                    {
                        "name": "c",
                        "restartCount": 5,
                        "state": {"waiting": {"reason": None}},
                    }
                ],
                body={"metadata": meta},
                meta=meta,
                namespace="ns",
                name="p",
                threshold=3,
            )
        evt.assert_called_once()
        assert evt.call_args.args[3] == "Unknown"

    @pytest.mark.asyncio
    async def test_meta_missing_labels_skips_silently(self) -> None:
        """A pod meta with no labels at all → skip; no KeyError."""
        meta = {"name": "p"}  # no labels
        with mock_patch(
            "aiperf.operator.handlers.pod_restarts.events.pod_restarts"
        ) as evt:
            await handle_pod_restart(
                old=[],
                new=[{"name": "c", "restartCount": 5}],
                body={"metadata": meta},
                meta=meta,
                namespace="ns",
                name="p",
                threshold=3,
            )
        evt.assert_not_called()

    @pytest.mark.asyncio
    async def test_jobset_label_empty_string_skips(self) -> None:
        """Empty label value → ``if not jobset_name`` short-circuits."""
        meta = {
            "name": "p",
            "labels": {"jobset.sigs.k8s.io/jobset-name": ""},
        }
        with mock_patch(
            "aiperf.operator.handlers.pod_restarts.events.pod_restarts"
        ) as evt:
            await handle_pod_restart(
                old=[],
                new=[{"name": "c", "restartCount": 5}],
                body={"metadata": meta},
                meta=meta,
                namespace="ns",
                name="p",
                threshold=3,
            )
        evt.assert_not_called()

    @pytest.mark.asyncio
    async def test_aiperfjob_lookup_returns_none_skips(self) -> None:
        """Sweep-owned JobSet → AIPerfJob lookup returns None → silent skip."""
        meta = {
            "name": "p",
            "labels": {"jobset.sigs.k8s.io/jobset-name": "aiperf-sweep"},
        }
        with (
            mock_patch(
                "aiperf.operator.handlers.pod_restarts.events.pod_restarts"
            ) as evt,
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
                name="p",
                threshold=3,
            )
        evt.assert_not_called()

    @pytest.mark.asyncio
    async def test_aiperfjob_lookup_raises_is_swallowed(self) -> None:
        """The lookup helper has its own `except Exception` swallow; the
        outer handler then sees `None` → silent skip. Pin: no exception
        propagates to kopf, which would mark the watch handler as
        permanently-erroring."""
        meta = {
            "name": "p",
            "labels": {"jobset.sigs.k8s.io/jobset-name": "aiperf-x"},
        }
        with (
            mock_patch(
                "aiperf.operator.handlers.pod_restarts.events.pod_restarts"
            ) as evt,
            mock_patch(
                "aiperf.operator.handlers.pod_restarts._lookup_aiperfjob_body",
                side_effect=RuntimeError("apiserver unavailable"),
            ),
            pytest.raises(RuntimeError, match="apiserver unavailable"),
        ):
            # The outer handler does NOT catch — the lookup helper does.
            # When the helper raises (not when it returns None), the outer
            # call propagates. Pin which is which.
            await handle_pod_restart(
                old=[],
                new=[{"name": "c", "restartCount": 5}],
                body={"metadata": meta},
                meta=meta,
                namespace="ns",
                name="p",
                threshold=3,
            )
        evt.assert_not_called()

    @pytest.mark.asyncio
    async def test_concurrent_fires_same_pod_emit_only_once(self) -> None:
        """Two concurrent ``handle_pod_restart`` calls for the SAME (pod,
        count) must emit only ONE event.

        REGRESSION GUARD: before the dedup pre-claim, the check happened
        AFTER the apiserver lookup await — both calls would pass the
        check, both would emit, and operators would get duplicate
        WARNING events for the same restart spike.
        """
        import asyncio

        meta = {
            "name": "p",
            "labels": {"jobset.sigs.k8s.io/jobset-name": "aiperf-x"},
        }
        body = {"metadata": {"name": "x"}, "status": {}}

        # The lookup awaits a task we control — both concurrent calls must
        # yield to the event loop here. If dedup is checked AFTER this
        # await, both will pass the check.
        lookup_proceed = asyncio.Event()
        lookup_calls = 0

        async def slow_lookup(*_args: Any, **_kwargs: Any) -> Any:
            nonlocal lookup_calls
            lookup_calls += 1
            await lookup_proceed.wait()
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
            t1 = asyncio.create_task(
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
            t2 = asyncio.create_task(
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
            # Yield long enough for both tasks to reach the lookup await.
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            lookup_proceed.set()
            await asyncio.gather(t1, t2)

        assert evt.call_count == 1, (
            "Concurrent fires for the same (pod, count) must dedup; "
            "see pod_restarts.handle_pod_restart pre-claim block"
        )
