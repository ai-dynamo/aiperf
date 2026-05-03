# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.common.models import CreditPhaseStats
from aiperf.credit.sticky_router import StickyCreditRouter, WorkerLoad
from aiperf.credit.structs import Credit, TurnToSend
from aiperf.plugin.enums import TimingMode
from aiperf.records.records_tracker import RecordsTracker
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.phase.credit_counter import CreditCounter
from aiperf.timing.phase.lifecycle import PhaseLifecycle
from aiperf.timing.phase.progress_tracker import PhaseProgressTracker
from aiperf.timing.phase.stop_conditions import StopConditionChecker


def _turn(cid="c1", tidx=0, nt=1, xcid=None):
    return TurnToSend(
        conversation_id=cid,
        x_correlation_id=xcid or f"x-{cid}",
        turn_index=tidx,
        num_turns=nt,
    )


def _credit(ph=CreditPhase.PROFILING, cid=1, conv="c1", tidx=0, nt=1):
    return Credit(
        id=cid,
        phase=ph,
        conversation_id=conv,
        x_correlation_id=f"x-{conv}",
        turn_index=tidx,
        num_turns=nt,
        issued_at_ns=time.time_ns(),
    )


def _components(cfg):
    lc = PhaseLifecycle(cfg)
    pr = PhaseProgressTracker(cfg)
    sc = StopConditionChecker(config=cfg, lifecycle=lc, counter=pr.counter)
    return lc, pr, sc


def _cfg(
    ph=CreditPhase.PROFILING, req=None, dur=None, grace=None, sess=None, seamless=False
):
    return CreditPhaseConfig(
        phase=ph,
        timing_mode=TimingMode.REQUEST_RATE,
        total_expected_requests=req,
        expected_duration_sec=dur,
        grace_period_sec=grace,
        expected_num_sessions=sess,
        seamless=seamless,
    )


@pytest.mark.asyncio
class TestCreditReturnRace:
    async def test_late_return_after_complete(self):
        cfg = _cfg(req=10, dur=1.0, grace=0.5)
        lc, pr, _ = _components(cfg)
        lc.start()
        for i in range(10):
            pr.increment_sent(_turn(f"c{i}"))
        lc.mark_sending_complete()
        pr.freeze_sent_counts()
        for i in range(9):
            pr.increment_returned(
                _credit(cid=i, conv=f"c{i}").is_final_turn, cancelled=False
            )
        lc.mark_complete(grace_period_triggered=True)
        pr.freeze_completed_counts()
        assert lc.is_complete
        late = _credit(cid=9, conv="c9")
        if not lc.is_complete:
            pr.increment_returned(late.is_final_turn, cancelled=False)
        assert pr.create_stats(lc).requests_completed == 9

    async def test_concurrent_returns_completion(self):
        cfg = _cfg(req=3)
        lc, pr, _ = _components(cfg)
        lc.start()
        for i in range(3):
            pr.increment_sent(_turn(f"c{i}"))
        lc.mark_sending_complete()
        pr.freeze_sent_counts()
        pr.increment_returned(_credit(cid=0, conv="c0").is_final_turn, cancelled=False)
        r1 = pr.increment_returned(
            _credit(cid=1, conv="c1").is_final_turn, cancelled=False
        )
        if r1:
            pr.all_credits_returned_event.set()
        r2 = pr.increment_returned(
            _credit(cid=2, conv="c2").is_final_turn, cancelled=False
        )
        if r2:
            pr.all_credits_returned_event.set()
        assert (r1 and not r2) or (not r1 and r2)
        assert pr.all_credits_returned_event.is_set()

    async def test_cancelled_and_completed_both_counted(self):
        cfg = _cfg(req=4)
        lc, pr, _ = _components(cfg)
        lc.start()
        for i in range(4):
            pr.increment_sent(_turn(f"c{i}"))
        lc.mark_sending_complete()
        pr.freeze_sent_counts()
        for i in range(2):
            r = pr.increment_returned(
                _credit(cid=i, conv=f"c{i}").is_final_turn, cancelled=False
            )
            if r:
                pr.all_credits_returned_event.set()
        for i in range(2, 4):
            r = pr.increment_returned(
                _credit(cid=i, conv=f"c{i}").is_final_turn, cancelled=True
            )
            if r:
                pr.all_credits_returned_event.set()
            if i == 3:
                assert r
        assert pr.all_credits_returned_event.is_set()
        st = pr.create_stats(lc)
        assert st.requests_completed == 2 and st.requests_cancelled == 2


class TestRecordsManagerRace:
    @pytest.mark.parametrize("records_first", [True, False], ids=["records_first", "phase_first"])  # fmt: skip
    def test_records_vs_phase_complete_order(self, records_first):
        rt = RecordsTracker()
        if records_first:
            rt.update_phase_info(
                CreditPhaseStats(
                    phase=CreditPhase.PROFILING,
                    total_expected_requests=5,
                    start_ns=1000,
                )
            )
            for _ in range(5):
                rt._get_phase_tracker(CreditPhase.PROFILING).increment_success_records()
            assert not rt.check_and_set_all_records_received_for_phase(
                CreditPhase.PROFILING
            )
            rt.update_phase_info(
                CreditPhaseStats(
                    phase=CreditPhase.PROFILING,
                    final_requests_completed=5,
                    start_ns=1000,
                    requests_end_ns=2000,
                )
            )
            assert rt.check_and_set_all_records_received_for_phase(
                CreditPhase.PROFILING
            )
        else:
            rt.update_phase_info(
                CreditPhaseStats(
                    phase=CreditPhase.PROFILING,
                    final_requests_completed=3,
                    start_ns=1000,
                    requests_end_ns=2000,
                )
            )
            assert not rt.check_and_set_all_records_received_for_phase(
                CreditPhase.PROFILING
            )
            for _ in range(3):
                rt._get_phase_tracker(CreditPhase.PROFILING).increment_success_records()
            assert rt.check_and_set_all_records_received_for_phase(
                CreditPhase.PROFILING
            )

    def test_duplicate_completion_returns_false(self):
        rt = RecordsTracker()
        rt.update_phase_info(
            CreditPhaseStats(
                phase=CreditPhase.PROFILING, final_requests_completed=1, start_ns=1000
            )
        )
        rt._get_phase_tracker(CreditPhase.PROFILING).increment_success_records()
        assert rt.check_and_set_all_records_received_for_phase(CreditPhase.PROFILING)
        assert not rt.check_and_set_all_records_received_for_phase(
            CreditPhase.PROFILING
        )


@pytest.mark.asyncio
class TestStickyRouterWorkerRace:
    async def test_credit_to_unregistered_worker(self, service_config):
        r = StickyCreditRouter(service_config=service_config, service_id="tr")
        r._workers = {"w1": WorkerLoad(worker_id="w1", in_flight_credits=5)}
        r._workers["w1"].active_credit_ids = set(range(5))
        r._workers_cache = list(r._workers.values())
        r._cancellation_pending = True
        r._unregister_worker("w1")
        r._track_credit_returned("w1", 0, cancelled=True, error_reported=False)

    async def test_worker_registration_during_routing(self, service_config):
        r = StickyCreditRouter(service_config=service_config, service_id="tr")
        r._register_worker("w1")
        assert len(r._workers) == 1
        r._register_worker("w2")
        assert len(r._workers) == 2 and {w.worker_id for w in r._workers_cache} == {
            "w1",
            "w2",
        }

    async def test_worker_unregister_clears_cache(self, service_config):
        r = StickyCreditRouter(service_config=service_config, service_id="tr")
        r._register_worker("w1")
        r._register_worker("w2")
        r._unregister_worker("w1")
        assert len(r._workers) == 1 and r._workers_cache[0].worker_id == "w2"


class TestCreditCounterAtomicity:
    def test_unique_indices(self):
        cnt = CreditCounter(_cfg(req=100))
        idxs = [cnt.increment_sent(_turn(f"c{i}"))[0] for i in range(100)]
        assert idxs == list(range(100)) and cnt.requests_sent == 100

    def test_returned_tracking(self):
        cnt = CreditCounter(_cfg(req=10))
        for i in range(10):
            cnt.increment_sent(_turn(f"c{i}"))
        cnt.freeze_sent_counts()
        for i in range(5):
            cnt.increment_returned(
                _credit(cid=i, conv=f"c{i}").is_final_turn, cancelled=False
            )
        for i in range(5, 10):
            r = cnt.increment_returned(
                _credit(cid=i, conv=f"c{i}").is_final_turn, cancelled=True
            )
            if i == 9:
                assert r
        assert (
            cnt.requests_completed == 5
            and cnt.requests_cancelled == 5
            and cnt.in_flight == 0
        )

    def test_multi_turn_sessions(self):
        cnt = CreditCounter(_cfg(sess=3))
        for i in range(3):
            cnt.increment_sent(_turn(f"c{i}", tidx=0, nt=2))
        assert cnt.sent_sessions == 3 and cnt.total_session_turns == 6
        for i in range(3):
            cnt.increment_sent(
                TurnToSend(
                    conversation_id=f"c{i}",
                    x_correlation_id=f"x-c{i}",
                    turn_index=1,
                    num_turns=2,
                )
            )
        assert cnt.sent_sessions == 3 and cnt.requests_sent == 6
        cnt.freeze_sent_counts()
        for i in range(3):
            cnt.increment_returned(
                _credit(cid=i * 2, conv=f"c{i}", tidx=0, nt=2).is_final_turn,
                cancelled=False,
            )
        assert cnt.completed_sessions == 0
        for i in range(3):
            cnt.increment_returned(
                _credit(cid=i * 2 + 1, conv=f"c{i}", tidx=1, nt=2).is_final_turn,
                cancelled=False,
            )
        assert cnt.completed_sessions == 3

    def test_cancelled_session_tracking(self):
        cnt = CreditCounter(_cfg(sess=3))
        for i in range(3):
            cnt.increment_sent(_turn(f"c{i}"))
        assert cnt.sent_sessions == 3 and cnt.in_flight_sessions == 3
        cnt.freeze_sent_counts()
        cnt.increment_returned(_credit(cid=0, conv="c0").is_final_turn, cancelled=False)
        cnt.increment_returned(_credit(cid=1, conv="c1").is_final_turn, cancelled=False)
        assert cnt.completed_sessions == 2 and cnt.in_flight_sessions == 1
        cnt.increment_returned(_credit(cid=2, conv="c2").is_final_turn, cancelled=True)
        assert (
            cnt.completed_sessions == 2
            and cnt.cancelled_sessions == 1
            and cnt.in_flight_sessions == 0
        )

    def test_in_flight_sessions_incomplete(self):
        cnt = CreditCounter(_cfg(req=6))
        for i in range(3):
            cnt.increment_sent(_turn(f"c{i}", tidx=0, nt=2))
        assert cnt.sent_sessions == 3 and cnt.in_flight_sessions == 3
        for i in range(3):
            cr = _credit(cid=i, conv=f"c{i}", tidx=0, nt=2)
            assert not cr.is_final_turn
            cnt.increment_returned(cr.is_final_turn, cancelled=False)
        assert cnt.completed_sessions == 0 and cnt.in_flight_sessions == 3


@pytest.mark.asyncio
class TestDeadlockPrevention:
    async def test_all_sent_event_set(self):
        cfg = _cfg(req=3)
        lc, pr, _ = _components(cfg)
        lc.start()
        for i in range(3):
            _, final = pr.increment_sent(_turn(f"c{i}"))
            if final:
                lc.mark_sending_complete()
                pr.freeze_sent_counts()
                pr.all_credits_sent_event.set()
        assert pr.all_credits_sent_event.is_set() and lc.is_sending_complete

    async def test_all_returned_event_set(self):
        cfg = _cfg(req=3)
        lc, pr, _ = _components(cfg)
        lc.start()
        for i in range(3):
            pr.increment_sent(_turn(f"c{i}"))
        lc.mark_sending_complete()
        pr.freeze_sent_counts()
        for i in range(3):
            if pr.increment_returned(
                _credit(cid=i, conv=f"c{i}").is_final_turn, cancelled=False
            ):
                pr.all_credits_returned_event.set()
        assert pr.all_credits_returned_event.is_set()

    async def test_duration_phase_no_credits(self):
        cfg = _cfg(dur=1.0)
        lc, pr, _ = _components(cfg)
        lc.start()
        lc.mark_sending_complete()
        pr.freeze_sent_counts()
        assert pr.check_all_returned_or_cancelled()


@pytest.mark.asyncio
class TestStickySessionRace:
    async def test_session_eviction_before_turn_completes(self, service_config):
        r = StickyCreditRouter(service_config=service_config, service_id="tr")
        r._router_client.send_to = AsyncMock()
        r._register_worker("w1")
        xcid = "multi"
        for tidx in range(3):
            await r.send_credit(
                Credit(
                    id=tidx,
                    phase=CreditPhase.PROFILING,
                    conversation_id="c1",
                    x_correlation_id=xcid,
                    turn_index=tidx,
                    num_turns=3,
                    issued_at_ns=time.time_ns(),
                )
            )
        assert xcid not in r._sticky_sessions
        assert r._workers["w1"].in_flight_credits == 3
        for cid in [2, 0, 1]:
            r._track_credit_returned("w1", cid, cancelled=False, error_reported=False)
        assert (
            r._workers["w1"].in_flight_credits == 0
            and r._workers["w1"].total_completed_credits == 3
        )

    async def test_worker_unregisters_mid_session(self, service_config):
        r = StickyCreditRouter(service_config=service_config, service_id="tr")
        r._router_client.send_to = AsyncMock()
        r._register_worker("w1")
        r._register_worker("w2")
        xcid = "reassign"
        r._workers["w2"].in_flight_credits = 10
        r._workers_by_load[10].add("w2")
        r._workers_by_load[0].discard("w2")
        await r.send_credit(
            Credit(
                id=0,
                phase=CreditPhase.PROFILING,
                conversation_id="c1",
                x_correlation_id=xcid,
                turn_index=0,
                num_turns=2,
                issued_at_ns=time.time_ns(),
            )
        )
        w0 = r._router_client.send_to.call_args[0][0]
        assert r._sticky_sessions[xcid] == w0
        r._cancellation_pending = True
        r._unregister_worker(w0)
        r._min_load = 10
        if xcid in r._sticky_sessions:
            del r._sticky_sessions[xcid]
        await r.send_credit(
            Credit(
                id=1,
                phase=CreditPhase.PROFILING,
                conversation_id="c1",
                x_correlation_id=xcid,
                turn_index=1,
                num_turns=2,
                issued_at_ns=time.time_ns(),
            )
        )
        assert r._router_client.send_to.call_args[0][0] == "w2"


@pytest.mark.asyncio
class TestMultiTurnCreditRace:
    async def test_interleaved_conversations(self):
        cfg = _cfg(sess=2)
        lc, pr, _ = _components(cfg)
        lc.start()
        pr.increment_sent(_turn("cA", 0, 3))
        pr.increment_sent(_turn("cB", 0, 3))
        pr.increment_sent(
            TurnToSend(
                conversation_id="cA", x_correlation_id="x-cA", turn_index=1, num_turns=3
            )
        )
        pr.increment_sent(
            TurnToSend(
                conversation_id="cB", x_correlation_id="x-cB", turn_index=1, num_turns=3
            )
        )
        pr.increment_sent(
            TurnToSend(
                conversation_id="cA", x_correlation_id="x-cA", turn_index=2, num_turns=3
            )
        )
        _, final = pr.increment_sent(
            TurnToSend(
                conversation_id="cB", x_correlation_id="x-cB", turn_index=2, num_turns=3
            )
        )
        assert final
        lc.mark_sending_complete()
        pr.freeze_sent_counts()
        st = pr.create_stats(lc)
        assert (
            st.sent_sessions == 2
            and st.total_session_turns == 6
            and st.requests_sent == 6
        )

    async def test_partial_cancellation(self):
        cfg = _cfg(req=3)
        lc, pr, _ = _components(cfg)
        lc.start()
        for i in range(3):
            pr.increment_sent(
                TurnToSend(
                    conversation_id="c1",
                    x_correlation_id="x-c1",
                    turn_index=i,
                    num_turns=3,
                )
            )
        lc.mark_sending_complete()
        pr.freeze_sent_counts()
        pr.increment_returned(
            _credit(cid=0, conv="c1", tidx=0, nt=3).is_final_turn, cancelled=False
        )
        pr.increment_returned(
            _credit(cid=1, conv="c1", tidx=1, nt=3).is_final_turn, cancelled=True
        )
        r = pr.increment_returned(
            _credit(cid=2, conv="c1", tidx=2, nt=3).is_final_turn, cancelled=False
        )
        assert r
        st = pr.create_stats(lc)
        assert (
            st.requests_completed == 2
            and st.requests_cancelled == 1
            and st.completed_sessions == 1
        )


@pytest.mark.asyncio
class TestFullCreditFlow:
    @pytest.mark.parametrize("order,cancelled", [([2, 0, 1], [False, False, False]), ([3, 0, 4, 2, 1], [True, False, True, False, False])], ids=["out_of_order", "mixed"])  # fmt: skip
    async def test_out_of_order_returns(self, order, cancelled):
        n = len(order)
        cfg = _cfg(req=n)
        lc, pr, _ = _components(cfg)
        lc.start()
        for i in range(n):
            pr.increment_sent(_turn(f"c{i}"))
        lc.mark_sending_complete()
        pr.freeze_sent_counts()
        for i, cnum in enumerate(order):
            if pr.increment_returned(
                _credit(cid=cnum, conv=f"c{cnum}").is_final_turn, cancelled=cancelled[i]
            ):
                pr.all_credits_returned_event.set()
        assert pr.all_credits_returned_event.is_set()
        st = pr.create_stats(lc)
        assert st.requests_completed == sum(1 for c in cancelled if not c)
        assert st.requests_cancelled == sum(1 for c in cancelled if c)


@pytest.mark.asyncio
class TestPhaseStateMachine:
    async def test_state_ordering(self):
        cfg = _cfg(req=5)
        lc, pr, _ = _components(cfg)
        assert not lc.is_started and not lc.is_sending_complete and not lc.is_complete
        lc.start()
        assert lc.is_started and not lc.is_sending_complete
        for i in range(5):
            pr.increment_sent(_turn(f"c{i}"))
        lc.mark_sending_complete()
        pr.freeze_sent_counts()
        assert lc.is_sending_complete and not lc.is_complete
        for i in range(5):
            if pr.increment_returned(
                _credit(cid=i, conv=f"c{i}").is_final_turn, cancelled=False
            ):
                pr.all_credits_returned_event.set()
        assert pr.all_credits_returned_event.is_set()
        lc.mark_complete()
        pr.freeze_completed_counts()
        assert lc.is_complete

    async def test_cannot_send_after_complete(self):
        cfg = _cfg(req=1)
        lc, pr, sc = _components(cfg)
        lc.start()
        pr.increment_sent(_turn("c1"))
        lc.mark_sending_complete()
        pr.freeze_sent_counts()
        pr.increment_returned(_credit(cid=0, conv="c1").is_final_turn, cancelled=False)
        lc.mark_complete()
        pr.freeze_completed_counts()
        assert lc.is_complete and not sc.can_send_any_turn()


class TestRecordsTrackerPhase:
    def test_additive_updates(self):
        rt = RecordsTracker()
        rt.update_phase_info(
            CreditPhaseStats(
                phase=CreditPhase.PROFILING, total_expected_requests=100, start_ns=1000
            )
        )
        ph = rt._get_phase_tracker(CreditPhase.PROFILING)
        assert ph._start_ns == 1000 and ph._final_requests_completed is None
        rt.update_phase_info(
            CreditPhaseStats(
                phase=CreditPhase.PROFILING, final_requests_sent=95, start_ns=1000
            )
        )
        rt.update_phase_info(
            CreditPhaseStats(
                phase=CreditPhase.PROFILING,
                final_requests_completed=90,
                requests_end_ns=2000,
                start_ns=1000,
            )
        )
        assert ph._final_requests_completed == 90 and ph._requests_end_ns == 2000

    def test_multi_worker_aggregation(self):
        rt = RecordsTracker()
        rt.update_phase_info(
            CreditPhaseStats(
                phase=CreditPhase.PROFILING, final_requests_completed=30, start_ns=1000
            )
        )
        ph = rt._get_phase_tracker(CreditPhase.PROFILING)
        for _ in range(28):
            ph.increment_success_records()
        for _ in range(2):
            ph.increment_error_records()
        assert rt.check_and_set_all_records_received_for_phase(CreditPhase.PROFILING)
        st = rt.create_stats_for_phase(CreditPhase.PROFILING)
        assert (
            st.success_records == 28
            and st.error_records == 2
            and st.total_records == 30
        )


@pytest.mark.asyncio
class TestWarmupToProfilingTransition:
    async def test_independent_counters(self):
        wcfg = _cfg(ph=CreditPhase.WARMUP, req=5, seamless=True)
        pcfg = _cfg(ph=CreditPhase.PROFILING, req=10)
        wlc, wpr, _ = _components(wcfg)
        plc, ppr, _ = _components(pcfg)
        wlc.start()
        for i in range(5):
            wpr.increment_sent(_turn(f"w{i}"))
        wlc.mark_sending_complete()
        wpr.freeze_sent_counts()
        plc.start()
        for i in range(10):
            ppr.increment_sent(_turn(f"p{i}"))
        plc.mark_sending_complete()
        ppr.freeze_sent_counts()
        wpr.increment_returned(
            Credit(
                id=0,
                phase=CreditPhase.WARMUP,
                conversation_id="w0",
                x_correlation_id="x-w0",
                turn_index=0,
                num_turns=1,
                issued_at_ns=time.time_ns(),
            ).is_final_turn,
            cancelled=False,
        )
        ppr.increment_returned(
            Credit(
                id=0,
                phase=CreditPhase.PROFILING,
                conversation_id="p0",
                x_correlation_id="x-p0",
                turn_index=0,
                num_turns=1,
                issued_at_ns=time.time_ns(),
            ).is_final_turn,
            cancelled=False,
        )
        wst, pst = wpr.create_stats(wlc), ppr.create_stats(plc)
        assert wst.requests_completed == 1 and wst.requests_sent == 5
        assert pst.requests_completed == 1 and pst.requests_sent == 10


@pytest.mark.asyncio
class TestRouterLoadBalancing:
    async def test_tie_selection(self, service_config):
        r = StickyCreditRouter(service_config=service_config, service_id="tr")
        r._router_client.send_to = AsyncMock()
        for w in ["w1", "w2", "w3"]:
            r._register_worker(w)
            r._workers[w].in_flight_credits = 5
        r._workers_by_load.clear()
        r._workers_by_load[5] = {"w1", "w2", "w3"}
        r._min_load = 5
        sel = set()
        for i in range(10):
            await r.send_credit(_credit(cid=i, conv=f"c{i}"))
            sel.add(r._router_client.send_to.call_args[0][0])
        assert all(w in {"w1", "w2", "w3"} for w in sel)

    async def test_prefers_lower_load(self, service_config):
        r = StickyCreditRouter(service_config=service_config, service_id="tr")
        r._router_client.send_to = AsyncMock()
        for w in ["w1", "w2", "w3"]:
            r._register_worker(w)
        for w in ["w1", "w2", "w3"]:
            r._workers_by_load[0].discard(w)
        (
            r._workers["w1"].in_flight_credits,
            r._workers["w2"].in_flight_credits,
            r._workers["w3"].in_flight_credits,
        ) = 20, 1, 10
        r._workers_by_load[20].add("w1")
        r._workers_by_load[1].add("w2")
        r._workers_by_load[10].add("w3")
        r._min_load = 1
        for i in range(5):
            await r.send_credit(_credit(cid=i, conv=f"c{i}"))
            assert r._router_client.send_to.call_args[0][0] == "w2"

    async def test_atomic_load_updates(self, service_config):
        r = StickyCreditRouter(service_config=service_config, service_id="tr")
        r._register_worker("w1")
        w = r._workers["w1"]
        for i in range(100):
            r._track_credit_sent("w1", i)
            assert w.in_flight_credits == i + 1
        for i in range(100):
            r._track_credit_returned("w1", i, cancelled=False, error_reported=False)
            assert w.in_flight_credits == 99 - i
        assert (
            w.in_flight_credits == 0
            and w.total_sent_credits == 100
            and w.total_completed_credits == 100
        )


@pytest.mark.asyncio
class TestCancellation:
    async def test_cancel_snapshots_state(self, service_config):
        r = StickyCreditRouter(service_config=service_config, service_id="tr")
        r._router_client = MagicMock()
        r._router_client.send_to = AsyncMock()
        r._workers = {
            "w1": WorkerLoad(worker_id="w1", in_flight_credits=3),
            "w2": WorkerLoad(worker_id="w2", in_flight_credits=2),
        }
        r._workers["w1"].active_credit_ids = {1, 2, 3}
        r._workers["w2"].active_credit_ids = {4, 5}
        r._workers_cache = list(r._workers.values())
        await r.cancel_all_credits()
        assert r._router_client.send_to.call_count == 2
        calls = {
            c[0][0]: set(c[0][1].credit_ids)
            for c in r._router_client.send_to.call_args_list
        }
        assert calls["w1"] == {1, 2, 3} and calls["w2"] == {4, 5}

    async def test_cancel_skips_no_inflight(self, service_config):
        r = StickyCreditRouter(service_config=service_config, service_id="tr")
        r._router_client = MagicMock()
        r._router_client.send_to = AsyncMock()
        r._workers = {
            "w1": WorkerLoad(worker_id="w1", in_flight_credits=0),
            "w2": WorkerLoad(worker_id="w2", in_flight_credits=5),
        }
        r._workers["w2"].active_credit_ids = set(range(5))
        r._workers_cache = list(r._workers.values())
        await r.cancel_all_credits()
        assert (
            r._router_client.send_to.call_count == 1
            and r._router_client.send_to.call_args[0][0] == "w2"
        )


@pytest.mark.asyncio
class TestHighVolume:
    async def test_1000_credits(self):
        cfg = _cfg(req=1000)
        lc, pr, _ = _components(cfg)
        lc.start()
        for i in range(1000):
            pr.increment_sent(_turn(f"c{i}"))
        lc.mark_sending_complete()
        pr.freeze_sent_counts()
        order = list(range(1000))
        for s in range(0, 1000, 100):
            order[s : s + 100] = reversed(order[s : s + 100])
        for i, cn in enumerate(order):
            r = pr.increment_returned(
                _credit(cid=cn, conv=f"c{cn}").is_final_turn, cancelled=False
            )
            if r:
                pr.all_credits_returned_event.set()
            if i == 999:
                assert r
        assert pr.all_credits_returned_event.is_set()
        st = pr.create_stats(lc)
        assert (
            st.requests_sent == 1000
            and st.requests_completed == 1000
            and st.in_flight_requests == 0
        )

    async def test_100_sessions_5_turns(self):
        cfg = _cfg(sess=100)
        lc, pr, _ = _components(cfg)
        lc.start()
        for s in range(100):
            for t in range(5):
                pr.increment_sent(
                    TurnToSend(
                        conversation_id=f"s{s}",
                        x_correlation_id=f"x-s{s}",
                        turn_index=t,
                        num_turns=5,
                    )
                )
        lc.mark_sending_complete()
        pr.freeze_sent_counts()
        st = pr.create_stats(lc)
        assert (
            st.sent_sessions == 100
            and st.total_session_turns == 500
            and st.requests_sent == 500
        )
        for s in range(100):
            for t in range(5):
                cr = Credit(
                    id=s * 5 + t,
                    phase=CreditPhase.PROFILING,
                    conversation_id=f"s{s}",
                    x_correlation_id=f"x-s{s}",
                    turn_index=t,
                    num_turns=5,
                    issued_at_ns=time.time_ns(),
                )
                if pr.increment_returned(cr.is_final_turn, cancelled=False):
                    pr.all_credits_returned_event.set()
        assert pr.all_credits_returned_event.is_set()
        st = pr.create_stats(lc)
        assert st.completed_sessions == 100
