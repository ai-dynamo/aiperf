# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adversarial probes against the session manager / worker dispatch / MetricInputs wire.

Validates the post-split invariants:

* ``RawPayloadSession`` <-> ``MetricInputs.payload_bytes is None`` coherence.
* ``ContentSession`` <-> inline ``payload_bytes`` set on the wire.
* ``MetricInputs`` lossless msgspec.msgpack round-trip including bytes.
* ``RawPayloadSession.advance_turn`` index bounds.
* Polymorphic FORK refcount / pinning guards against ``RawPayloadSession``.
* ``_enrich_request_record`` field projection is exhaustive and lossless.

Notes:

* MetricInputs is a msgspec.Struct since Phase 3a.3; ``payload_bytes`` is a
  ``bytes | None`` field (Phase 3c: msgpack on the wire), exposed via the
  ``payload_bytes_or_none`` property. The wire rides the payload as a
  length-prefixed msgpack ``bin`` span -- zero base64 inflation,
  binary-transparent.
* AIPerfBaseModel uses ``extra="allow"`` for the embedding Pydantic models,
  but MetricInputs itself (msgspec.Struct) does not preserve unknown fields.
* ``UserSessionManager.store`` is last-write-wins, even across session types --
  a ``create_raw_payload_session`` call silently replaces a ``ContentSession``
  on the same x_correlation_id without raising.
"""

from __future__ import annotations

import msgspec
import pytest
from pydantic import ValidationError
from pytest import param

from aiperf.common.enums import (
    ConversationBranchMode,
    ConversationContextMode,
    CreditPhase,
    ModelSelectionStrategy,
)
from aiperf.common.models import (
    Conversation,
    MetricInputs,
    RequestInfo,
    RequestRecord,
    Turn,
)
from aiperf.common.models.dataset_models import ConversationBranchInfo, Text
from aiperf.common.models.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.plugin.enums import EndpointType
from aiperf.workers.inference_client import InferenceClient
from aiperf.workers.session_manager import (
    RawPayloadSession,
    UserSessionManager,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _model_endpoint() -> ModelEndpointInfo:
    return ModelEndpointInfo(
        models=ModelListInfo(
            models=[ModelInfo(name="test-model")],
            model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
        ),
        endpoint=EndpointInfo(
            type=EndpointType.CHAT,
            base_url="http://localhost:8000/v1/test",
        ),
    )


def _base_metric_inputs(**overrides) -> MetricInputs:
    """Minimal valid MetricInputs that callers can selectively override."""
    base = dict(
        credit_num=0,
        credit_phase=CreditPhase.PROFILING,
        conversation_id="conv-id",
        turn_index=0,
        x_request_id="req-1",
        x_correlation_id="corr-1",
    )
    base.update(overrides)
    return MetricInputs(**base)


def _base_request_info(**overrides) -> RequestInfo:
    base = dict(
        model_endpoint=_model_endpoint(),
        credit_num=0,
        credit_phase=CreditPhase.PROFILING,
        conversation_id="conv-id",
        turn_index=0,
        x_request_id="req-1",
        x_correlation_id="corr-1",
    )
    base.update(overrides)
    return RequestInfo(**base)


@pytest.fixture
def manager() -> UserSessionManager:
    return UserSessionManager()


@pytest.fixture
def conv_two_turns() -> Conversation:
    return Conversation(
        session_id="cv1",
        turns=[
            Turn(role="user", texts=[Text(contents=["t1"])]),
            Turn(role="user", texts=[Text(contents=["t2"])]),
        ],
    )


# ---------------------------------------------------------------------------
# RawPayloadSession boundaries
# ---------------------------------------------------------------------------


class TestRawPayloadSessionBoundaries:
    """Boundary probes on the format-specific RawPayloadSession.advance_turn."""

    def test_num_turns_zero_constructs_but_cannot_advance(self) -> None:
        """num_turns=0 passes ge=0 validation but advance_turn(0) must raise.

        A zero-turn session is degenerate but constructible -- the guard
        lives in advance_turn, not the constructor. Verifies the bound
        message exposes the actual num_turns.
        """
        s = RawPayloadSession(x_correlation_id="x", num_turns=0, conversation_id="c")
        with pytest.raises(ValueError, match="0 turns"):
            s.advance_turn(0)

    @pytest.mark.parametrize(
        ("bad_index", "match"),
        [
            param(-1, "negative", id="negative"),
            param(-100, "negative", id="very_negative"),
            param(3, "out of range", id="equal_to_num_turns"),
            param(100, "out of range", id="far_beyond"),
        ],
    )
    def test_advance_turn_rejects_bad_index(self, bad_index: int, match: str) -> None:
        s = RawPayloadSession(x_correlation_id="x", num_turns=3, conversation_id="c")
        with pytest.raises(ValueError, match=match):
            s.advance_turn(bad_index)

    def test_advance_turn_idempotent_same_index(self) -> None:
        """Advancing to the same index twice is a pure cursor write, no state corruption."""
        s = RawPayloadSession(x_correlation_id="x", num_turns=2, conversation_id="c")
        s.advance_turn(1)
        assert s.turn_index == 1
        s.advance_turn(1)
        assert s.turn_index == 1

    def test_advance_turn_can_go_backwards(self) -> None:
        """advance_turn does not enforce monotonicity. Backwards is legal."""
        s = RawPayloadSession(x_correlation_id="x", num_turns=3, conversation_id="c")
        s.advance_turn(2)
        s.advance_turn(0)
        assert s.turn_index == 0


# ---------------------------------------------------------------------------
# Polymorphic guards on FORK-pinning paths
# ---------------------------------------------------------------------------


class TestForkPinningPolymorphism:
    """pin/release/evict must distinguish ContentSession from RawPayloadSession."""

    def test_pin_for_fork_child_on_raw_payload_raises_typeerror(
        self, manager: UserSessionManager
    ) -> None:
        """FORK + PAYLOAD_BYTES is refused at format-selection but the
        defense-in-depth guard at session_manager.py:372 must raise if a
        raw-payload session somehow lands in the cache. Otherwise FORK
        accounting would silently no-op.
        """
        manager.store(
            "xc",
            RawPayloadSession(x_correlation_id="xc", num_turns=1, conversation_id="c"),
        )
        with pytest.raises(TypeError, match="not a.*ContentSession"):
            manager.pin_for_fork_child("xc")

    def test_pin_for_fork_child_missing_session_raises_keyerror(
        self, manager: UserSessionManager
    ) -> None:
        with pytest.raises(KeyError, match="parent already evicted"):
            manager.pin_for_fork_child("nope")

    def test_release_fork_child_on_raw_payload_is_silent_noop(
        self, manager: UserSessionManager
    ) -> None:
        """release races against eviction; it must not raise on a
        non-ContentSession (defensive contract per session_manager.py:419)."""
        manager.store(
            "y",
            RawPayloadSession(x_correlation_id="y", num_turns=1, conversation_id="c"),
        )
        # Should be a no-op; session stays.
        manager.release_fork_child("y")
        assert isinstance(manager.get("y"), RawPayloadSession)

    def test_evict_if_unpinned_falls_through_for_raw_payload(
        self, manager: UserSessionManager
    ) -> None:
        """A RawPayloadSession has no fork_refcount; evict_if_unpinned
        treats it as a plain evict. Verifies session_manager.py:443-445."""
        manager.store(
            "y",
            RawPayloadSession(x_correlation_id="y", num_turns=1, conversation_id="c"),
        )
        manager.evict_if_unpinned("y")
        assert manager.get("y") is None

    def test_evict_if_unpinned_unknown_correlation_is_noop(
        self, manager: UserSessionManager
    ) -> None:
        # Must not raise on a missing session.
        manager.evict_if_unpinned("not-there")
        assert manager.get("not-there") is None

    def test_seed_from_parent_returns_silently_if_either_side_raw(
        self, manager: UserSessionManager, conv_two_turns: Conversation
    ) -> None:
        """seed_from_parent must defensively short-circuit when either side
        is a RawPayloadSession -- documented at session_manager.py:400-403."""
        parent = manager.create_content_session(
            x_correlation_id="parent", conversation=conv_two_turns, num_turns=2
        )
        parent.turn_list = list(conv_two_turns.turns)
        manager.store(
            "child",
            RawPayloadSession(
                x_correlation_id="child", num_turns=1, conversation_id="cv1"
            ),
        )
        manager.seed_from_parent("child", "parent")
        # Child is RawPayloadSession -- has no turn_list to mutate; must not raise.
        assert isinstance(manager.get("child"), RawPayloadSession)


# ---------------------------------------------------------------------------
# Refcount semantics on ContentSession
# ---------------------------------------------------------------------------


class TestForkRefcount:
    """Off-by-one and pending-eviction interactions on ContentSession."""

    def test_pin_release_balances_to_zero(
        self, manager: UserSessionManager, conv_two_turns: Conversation
    ) -> None:
        s = manager.create_content_session(
            x_correlation_id="p", conversation=conv_two_turns, num_turns=2
        )
        for _ in range(5):
            manager.pin_for_fork_child("p")
        assert s.fork_refcount == 5
        for _ in range(5):
            manager.release_fork_child("p")
        assert s.fork_refcount == 0

    def test_release_below_zero_is_floored(
        self, manager: UserSessionManager, conv_two_turns: Conversation
    ) -> None:
        """release races against eviction in practice and must NOT go
        negative -- the floor at session_manager.py:421 guards this."""
        s = manager.create_content_session(
            x_correlation_id="p", conversation=conv_two_turns, num_turns=2
        )
        manager.release_fork_child("p")
        manager.release_fork_child("p")
        assert s.fork_refcount == 0

    def test_pending_fork_eviction_triggers_on_last_release(
        self, manager: UserSessionManager, conv_two_turns: Conversation
    ) -> None:
        """When pending_fork_eviction is set, the session evicts the
        moment refcount drops to zero -- session_manager.py:422-423."""
        s = manager.create_content_session(
            x_correlation_id="p", conversation=conv_two_turns, num_turns=2
        )
        manager.pin_for_fork_child("p")
        manager.pin_for_fork_child("p")
        s.pending_fork_eviction = True
        manager.release_fork_child("p")
        assert manager.get("p") is s, "first release: refcount=1, must NOT evict"
        manager.release_fork_child("p")
        assert manager.get("p") is None, "last release: refcount=0 + pending => evict"

    def test_evict_if_unpinned_respects_pending_flag(
        self, manager: UserSessionManager, conv_two_turns: Conversation
    ) -> None:
        """Even at refcount==0, pending_fork_eviction keeps the session
        resident so the orchestrator's about-to-arrive children can seed
        from it -- session_manager.py:448-449."""
        s = manager.create_content_session(
            x_correlation_id="p", conversation=conv_two_turns, num_turns=2
        )
        s.pending_fork_eviction = True
        manager.evict_if_unpinned("p")
        assert manager.get("p") is s

    def test_fork_parent_with_zero_turns_is_constructible(
        self, manager: UserSessionManager
    ) -> None:
        """A FORK-declaring conversation with empty turn list must not crash
        create_content_session even though no FORK child can ever be seeded
        from it. _compute_is_fork_parent reads conversation.branches, not turns."""
        conv = Conversation(
            session_id="cv0",
            turns=[],
            branches=[
                ConversationBranchInfo(
                    branch_id="b1",
                    mode=ConversationBranchMode.FORK,
                    child_conversation_ids=["child-1"],
                )
            ],
        )
        s = manager.create_content_session(
            x_correlation_id="p0", conversation=conv, num_turns=0
        )
        assert s.is_fork_parent is True
        assert s.num_turns == 0


# ---------------------------------------------------------------------------
# UserSessionManager storage semantics
# ---------------------------------------------------------------------------


class TestSessionStorageSemantics:
    """Race / upsert semantics on the cache map keyed by x_correlation_id."""

    def test_create_content_then_raw_with_same_corr_raises(
        self, manager: UserSessionManager, conv_two_turns: Conversation
    ) -> None:
        """``UserSessionManager.store`` refuses to silently replace an
        existing session with one of a different concrete type — that
        indicates a sticky-router uniqueness violation. Same-type
        re-stores are still permitted as a legitimate refresh."""
        manager.create_content_session(
            x_correlation_id="xc", conversation=conv_two_turns, num_turns=1
        )
        with pytest.raises(RuntimeError, match="sticky-router uniqueness violation"):
            manager.create_raw_payload_session(
                x_correlation_id="xc", conversation_id="c", num_turns=1
            )
        # Existing session is untouched.
        from aiperf.workers.session_manager import ContentSession

        assert isinstance(manager.get("xc"), ContentSession)

    def test_evict_unknown_correlation_is_noop(
        self, manager: UserSessionManager
    ) -> None:
        manager.evict("nope")  # pop(..., None)
        assert manager.get("nope") is None

    def test_set_default_context_mode_propagates_to_new_sessions(
        self, manager: UserSessionManager, conv_two_turns: Conversation
    ) -> None:
        manager.set_default_context_mode(
            ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES
        )
        # Conversation with no override + no FORK -> inherits dataset default.
        s = manager.create_content_session(
            x_correlation_id="xc", conversation=conv_two_turns, num_turns=1
        )
        assert s.context_mode == ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES


# ---------------------------------------------------------------------------
# MetricInputs validation
# ---------------------------------------------------------------------------


class TestMetricInputsValidation:
    """Field-level invariants on the wire model.

    msgspec.Struct does not natively express ``ge=0`` constraints (no Pydantic
    field-level validators), and the Phase 3a.3 migration intentionally drops
    those gates -- the worker contract never produces negative routing values.
    Negative-input rejection tests from the Pydantic era are removed; the
    construction smoke-tests below pin only what msgspec actually enforces.
    """

    def test_extreme_credit_num_accepted(self) -> None:
        """No upper bound on credit_num; 64-bit-ish values must pass."""
        m = _base_metric_inputs(credit_num=2**62)
        assert m.credit_num == 2**62

    def test_missing_required_field_raises(self) -> None:
        """msgspec.Struct raises ``TypeError`` (not ``pydantic.ValidationError``)
        on missing required kwargs."""
        with pytest.raises(TypeError):
            MetricInputs(credit_num=0)  # type: ignore[call-arg]

    def test_extra_fields_rejected_by_msgspec(self) -> None:
        """msgspec.Struct rejects unknown kwargs at construction (unlike the
        Pydantic-era ``extra='allow'`` behaviour)."""
        with pytest.raises(TypeError):
            _base_metric_inputs(mystery_field="hello")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# MetricInputs msgpack round-trip (the wire contract)
# ---------------------------------------------------------------------------


class TestMetricInputsWireRoundtrip:
    """The whole point of MetricInputs is the lossless wire trip."""

    @pytest.mark.parametrize(
        "payload",
        [
            param(None, id="none_payload"),
            param(b'{"messages":[]}', id="utf8_json_bytes"),
            param(b'{"\xe2\x9c\x93":1}', id="utf8_multibyte"),
            param(b'"' + b"x" * 4094 + b'"', id="medium_size_string"),
            # msgpack is binary-transparent: non-JSON bytes round-trip too.
            param(b"\x00\x01\xff\xfe", id="non_json_binary"),
            param(b"", id="empty_bytes"),
        ],
    )
    def test_lossless_roundtrip_for_payloads(self, payload: bytes | None) -> None:
        """Every payload byte-string round-trips byte-for-byte via msgspec.msgpack."""
        m = _base_metric_inputs(payload_bytes=payload)
        wire = msgspec.msgpack.encode(m)
        m2 = msgspec.msgpack.decode(wire, type=MetricInputs)
        assert m2.payload_bytes_or_none == payload

    def test_payload_rides_verbatim_not_base64(self) -> None:
        """payload_bytes rides the msgpack wire as a length-prefixed bin span --
        no base64, byte-identical."""
        payload = b'{"messages":[{"role":"user","content":"x"}]}'
        mi = MetricInputs(
            credit_num=0,
            credit_phase=CreditPhase.PROFILING,
            conversation_id="c",
            turn_index=0,
            x_request_id="r",
            x_correlation_id="x",
            payload_bytes=payload,
        )
        wire = msgspec.msgpack.encode(mi)
        # Payload bytes appear verbatim inside the msgpack envelope.
        assert payload in wire
        rt = msgspec.msgpack.decode(wire, type=MetricInputs)
        assert rt.payload_bytes == payload
        assert rt.payload_bytes_or_none == payload

    def test_none_distinguishable_from_payload(self) -> None:
        """``payload_bytes=None`` -> wire-side None; an actual payload -> raw bytes."""
        m_none = _base_metric_inputs(payload_bytes=None)
        m_val = _base_metric_inputs(payload_bytes=b'{"a":1}')
        rt_none = msgspec.msgpack.decode(
            msgspec.msgpack.encode(m_none), type=MetricInputs
        )
        rt_val = msgspec.msgpack.decode(
            msgspec.msgpack.encode(m_val), type=MetricInputs
        )
        assert rt_none.payload_bytes_or_none is None
        assert rt_val.payload_bytes_or_none == b'{"a":1}'

    def test_all_fields_lossless(self) -> None:
        """Every field that the worker populates must survive the wire trip."""
        m = MetricInputs(
            credit_num=42,
            credit_phase=CreditPhase.WARMUP,
            conversation_id="abc",
            turn_index=7,
            x_request_id="req-x",
            x_correlation_id="corr-y",
            credit_issued_ns=1_700_000_000_000_000_000,
            agent_depth=3,
            parent_correlation_id="papa",
            payload_bytes=b'{"hi":1}',
        )
        rt = msgspec.msgpack.decode(msgspec.msgpack.encode(m), type=MetricInputs)
        assert rt.credit_num == m.credit_num
        assert rt.credit_phase == m.credit_phase
        assert rt.conversation_id == m.conversation_id
        assert rt.turn_index == m.turn_index
        assert rt.x_request_id == m.x_request_id
        assert rt.x_correlation_id == m.x_correlation_id
        assert rt.credit_issued_ns == m.credit_issued_ns
        assert rt.agent_depth == m.agent_depth
        assert rt.parent_correlation_id == m.parent_correlation_id
        assert rt.payload_bytes_or_none == m.payload_bytes_or_none

    def test_credit_phase_enum_roundtrips(self) -> None:
        """CreditPhase is a CaseInsensitiveStrEnum -- msgspec encodes the
        underlying string value and decodes back to the enum."""
        for phase in (CreditPhase.WARMUP, CreditPhase.PROFILING):
            m = _base_metric_inputs(credit_phase=phase)
            wire = msgspec.msgpack.encode(m)
            rt = msgspec.msgpack.decode(wire, type=MetricInputs)
            assert rt.credit_phase == phase
            assert isinstance(rt.credit_phase, CreditPhase)

    def test_request_record_with_metric_inputs_roundtrip(self) -> None:
        """End-to-end: ``RequestRecord`` (Task 3a.4: now msgspec.Struct) ferries
        its embedded ``MetricInputs`` Struct through ``msgspec.msgpack`` natively."""
        mi = _base_metric_inputs(payload_bytes=b'{"a":1}')
        rec = RequestRecord(metric_inputs=mi, model_name="m", status=200)
        wire = msgspec.msgpack.encode(rec)
        rt = msgspec.msgpack.decode(wire, type=RequestRecord)
        assert rt.metric_inputs is not None
        assert rt.metric_inputs.credit_num == mi.credit_num
        assert rt.metric_inputs.payload_bytes_or_none == b'{"a":1}'
        # Wire form: payload bytes appear verbatim (length-prefixed bin span).
        assert b'{"a":1}' in wire

    def test_request_record_without_metric_inputs_roundtrip(self) -> None:
        rec = RequestRecord(metric_inputs=None)
        rt = msgspec.msgpack.decode(msgspec.msgpack.encode(rec), type=RequestRecord)
        assert rt.metric_inputs is None

    def test_request_record_with_none_payload_bytes_roundtrip(self) -> None:
        """``RequestRecord.metric_inputs.payload_bytes=None`` round-trips
        through msgspec.msgpack as None."""
        mi = _base_metric_inputs(payload_bytes=None)
        rec = RequestRecord(metric_inputs=mi)
        rt = msgspec.msgpack.decode(msgspec.msgpack.encode(rec), type=RequestRecord)
        assert rt.metric_inputs is not None
        assert rt.metric_inputs.payload_bytes_or_none is None


# ---------------------------------------------------------------------------
# _enrich_request_record projection
# ---------------------------------------------------------------------------


class TestEnrichRequestRecord:
    """Pure projection: every routing field on RequestInfo must flow onto MetricInputs."""

    def test_full_field_projection_with_inline_bytes(self) -> None:
        ri = _base_request_info(
            credit_num=5,
            credit_phase=CreditPhase.WARMUP,
            conversation_id="conv-x",
            turn_index=2,
            x_request_id="req-x",
            x_correlation_id="corr-x",
            credit_issued_ns=12345,
            agent_depth=3,
            parent_correlation_id="papa",
            payload_bytes=b'{"hi":1}',
            from_mmap=False,
        )
        rec = RequestRecord()
        InferenceClient._enrich_request_record(rec, ri)
        mi = rec.metric_inputs
        assert mi is not None
        assert mi.credit_num == 5
        assert mi.credit_phase == CreditPhase.WARMUP
        assert mi.conversation_id == "conv-x"
        assert mi.turn_index == 2
        assert mi.x_request_id == "req-x"
        assert mi.x_correlation_id == "corr-x"
        assert mi.credit_issued_ns == 12345
        assert mi.agent_depth == 3
        assert mi.parent_correlation_id == "papa"
        assert mi.payload_bytes_or_none == b'{"hi":1}'

    @pytest.mark.parametrize(
        "payload_bytes",
        [
            param(b'{"hi":1}', id="payload_present"),
            param(None, id="payload_none"),
        ],
    )
    def test_from_mmap_drops_payload_bytes(self, payload_bytes: bytes | None) -> None:
        """from_mmap=True -> wire-side MetricInputs.payload_bytes must be None
        (the records-process resolves via its own mmap client)."""
        ri = _base_request_info(payload_bytes=payload_bytes, from_mmap=True)
        rec = RequestRecord()
        InferenceClient._enrich_request_record(rec, ri)
        assert rec.metric_inputs is not None
        assert rec.metric_inputs.payload_bytes_or_none is None

    def test_root_session_no_parent_correlation_id(self) -> None:
        """Root sessions: parent_correlation_id=None must flow through, not be coerced."""
        ri = _base_request_info(agent_depth=0, parent_correlation_id=None)
        rec = RequestRecord()
        InferenceClient._enrich_request_record(rec, ri)
        assert rec.metric_inputs is not None
        assert rec.metric_inputs.parent_correlation_id is None
        assert rec.metric_inputs.agent_depth == 0

    def test_does_not_mutate_request_info(self) -> None:
        """_enrich_request_record is a pure projection -- RequestInfo unchanged."""
        ri = _base_request_info(
            credit_num=9,
            payload_bytes=b"payload",
            from_mmap=True,
        )
        before = ri.model_dump()
        rec = RequestRecord()
        InferenceClient._enrich_request_record(rec, ri)
        assert ri.model_dump() == before

    def test_enrich_then_roundtrip_lossless(self) -> None:
        """End-to-end: enrich produces a wire-safe record."""
        ri = _base_request_info(
            credit_num=11,
            payload_bytes=b'{"x":1}',
            from_mmap=False,
            credit_issued_ns=99,
            agent_depth=2,
            parent_correlation_id="pa",
        )
        rec = RequestRecord()
        InferenceClient._enrich_request_record(rec, ri)
        rt = msgspec.json.decode(msgspec.json.encode(rec), type=RequestRecord)
        assert rt.metric_inputs is not None
        assert rt.metric_inputs.credit_num == 11
        assert rt.metric_inputs.payload_bytes_or_none == b'{"x":1}'
        assert rt.metric_inputs.credit_issued_ns == 99
        assert rt.metric_inputs.agent_depth == 2
        assert rt.metric_inputs.parent_correlation_id == "pa"


# ---------------------------------------------------------------------------
# RequestInfo invariants exercised by the worker
# ---------------------------------------------------------------------------


class TestRequestInfoFieldEnforcement:
    """Field-level Pydantic constraints the worker depends on."""

    def test_negative_credit_num_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _base_request_info(credit_num=-1)

    def test_negative_credit_issued_ns_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _base_request_info(credit_issued_ns=-1)

    def test_url_index_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _base_request_info(url_index=-1)

    def test_x_request_id_must_be_string(self) -> None:
        """Worker contract: x_request_id is `str`, never int. Pydantic enforces
        strict type checking here -- ints don't coerce to str."""
        with pytest.raises(ValidationError, match="x_request_id"):
            _base_request_info(x_request_id=12345)  # type: ignore[arg-type]
