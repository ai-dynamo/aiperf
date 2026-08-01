"""Tests for dataset model markers and metadata projection."""


def test_conversation_metadata_propagates_is_orchestrator():
    from aiperf.common.models.dataset_models import Conversation

    conv = Conversation(session_id="start", turns=[], is_orchestrator=True)
    meta = conv.metadata()
    assert conv.is_orchestrator is True
    assert meta.is_orchestrator is True


def test_conversation_is_orchestrator_defaults_false():
    from aiperf.common.models.dataset_models import Conversation

    conv = Conversation(session_id="c", turns=[])
    assert conv.is_orchestrator is False
    assert conv.metadata().is_orchestrator is False
