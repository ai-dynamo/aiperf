# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cache-bust plumbing tests for ConversationSource / SampledSession.

Covers Slice 2 of the cache-bust subsystem: the marker text + target enum
ride on a ``SampledSession`` and flow through ``build_first_turn`` into
``TurnToSend``. Verifies the keyword-only signatures on
``start_branch_child`` and ``start_pre_session_child`` accept the new
fields with sensible defaults so existing callers keep working.
"""

import pytest

from aiperf.common.enums import CacheBustTarget, ConversationBranchMode
from aiperf.common.models import ConversationMetadata, DatasetMetadata, TurnMetadata
from aiperf.plugin import plugins
from aiperf.plugin.enums import DatasetSamplingStrategy, PluginType
from aiperf.timing.conversation_source import ConversationSource, SampledSession


def _mk_source(ds: DatasetMetadata) -> ConversationSource:
    SamplerClass = plugins.get_class(PluginType.DATASET_SAMPLER, ds.sampling_strategy)
    sampler = SamplerClass(
        conversation_ids=[c.conversation_id for c in ds.conversations],
    )
    return ConversationSource(ds, sampler)


@pytest.fixture
def src() -> ConversationSource:
    ds = DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id="root-conv",
                turns=[TurnMetadata(timestamp_ms=0.0)],
                agent_depth=0,
            ),
            ConversationMetadata(
                conversation_id="child-conv",
                turns=[TurnMetadata(timestamp_ms=0.0)],
                agent_depth=1,
                parent_conversation_id="root-conv",
                is_root=False,
            ),
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    return _mk_source(ds)


class TestSampledSessionCacheBustDefaults:
    def test_defaults_when_constructed_directly(self):
        meta = ConversationMetadata(
            conversation_id="c",
            turns=[TurnMetadata(timestamp_ms=0.0)],
        )
        s = SampledSession(
            conversation_id="c",
            metadata=meta,
            x_correlation_id="x",
        )
        assert s.cache_bust_marker is None
        assert s.cache_bust_target is CacheBustTarget.NONE

    def test_explicit_fields_set(self):
        meta = ConversationMetadata(
            conversation_id="c",
            turns=[TurnMetadata(timestamp_ms=0.0)],
        )
        s = SampledSession(
            conversation_id="c",
            metadata=meta,
            x_correlation_id="x",
            cache_bust_marker="\n<!-- cb:s -->\n",
            cache_bust_target=CacheBustTarget.SYSTEM_SUFFIX,
        )
        assert s.cache_bust_marker == "\n<!-- cb:s -->\n"
        assert s.cache_bust_target is CacheBustTarget.SYSTEM_SUFFIX

    def test_build_first_turn_carries_cache_bust(self):
        meta = ConversationMetadata(
            conversation_id="c",
            turns=[TurnMetadata(timestamp_ms=0.0)],
        )
        s = SampledSession(
            conversation_id="c",
            metadata=meta,
            x_correlation_id="x",
            cache_bust_marker="\n<!-- cb:first -->\n",
            cache_bust_target=CacheBustTarget.FIRST_TURN_PREFIX,
        )
        tts = s.build_first_turn()
        assert tts.cache_bust_marker == "\n<!-- cb:first -->\n"
        assert tts.cache_bust_target is CacheBustTarget.FIRST_TURN_PREFIX


class TestStartBranchChildCacheBust:
    def test_default_no_cache_bust(self, src: ConversationSource):
        child = src.start_branch_child(
            parent_correlation_id="parent-corr",
            child_conversation_id="child-conv",
            agent_depth=1,
        )
        assert child.cache_bust_marker is None
        assert child.cache_bust_target is CacheBustTarget.NONE

    def test_caller_mints_marker_for_spawn_child(self, src: ConversationSource):
        child = src.start_branch_child(
            parent_correlation_id="parent-corr",
            child_conversation_id="child-conv",
            agent_depth=1,
            branch_mode=ConversationBranchMode.SPAWN,
            cache_bust_marker="\n<!-- cb:spawn-1 -->\n",
            cache_bust_target=CacheBustTarget.SYSTEM_PREFIX,
        )
        assert child.cache_bust_marker == "\n<!-- cb:spawn-1 -->\n"
        assert child.cache_bust_target is CacheBustTarget.SYSTEM_PREFIX
        # Marker flows through to the TurnToSend so the worker sees it.
        tts = child.build_first_turn()
        assert tts.cache_bust_marker == "\n<!-- cb:spawn-1 -->\n"
        assert tts.cache_bust_target is CacheBustTarget.SYSTEM_PREFIX


class TestStartPreSessionChildCacheBust:
    def test_default_no_cache_bust(self, src: ConversationSource):
        child = src.start_pre_session_child(child_conversation_id="child-conv")
        assert child.cache_bust_marker is None
        assert child.cache_bust_target is CacheBustTarget.NONE

    def test_caller_mints_marker_for_pre_session(self, src: ConversationSource):
        child = src.start_pre_session_child(
            child_conversation_id="child-conv",
            cache_bust_marker="\n<!-- cb:pre -->\n",
            cache_bust_target=CacheBustTarget.FIRST_TURN_SUFFIX,
        )
        assert child.cache_bust_marker == "\n<!-- cb:pre -->\n"
        assert child.cache_bust_target is CacheBustTarget.FIRST_TURN_SUFFIX
        tts = child.build_first_turn()
        assert tts.cache_bust_marker == "\n<!-- cb:pre -->\n"
        assert tts.cache_bust_target is CacheBustTarget.FIRST_TURN_SUFFIX
