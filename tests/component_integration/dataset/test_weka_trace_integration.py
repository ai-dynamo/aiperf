# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end: WekaTraceLoader -> DatasetMetadata -> validate_for_orchestrator_v1 passes."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aiperf.common.models import DatasetMetadata
from aiperf.common.validators.orchestrator_v1 import validate_for_orchestrator_v1
from aiperf.dataset.loader.weka_trace import WekaTraceLoader
from aiperf.plugin.enums import DatasetSamplingStrategy

FIXTURES = Path(__file__).parents[2] / "fixtures" / "weka_traces"


pytestmark = pytest.mark.component_integration


def _mk_user_config():
    uc = MagicMock()
    uc.input.random_seed = 0
    uc.input.fixed_schedule_start_offset = None
    uc.input.fixed_schedule_end_offset = None
    uc.input.ignore_trace_delays = False
    uc.input.use_think_time_only = False
    uc.input.use_end_to_start_delays = False
    uc.input.synthesis.max_isl = None
    uc.input.synthesis.max_osl = None
    uc.input.max_context_length = None
    uc.input.synthesis.should_synthesize.return_value = False
    uc.input.prompt.input_tokens.block_size = None
    uc.tokenizer.trust_remote_code = False
    uc.tokenizer.revision = None
    uc.tokenizer.name = "test-tok"
    uc.endpoint.model_names = ["claude-opus-4-5-20251101", "claude-haiku-4-5-20251001"]
    uc.loadgen.inter_turn_delay_cap_seconds = None
    return uc


def test_weka_trace_end_to_end_validates_for_orchestrator_v1(monkeypatch):
    uc = _mk_user_config()
    loader = WekaTraceLoader(
        filename=str(FIXTURES / "one_subagent.json"), user_config=uc
    )
    monkeypatch.setattr(
        loader, "synthesize_prompts_from_hash_ids", lambda rs: {r.key: "p" for r in rs}
    )
    pg = MagicMock()
    pg._corpus_size = 10000
    pg._tokenized_corpus = list(range(10000))
    pg.tokenizer.decode = lambda tokens: f"decoded-{len(tokens)}"
    loader.prompt_generator = pg
    loader._tokenizer_name = "t"
    loader._trust_remote_code = False
    loader._tokenizer_revision = None
    loader._block_size = 64

    convs = loader.convert_to_conversations(loader.load_dataset())
    md = DatasetMetadata(
        conversations=[c.to_metadata() for c in convs],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    # Should not raise.
    validate_for_orchestrator_v1(md)

    parent_md = next(c for c in md.conversations if c.conversation_id == "trace_sa")
    child_md = next(
        c for c in md.conversations if c.conversation_id == "trace_sa::sa:agent_001"
    )
    assert len(parent_md.branches) == 1
    assert parent_md.branches[0].child_conversation_ids == ["trace_sa::sa:agent_001"]
    assert len(parent_md.turns[1].prerequisites) == 1
    assert (
        parent_md.turns[1].prerequisites[0].branch_id == parent_md.branches[0].branch_id
    )
    assert len(child_md.turns) == 1
