# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``--end-conversation-on-truncation`` against the mock server.

The mock reports ``finish_reason=length`` whenever it generates exactly
``max_tokens``, and a ``qwen`` model name streams ~250 reasoning tokens first, so
a small ``--osl`` truncates every turn mid-reasoning.
"""

from pathlib import Path

import pytest
from pytest import param

from tests.harness.utils import AIPerfCLI, AIPerfMockServer
from tests.integration.conftest import IntegrationTestDefaults as defaults

DAG_FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "dag" / "full.dag.jsonl"
)
NUM_CONVERSATIONS = 6
TURNS = 3


@pytest.mark.integration
@pytest.mark.asyncio
class TestEndConversationOnTruncation:
    async def _run(
        self,
        cli: AIPerfCLI,
        server: AIPerfMockServer,
        *,
        model: str,
        extra_flags: str,
    ):
        return await cli.run(
            f"""
            aiperf profile \
                --model {model} \
                --url {server.url} \
                --endpoint-type chat \
                --conversation-num {NUM_CONVERSATIONS} \
                --conversation-turn-mean {TURNS} \
                --conversation-turn-stddev 0 \
                --osl 20 \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --export-level raw \
                --ui {defaults.ui} \
                {extra_flags}
            """
        )

    @pytest.mark.parametrize(
        "model,streaming",
        [
            param("test-model", "", id="content-non-streaming"),
            param("qwen-test", "--streaming", id="reasoning-only-streaming"),
        ],
    )  # fmt: skip
    async def test_truncated_turn_ends_each_conversation(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        model: str,
        streaming: str,
    ):
        result = await self._run(
            cli,
            aiperf_mock_server,
            model=model,
            extra_flags=f"--end-conversation-on-truncation {streaming}",
        )

        assert result.request_count == NUM_CONVERSATIONS
        assert {r.metadata.turn_index for r in result.raw_records} == {0}
        assert result.json.conversations_ended_on_truncation == NUM_CONVERSATIONS
        assert result.json.turns_skipped_on_truncation == NUM_CONVERSATIONS * (
            TURNS - 1
        )

    async def test_option_off_sends_every_turn(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        result = await self._run(
            cli,
            aiperf_mock_server,
            model="qwen-test",
            extra_flags="--streaming",
        )

        assert result.request_count == NUM_CONVERSATIONS * TURNS
        assert result.json.conversations_ended_on_truncation is None

    async def test_dag_dataset_rejects_option(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        result = await cli.run(
            f"""
            aiperf profile \
                --model test-model \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --input-file {DAG_FIXTURE} \
                --custom-dataset-type dag_jsonl \
                --num-conversations 1 \
                --concurrency 1 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui} \
                --end-conversation-on-truncation
            """,
            assert_success=False,
        )

        assert result.exit_code != 0
        assert "not supported for DAG datasets" in result.log
