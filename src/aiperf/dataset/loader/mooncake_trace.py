# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from aiperf.common.enums import ConversationContextMode
from aiperf.common.models import Turn
from aiperf.dataset.loader.base_loader import LoaderProbeData
from aiperf.dataset.loader.base_trace_loader import BaseTraceDatasetLoader
from aiperf.dataset.loader.models import MooncakeTrace
from aiperf.dataset.loader.speed_bench import is_speed_bench_row


class MooncakeTraceDatasetLoader(BaseTraceDatasetLoader[MooncakeTrace]):
    """A dataset loader that loads Mooncake trace data from a file.

    Loads Mooncake trace data from a file and converts the data into
    a list of conversations for dataset manager.

    Each line in the file represents a single trace entry and will be
    converted to a separate conversation with a unique session ID.

    Example:
    Fixed schedule version
    ```json
    {"timestamp": 1000, "input_length": 300, "output_length": 40, "hash_ids": [123, 456]}
    ```

    Multi-turn version
    ```json
    {"session_id": "abc-123", "input_length": 300, "output_length": 40},
    {"session_id": "abc-123", "delay": 2, "input_length": 150, "output_length": 20}
    ```
    """

    @classmethod
    def can_load(
        cls, data: LoaderProbeData | None = None, filename: str | Path | None = None
    ) -> bool:
        """Check if this loader can handle the given data format.

        For mooncake trace data, simply validate the data against the MooncakeTrace model.
        This will handle all of the validation logic for the different input combinations.
        """
        if data is None:
            return False
        if is_speed_bench_row(data):
            return False

        try:
            MooncakeTrace.model_validate(data)
            return True
        except ValidationError:
            return False

    # ------------------------------------------------------------------
    # Template-method hooks (see BaseTraceDatasetLoader.load_dataset)
    # ------------------------------------------------------------------

    def _parse_trace(self, record: dict) -> MooncakeTrace:
        return MooncakeTrace.model_validate(record)

    def _group_traces(
        self, items: list[MooncakeTrace]
    ) -> dict[str, list[MooncakeTrace]]:
        self._warn_ignored_output_length(items)
        data: dict[str, list[MooncakeTrace]] = defaultdict(list)
        for trace in items:
            session_id = trace.session_id or self.session_id_generator.next()
            data[session_id].append(trace)
        return dict(data)

    @staticmethod
    def _endpoint_produces_tokens(endpoint_type: str) -> bool:
        """Whether an endpoint returns generated tokens (cached per endpoint)."""
        from aiperf.plugin import plugins

        return bool(plugins.get_endpoint_metadata(endpoint_type).produces_tokens)

    def _warn_ignored_output_length(self, items: list[MooncakeTrace]) -> None:
        """Warn once per endpoint about output_length that cannot apply.

        ``output_length`` is one of the four native Mooncake fields, so real
        traces carry it on every row -- including rows routed to an endpoint
        that generates no tokens (embeddings, rankings, image generation).
        Rather than reject those rows or log per request, drop the value in
        :meth:`_build_turn` and account for it once here, so the user learns
        the field was ignored without the count being buried in per-request
        noise.
        """
        ignored: dict[str, int] = defaultdict(int)
        for trace in items:
            if (
                trace.endpoint_type is not None
                and trace.output_length is not None
                and not self._endpoint_produces_tokens(trace.endpoint_type)
            ):
                ignored[trace.endpoint_type] += 1

        for endpoint_type, count in sorted(ignored.items()):
            self.warning(
                f"{count:,} row(s) routed to '{endpoint_type}' carry "
                f"'output_length'; that endpoint generates no tokens, so the "
                f"value does not apply and is ignored."
            )

    # ------------------------------------------------------------------
    # Conversation-building hooks
    # ------------------------------------------------------------------

    def _infer_context_mode(
        self, traces: list[MooncakeTrace]
    ) -> ConversationContextMode | None:
        """Auto-detect MESSAGE_ARRAY_WITH_RESPONSES when all traces are self-contained.

        Self-contained traces (pre-built `messages` or verbatim `payload`) bypass
        endpoint formatting and need to be replayed verbatim. Mixed sessions that
        combine self-contained traces with synthesized prompts, or mix `messages`
        and `payload` modes, are rejected.
        """
        msg_trace_count = sum(1 for trace in traces if trace.messages is not None)
        payload_trace_count = sum(1 for trace in traces if trace.payload is not None)
        self_contained_count = msg_trace_count + payload_trace_count

        if msg_trace_count > 0 and payload_trace_count > 0:
            raise ValueError(
                f"mooncake trace: mixed session contains {msg_trace_count} "
                f"`messages` trace(s) and {payload_trace_count} `payload` "
                f"trace(s); each session must use exactly one mode. Split "
                f"the offending sessions or convert all entries to a single "
                f"self-contained mode."
            )
        if self_contained_count == len(traces) and self_contained_count > 0:
            return ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES
        if self_contained_count > 0:
            raise ValueError(
                "Mixed Mooncake sessions with both raw `messages`/`payload` and synthesized prompts are unsupported."
            )
        return None

    def _get_text_input(self, trace: MooncakeTrace) -> str | None:
        if trace.messages is not None or trace.payload is not None:
            return ""
        return trace.text_input

    def _build_turn(self, trace: MooncakeTrace, prompt: str) -> Turn:
        # Verbatim payload/messages turns must honor --inter-turn-delay-cap-seconds
        # too: clamp the recorded delay via the shared DelayCapTracker, matching
        # both v1 and the synthesized-prompt branch (super()._build_turn, which
        # clamps in BaseTraceDatasetLoader). Without this the cap was silently
        # ignored for self-contained Mooncake sessions.
        if trace.payload is not None:
            return Turn(
                timestamp=trace.timestamp,
                delay=self._delay_cap_tracker.clamp(trace.delay),
                max_tokens=trace.output_length,
                raw_payload=trace.payload,
                extra_body=trace.extra,
            )
        if trace.messages is not None:
            return Turn(
                timestamp=trace.timestamp,
                delay=self._delay_cap_tracker.clamp(trace.delay),
                max_tokens=self._effective_max_tokens(trace),
                raw_messages=trace.messages,
                raw_tools=trace.tools,
                extra_body=trace.extra,
                endpoint_type=trace.endpoint_type,
            )
        turn = super()._build_turn(trace, prompt)
        if trace.extra is not None:
            turn.extra_body = trace.extra
        turn.endpoint_type = trace.endpoint_type
        turn.max_tokens = self._effective_max_tokens(trace)
        return turn

    def _effective_max_tokens(self, trace: MooncakeTrace) -> int | None:
        """Return the row's output_length, or None when the endpoint ignores it.

        Endpoints that generate no tokens reject a max_tokens they were handed
        (``EmbeddingsEndpoint`` logs an error for every such request). Dropping
        it here keeps that path silent; the count is reported once at load time
        by :meth:`_warn_ignored_output_length`.
        """
        if trace.endpoint_type is None or trace.output_length is None:
            return trace.output_length
        if self._endpoint_produces_tokens(trace.endpoint_type):
            return trace.output_length
        return None

    # ------------------------------------------------------------------
    # Synthesis hooks
    # ------------------------------------------------------------------

    def _synthesis_exclude_fields(self) -> frozenset[str]:
        return frozenset({"type"})

    def _reconstruct_traces(
        self, originals: list[MooncakeTrace], synth_dicts: list[dict[str, Any]]
    ) -> list[MooncakeTrace]:
        return [MooncakeTrace.model_validate(t) for t in synth_dicts]
