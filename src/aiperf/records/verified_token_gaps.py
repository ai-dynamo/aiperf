# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from itertools import groupby, pairwise
from operator import itemgetter

from aiperf.common.models.record_models import RequestRecord, SSEMessage
from aiperf.common.types import JsonObject


def _matches_one_content_token(text: object, logprobs: object) -> bool:
    """Bind the server's sampled token to the text actually delivered in this event."""
    if not isinstance(text, str) or not text or not isinstance(logprobs, dict):
        return False
    tokens = logprobs.get("content")
    if not isinstance(tokens, list) or len(tokens) != 1 or logprobs.get("refusal"):
        return False
    token = tokens[0]
    if not isinstance(token, dict) or not isinstance(token.get("token"), str):
        return False
    octets = token.get("bytes")
    return (
        isinstance(octets, list)
        and all(type(octet) is int and 0 <= octet <= 255 for octet in octets)
        and bytes(octets) == text.encode("utf-8")
    )


def _plain_text_choice(chunk: JsonObject) -> tuple[JsonObject, JsonObject] | None:
    choices = chunk.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        return None
    choice = choices[0]
    if not isinstance(choice, dict) or type(choice.get("index")) is not int:
        return None
    delta = choice.get("delta")
    if choice["index"] != 0 or not isinstance(delta, dict):
        return None
    if any(value for key, value in delta.items() if key not in {"role", "content"}):
        return None
    return choice, delta


def _completed_sse_events(record: RequestRecord) -> list[SSEMessage]:
    end_perf_ns = record.end_perf_ns
    if end_perf_ns is None:
        return []
    events: list[SSEMessage] = []
    previous_ns = record.start_perf_ns
    for response in record.responses or ():
        if not isinstance(response, SSEMessage):
            return []
        if not previous_ns <= response.perf_ns <= end_perf_ns:
            return []
        previous_ns = response.perf_ns
        if response.extract_data_content():
            events.append(response)
    if not events or events.pop().extract_data_content() != "[DONE]":
        return []
    return events


def _chat_content_events(record: RequestRecord) -> list[tuple[int, bool]]:
    events: list[tuple[int, bool]] = []
    finished = False
    for response in _completed_sse_events(record):
        chunk = response.get_json()
        if (
            not isinstance(chunk, dict)
            or chunk.get("object") != "chat.completion.chunk"
        ):
            return []
        if chunk.get("choices") == [] and isinstance(chunk.get("usage"), dict):
            continue
        parsed = _plain_text_choice(chunk)
        if finished or parsed is None:
            return []
        choice, delta = parsed
        reason = choice.get("finish_reason")
        if reason is not None and reason not in ("stop", "length"):
            return []
        finished = reason is not None
        text, logprobs = delta.get("content"), choice.get("logprobs")
        if text in (None, "") and logprobs is None and (finished or delta.get("role")):
            continue
        events.append((response.perf_ns, _matches_one_content_token(text, logprobs)))
    return events if finished else []


def observe_chat_token_gaps(record: RequestRecord) -> list[int]:
    """Read token evidence from completed, single-choice text Chat Completions SSE."""
    events = _chat_content_events(record)
    # SSE events completed in the same read share one timestamp. A bundled or
    # unverified event stays on the timeline as a barrier to joining its neighbors.
    token_times: list[int | None] = []
    for perf_ns, group in groupby(events, key=itemgetter(0)):
        members = list(group)
        token_times.append(perf_ns if len(members) == 1 and members[0][1] else None)
    return [
        end - start
        for start, end in pairwise(token_times)
        if start is not None and end is not None
    ]
