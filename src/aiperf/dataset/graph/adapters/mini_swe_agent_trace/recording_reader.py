# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed reader for Agent Trace Replay performance-replay recordings (`*.json`, `*.json.gz`).

Schema source (agent-trace-benchmark repo):
    src/minisweagent/recording/recorder.py  -- event envelope and payloads
    src/minisweagent/recording/replay.py    -- which fields drive a replay

One recording file is one agent task run. Its `events` list is a flat,
chronologically ordered log of `run_start` / `step_start` / `model_call` /
`tool_call` / `step_end` / `run_end` (plus `pinchbench_grade` on graded
PinchBench tasks). Only `model_call` and `tool_call` carry replay-relevant
data.

Two properties of the format decide the whole lowering:

* Every `model_call` carries a `provider_request` holding the EXACT request
  body that was sent (`recorder.py::record_model_call` copies it whenever the
  wrapper supplies one, and `wrappers.py` always does). Agent Trace Replay's own replay
  re-sends `provider_request["messages"]` verbatim rather than anything the
  agent computed (`replay.py::query`), so the full wire traffic of a replay is
  determined by the file before the run starts. Nothing needs to be simulated.
* `timestamp` is stamped by `recorder.py::record_event` when the event is
  RECORDED -- that is, after the call returned. It is the event END time; the
  start is `timestamp - duration_ns`. Getting this backwards would shift every
  node by its own duration and turn inter-call gaps negative.

`provider_request` also carries the recording endpoint's `api_base`/`api_key`.
Those are sanitized to placeholders in the shipped corpus, and this reader
drops them: the replay dispatches to the run's endpoint, never the recorded
one.
"""

from __future__ import annotations

import gzip
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import orjson
from pydantic import BaseModel, ConfigDict, Field

from aiperf.common.finite import FiniteFloat

# Wire discriminator prefix. The shipped corpus is `mini-swe-agent-recording-1.0`;
# matching on the prefix keeps a 1.x bump loadable rather than undetectable.
AGENT_TRACE_RECORDING_FORMAT_PREFIX = "mini-swe-agent-recording-"

MODEL_CALL = "model_call"
TOOL_CALL = "tool_call"

# `InterruptAgentFlow` and its subclasses (`minisweagent/exceptions.py`) are
# agent CONTROL FLOW, not command failures: the recorder stores any raised
# exception in a tool_call's `error`, but these are raised on INSPECTING the
# command's output, after it ran to completion (see
# `environments/docker.py`, "Raises Submitted if the output indicates task
# completion"). Treating them as failures would drop the terminal submit
# command of every graded trace.
AGENT_CONTROL_FLOW_ERRORS = frozenset(
    {
        "InterruptAgentFlow",
        "Submitted",
        "LimitsExceeded",
        "ReplayExhausted",
        "UserInterruption",
        "FormatError",
    }
)


class AgentTraceRecordingError(ValueError):
    """Raised when an Agent Trace Replay recording cannot be converted into a ParsedGraph."""


class EmptyAgentTraceRecordingError(AgentTraceRecordingError):
    """Raised when a recording (or directory of them) holds no usable model calls."""


class RecordedUsage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    prompt_tokens: int | None = Field(
        default=None, ge=0, description="Recorded input tokens for this call."
    )
    completion_tokens: int | None = Field(
        default=None,
        ge=0,
        description="Recorded output tokens; the wire generation cap for replay. "
        "Zero is legal (an aborted or empty turn) and upgrades to 1 at lowering.",
    )


class ResponseBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    usage: RecordedUsage | None = Field(
        default=None, description="Provider usage block, when the server returned one."
    )


class ResponseExtra(BaseModel):
    model_config = ConfigDict(extra="ignore")

    response: ResponseBody | None = Field(
        default=None, description="Raw provider response envelope."
    )


class ResponseMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    extra: ResponseExtra | None = Field(
        default=None, description="Recorder-attached response metadata."
    )

    @property
    def completion_tokens(self) -> int | None:
        """Recorded output-token count, or None when the server omitted usage."""
        usage = (
            self.extra.response.usage if self.extra and self.extra.response else None
        )
        return usage.completion_tokens if usage else None


class ProviderRequest(BaseModel):
    """The verbatim request body recorded for one model call.

    `api_base` / `api_key` / `timeout` / `max_retries` are deliberately not
    modeled: they describe the RECORDING endpoint and client, and the replay
    dispatches to the run's own endpoint.
    """

    model_config = ConfigDict(extra="ignore")

    messages: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Exact message array sent for this call, replayed verbatim.",
    )
    tools: list[dict[str, Any]] | None = Field(
        default=None,
        description="OpenAI-compatible tool definitions; part of the prompt's "
        "token footprint, so replayed even though no tool ever executes.",
    )
    model: str | None = Field(
        default=None,
        description="LiteLLM model string the recording was captured against "
        "(e.g. 'openai/qwen3.6:27b'), not necessarily an endpoint model id.",
    )
    temperature: float | None = Field(
        default=None, ge=0, description="Recorded sampling temperature."
    )
    top_p: float | None = Field(
        default=None, ge=0, le=1, description="Recorded nucleus-sampling p."
    )
    max_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Recorded request-level generation cap; superseded at replay "
        "by the per-call recorded output length.",
    )


class RecordedEvent(BaseModel):
    """One entry in a recording's `events` list."""

    model_config = ConfigDict(extra="ignore")

    id: int = Field(ge=0, description="Monotonic event index within the recording.")
    type: str = Field(description="Event kind, e.g. 'model_call' or 'tool_call'.")
    timestamp: FiniteFloat = Field(
        gt=0,
        description="Unix seconds at which the event was RECORDED, i.e. the "
        "event's END. The start is timestamp - duration_ns. Non-finite would "
        "silently poison every derived offset and edge delay.",
    )
    step: int | None = Field(
        default=None, ge=0, description="Agent step index; absent on run-level events."
    )
    duration_ns: int | None = Field(
        default=None,
        ge=0,
        description="Measured duration; absent on run-level events.",
    )
    provider_request: ProviderRequest | None = Field(
        default=None, description="Recorded request body (model_call only)."
    )
    response_message: ResponseMessage | None = Field(
        default=None, description="Recorded assistant response (model_call only)."
    )
    action: dict[str, Any] | None = Field(
        default=None,
        description="Executed command payload (tool_call only); its 'command' "
        "key is the shell command the recorded agent ran.",
    )
    error: dict[str, Any] | None = Field(
        default=None,
        description="Recorded terminal signal for this event, when there was "
        "one. NOT necessarily a failure: the same key carries agent CONTROL "
        "FLOW, notably the `Submitted` type a recorded agent raises after a "
        "command RAN successfully (see `command_completed`). Read the `type` "
        "key to tell the two apart rather than treating presence as failure.",
    )

    @property
    def duration_s(self) -> float:
        """Measured duration in seconds; 0.0 when the event carries none."""
        return (self.duration_ns or 0) / 1e9

    @property
    def start_unix_s(self) -> float:
        """Unix seconds at which the event began."""
        return self.timestamp - self.duration_s

    @property
    def command_completed(self) -> bool:
        """Did this tool_call's command run to completion?

        True when no error was recorded, and also when the recorded error is an
        agent control-flow signal -- those are raised on the command's output,
        so the command itself ran.
        """
        if self.error is None:
            return True
        return self.error.get("type") in AGENT_CONTROL_FLOW_ERRORS


class RecordingMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore")

    instance_id: str | None = Field(
        default=None, description="Task identifier, used as the trace id when present."
    )
    benchmark: str | None = Field(
        default=None, description="Originating benchmark family (swebench, pinchbench)."
    )
    model_name: str | None = Field(
        default=None, description="Model the trajectory was captured from."
    )
    docker_image: str | None = Field(
        default=None, description="Task-specific Docker image the trajectory ran in."
    )


class AgentTraceRecording(BaseModel):
    """One recorded agent task run."""

    model_config = ConfigDict(extra="ignore")

    format: str = Field(
        description="Schema discriminator, 'mini-swe-agent-recording-N.N'."
    )
    metadata: RecordingMetadata = Field(
        default_factory=RecordingMetadata, description="Task-level capture metadata."
    )
    events: list[RecordedEvent] = Field(
        default_factory=list, description="Chronological event log."
    )

    def model_calls(self) -> list[RecordedEvent]:
        """Successful `model_call` events in recorded order.

        Mirrors `validation.py::require_successful_model_calls`: Agent Trace Replay
        refuses to replay a recording containing a failed call at all, rather
        than skipping it, because a missing response would desynchronize the
        recorded response stream from the agent's step sequence. The same
        refusal applies here -- silently dropping a call would silently drop
        its prompt growth from the workload.
        """
        calls = [event for event in self.events if event.type == MODEL_CALL]
        for index, event in enumerate(calls, start=1):
            if event.error is not None or event.response_message is None:
                raise AgentTraceRecordingError(
                    f"model call {index} (step {event.step}) did not succeed; "
                    "Agent Trace Replay recordings must contain only successful model calls"
                )
            if event.provider_request is None:
                raise AgentTraceRecordingError(
                    f"model call {index} (step {event.step}) has no provider_request; "
                    "the recorded request body is required to replay it verbatim"
                )
        return calls

    def tool_calls_between(
        self, after_id: int, before_id: int | None
    ) -> list[RecordedEvent]:
        """Tool calls whose command ran, recorded after one model call.

        Bounded above by ``before_id`` (the next model call), or unbounded when
        it is None -- that is the trailing case, where the trajectory's final
        submit/finalize command follows the last model call with no successor.

        A single agent step may batch several tool calls; they are returned in
        recorded order so the replay executes them exactly as the capture did.
        Only genuinely failed commands are skipped; see
        :attr:`RecordedEvent.command_completed`.
        """
        return [
            event
            for event in self.events
            if event.type == TOOL_CALL
            and event.id > after_id
            and (before_id is None or event.id < before_id)
            and event.command_completed
        ]


def _open(path: Path):
    """Open a plain or gzipped recording for binary reading."""
    return (
        gzip.open(path, "rb") if path.name.lower().endswith(".gz") else path.open("rb")
    )


def read_recording(path: Path) -> AgentTraceRecording:
    """Decode one recording file (`.json` or `.json.gz`)."""
    try:
        with _open(path) as handle:
            payload = orjson.loads(handle.read())
    except (OSError, EOFError, orjson.JSONDecodeError) as exc:
        raise AgentTraceRecordingError(f"{path}: unreadable recording ({exc})") from exc
    if not isinstance(payload, dict):
        raise AgentTraceRecordingError(f"{path}: recording root is not a JSON object")
    return AgentTraceRecording.model_validate(payload)


def is_recording_file(path: Path) -> bool:
    """Cheap sniff: does ``path`` look like an Agent Trace Replay recording?

    Reads only the leading bytes -- a recording's `format` key is emitted
    first by the writer, so the discriminator is available without decoding a
    multi-megabyte message history. Never raises: an unreadable, truncated, or
    non-JSON file is simply not ours.
    """
    if not path.is_file():
        return False
    if not path.name.lower().endswith((".json", ".json.gz")):
        return False
    try:
        with _open(path) as handle:
            head = handle.read(4096)
    except (OSError, EOFError):
        return False
    if b'"format"' not in head:
        return False
    return AGENT_TRACE_RECORDING_FORMAT_PREFIX.encode() in head


def discover_recordings(path: Path) -> list[Path]:
    """Recording files for ``path``: itself, or every recording directly inside it.

    A directory is scanned one level deep and sorted by name so a corpus lowers
    in a stable order across hosts. `manifest.json` and any other non-recording
    JSON in the same directory is skipped by the content sniff rather than by
    name, so a renamed manifest cannot be mistaken for a trajectory.
    """
    if path.is_file():
        return [path] if is_recording_file(path) else []
    if not path.is_dir():
        return []
    return sorted(
        (child for child in path.iterdir() if is_recording_file(child)),
        key=lambda child: child.name,
    )


def iter_recordings(path: Path) -> Iterator[tuple[Path, AgentTraceRecording]]:
    """Yield ``(path, recording)`` for every recording under ``path``, in order."""
    for recording_path in discover_recordings(path):
        yield recording_path, read_recording(recording_path)
