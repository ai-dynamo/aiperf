# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Non-generating, exact-payload text decision adapters (enhancement #1536)."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, ClassVar

import orjson

from aiperf.common.finite import is_finite_value
from aiperf.common.models import (
    BaseResponseData,
    ErrorDetails,
    InferenceServerResponse,
    ParsedResponse,
    RequestInfo,
    RequestRecord,
)
from aiperf.endpoints.base_endpoint import BaseEndpoint


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _probability(value: Any) -> None:
    _require(
        type(value) in (float, int) and is_finite_value(value) and 0 <= value <= 1,
        "Invalid decision probability",
    )


def _identity(value: Any) -> tuple[type, Any]:
    _require(type(value) in (str, bool), "Choice values must be strings or booleans")
    return type(value), value


@dataclass(slots=True)
class DecisionResponseData(BaseResponseData):
    """Structured decisions; never expose their JSON as generated text."""

    body: dict[str, Any]
    """Complete server response, including actual usage and distributions."""


class _DecisionEndpoint(BaseEndpoint):
    """Shared transport handling, without translating either public contract."""

    systemone: ClassVar[bool] = False

    def format_payload(self, request_info: RequestInfo) -> dict[str, Any]:
        """Replay a complete dataset payload without merging generation defaults."""
        if request_info.payload_bytes is not None:
            body = orjson.loads(request_info.payload_bytes)
        else:
            _require(
                len(request_info.turns) == 1
                and request_info.turns[0].raw_payload is not None,
                "Decisions requires exactly one raw payload",
            )
            body = deepcopy(request_info.turns[0].raw_payload)
        self._validate_request(body)
        return body

    def get_endpoint_headers(self, request_info: RequestInfo) -> dict[str, str]:
        """Validate mmap/raw requests too: these bypass format_payload on send."""
        self.format_payload(request_info)
        return super().get_endpoint_headers(request_info)

    def _validate_request(self, body: Any) -> None:
        _require(isinstance(body, dict), "Decision payload must be an object")
        _require(
            "nvext" not in body, "nvext and native Decisions formats are not supported"
        )
        _require(
            isinstance(body.get("model"), str) and bool(body["model"]),
            "Explicit model is required",
        )
        _require(not body.get("stream", False), "Decisions must be non-streaming")
        _require(not body.get("images"), "This adapter qualifies text inputs only")
        evidence = body.get("state" if self.systemone else "input")
        _require(isinstance(evidence, (str, dict, list)), "Missing decision evidence")
        if not self.systemone and not isinstance(evidence, str):
            _require(
                isinstance(evidence, list) and bool(evidence),
                "OpenAI input must be text or user messages",
            )
            for message in evidence:
                _require(
                    isinstance(message, dict) and message.get("role") == "user",
                    "Only user text messages are supported",
                )
                content = message.get("content")
                _require(
                    isinstance(content, str)
                    or isinstance(content, list)
                    and bool(content)
                    and all(
                        isinstance(part, dict)
                        and part.get("type") == "input_text"
                        and isinstance(part.get("text"), str)
                        for part in content
                    ),
                    "Only text content is supported",
                )
        questions = body.get("questions")
        _require(
            isinstance(questions, dict if self.systemone else list) and bool(questions),
            "Missing ordered decision questions",
        )
        for question in questions.values() if self.systemone else questions:
            _require(
                isinstance(question, dict)
                and question.get("type")
                in ("noul" if self.systemone else "predicate", "choice", "score"),
                "Unsupported decision question",
            )

    def parse_response(self, response: InferenceServerResponse) -> ParsedResponse:
        """Validate the response envelope, typed outcomes, distributions and usage."""
        body = response.get_json()
        _require(
            isinstance(body, dict) and "error" not in body,
            "Expected successful decision JSON",
        )
        _require(
            isinstance(body.get("model"), str) and bool(body["model"]),
            "Missing decision model",
        )
        answers = body.get("answers")
        _require(
            isinstance(answers, dict if self.systemone else list) and bool(answers),
            "Missing decision answers",
        )
        refusals = 0
        for answer in answers.values() if self.systemone else answers:
            refusals += self._validate_answer(answer)
        usage = body.get("usage")
        if usage is not None:
            _require(isinstance(usage, dict), "Invalid decision usage")
            for key in ("input_tokens", "output_tokens"):
                _require(
                    type(usage.get(key)) is int and usage[key] >= 0,
                    "Invalid measured token count",
                )
            if "total_tokens" in usage:
                _require(
                    type(usage["total_tokens"]) is int
                    and usage["total_tokens"]
                    == usage["input_tokens"] + usage["output_tokens"],
                    "Inconsistent token total",
                )
            for detail_key, limit_key in (
                ("input_tokens_details", "input_tokens"),
                ("output_tokens_details", "output_tokens"),
            ):
                if detail_key in usage:
                    details = usage[detail_key]
                    _require(isinstance(details, dict), "Invalid usage details")
                    for count in details.values():
                        _require(
                            type(count) is int and 0 <= count <= usage[limit_key],
                            "Invalid nested measured token count",
                        )
        return ParsedResponse(
            perf_ns=response.perf_ns,
            data=DecisionResponseData(body=deepcopy(body)),
            usage=deepcopy(usage),
            metadata={
                "decision_question_count": len(answers),
                "decision_refusal_count": refusals,
            },
        )

    def _validate_answer(self, answer: Any) -> int:
        _require(isinstance(answer, dict), "Invalid decision answer")
        kind = answer.get("type")
        if not self.systemone:
            _require(
                answer.get("name") is None or isinstance(answer["name"], str),
                "Invalid answer name",
            )
            if kind == "refusal":
                _require(
                    not set(answer) - {"type", "name"},
                    "Refusal contains manufactured answer data",
                )
                return 1
        predicate = "noul" if self.systemone else "predicate"
        _require(kind in (predicate, "choice", "score"), "Invalid decision answer type")
        if kind == predicate:
            _probability(answer.get("noul" if self.systemone else "probability"))
            return 0
        _probability(answer.get("confidence"))
        values = answer.get("probabilities")
        _require(
            isinstance(values, dict if self.systemone else list) and bool(values),
            "Missing decision distribution",
        )
        if self.systemone:
            keys, probabilities = list(values), list(values.values())
        else:
            _require(
                all(isinstance(item, dict) for item in values),
                "Invalid probability entries",
            )
            keys = [item.get("value") for item in values]
            probabilities = [item.get("probability") for item in values]
        for probability in probabilities:
            _probability(probability)
        _require(
            abs(sum(probabilities) - 1) <= 1e-6,
            "Decision distribution is not normalized",
        )
        if kind == "choice":
            identities = [_identity(value) for value in keys]
            _require(
                len(set(identities)) == len(identities)
                and _identity(answer.get("choice")) in identities,
                "Invalid typed decision choice",
            )
        else:
            _require(
                keys
                == (
                    [str(i) for i in range(len(keys))]
                    if self.systemone
                    else list(range(len(keys)))
                )
                and all(
                    type(key) is str if self.systemone else type(key) is int
                    for key in keys
                ),
                "Invalid zero-based score ordinals",
            )
            if self.systemone:
                legend = answer.get("legend")
                _require(
                    isinstance(legend, dict)
                    and list(legend) == keys
                    and all(isinstance(label, str) for label in legend.values()),
                    "Invalid score legend",
                )
            else:
                _require(
                    all(isinstance(item.get("label"), str) for item in values),
                    "Missing score probability label",
                )
            score = answer.get("score")
            _require(
                type(score) in (int, float)
                and is_finite_value(score)
                and abs(score - sum(i * p for i, p in enumerate(probabilities)))
                <= 1e-6,
                "Score is not the expected ordinal",
            )
        return 0

    def extract_response_data(self, record: RequestRecord) -> list[ParsedResponse]:
        """Reconcile the actual request; retain malformed HTTP-200 wire evidence."""
        if record.error is not None:
            return []
        if record._parsed_responses_cache is not None:
            return record._parsed_responses_cache
        try:
            _require(
                record.request_info.payload_bytes is not None,
                "Decision validation requires retained request payload bytes",
            )
            request = orjson.loads(record.request_info.payload_bytes)
            self._validate_request(request)
            _require(
                len(record.responses) == 1, "Expected exactly one decision response"
            )
            parsed = self.parse_response(record.responses[0])
            self._correlate(request, parsed.data.body)
            record._parsed_responses_cache = [parsed]
        except (ValueError, TypeError, KeyError) as error:
            record.error = ErrorDetails.from_exception(error)
            record._parsed_responses_cache = []
        return record._parsed_responses_cache

    def _correlate(self, request: dict[str, Any], response: dict[str, Any]) -> None:
        questions, answers = request["questions"], response["answers"]
        _require(
            len(questions) == len(answers), "Decision answer count differs from request"
        )
        if self.systemone:
            _require(
                list(questions) == list(answers), "Question identifiers/order differ"
            )
            questions, answers = list(questions.values()), list(answers.values())
        for question, answer in zip(questions, answers, strict=True):
            if not self.systemone:
                _require(
                    answer.get("name") == question.get("name"),
                    "Question names/order differ",
                )
                if answer["type"] == "refusal":
                    continue
            _require(answer["type"] == question["type"], "Question type differs")
            if question["type"] == "choice":
                expected = (
                    list(question["criteria"])
                    if self.systemone
                    else [item["value"] for item in question["choices"]]
                )
                actual = (
                    list(answer["probabilities"])
                    if self.systemone
                    else [item["value"] for item in answer["probabilities"]]
                )
                _require(
                    [_identity(value) for value in actual]
                    == [_identity(value) for value in expected],
                    "Candidate type/order differs",
                )
            elif question["type"] == "score":
                levels = question["criteria" if self.systemone else "levels"]
                _require(
                    len(answer["probabilities"]) == len(levels),
                    "Score cardinality differs",
                )
                expected = (
                    levels if self.systemone else [level["label"] for level in levels]
                )
                actual = (
                    list(answer["legend"].values())
                    if self.systemone
                    else [item["label"] for item in answer["probabilities"]]
                )
                _require(actual == expected, "Score labels differ from request")

    def build_assistant_turn(self, record: RequestRecord) -> None:
        """Decisions are not generated assistant conversation text."""
        return None


class OpenAIDecisionsEndpoint(_DecisionEndpoint):
    """OpenAI Decisions schema at /v1/decisions, independent of serving engine."""


class SystemOneEndpoint(_DecisionEndpoint):
    """Jev System One schema at /v1/systemone, independent of serving engine."""

    systemone = True
