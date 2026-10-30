# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from unittest.mock import Mock

import orjson
import pytest

from aiperf.common.models import Turn
from aiperf.endpoints.decisions import OpenAIDecisionsEndpoint, SystemOneEndpoint
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType
from tests.unit.endpoints.conftest import (
    create_mock_response,
    create_model_endpoint,
    create_request_info,
)


@pytest.fixture(params=["decisions", "systemone"])
def case(request):
    dialect = request.param
    endpoint_type = (
        OpenAIDecisionsEndpoint if dialect == "decisions" else SystemOneEndpoint
    )
    endpoint = endpoint_type(create_model_endpoint(dialect))
    question = {"type": "predicate", "instructions": "Is it useful?"}
    answer = {"type": "predicate", "name": None, "probability": 0.8}
    body = {"model": "test-model", "input": "evidence", "questions": [question]}
    response = {"model": "test-model", "answers": [answer]}
    if dialect == "systemone":
        body = {
            "model": "test-model",
            "state": "evidence",
            "questions": {"q": {"type": "noul", "instructions": "Is it useful?"}},
        }
        response = {
            "model": "test-model",
            "answers": {"q": {"type": "noul", "noul": 0.8}},
        }
    return endpoint, body, response


def test_exact_raw_payload_and_no_mutation(case):
    endpoint, body, _ = case
    original = deepcopy(body)
    info = create_request_info(endpoint.model_endpoint, turns=[Turn(raw_payload=body)])
    assert endpoint.format_payload(info) == original
    assert body == original


def test_structured_response_is_not_text_and_usage_unknown(case):
    endpoint, _, body = case
    parsed = endpoint.parse_response(create_mock_response(json_data=body))
    assert parsed.data.body == body
    assert parsed.data.get_text() == ""
    assert parsed.usage is None
    assert parsed.metadata["decision_question_count"] == 1
    assert parsed.metadata["decision_refusal_count"] == 0


def test_measured_generated_usage_is_not_forced_to_zero(case):
    endpoint, _, body = case
    body["usage"] = {"input_tokens": 10, "output_tokens": 1}
    parsed = endpoint.parse_response(create_mock_response(json_data=body))
    assert parsed.usage.prompt_tokens == 10
    assert parsed.usage.completion_tokens == 1


@pytest.mark.parametrize(
    "selector", [{}, {"format": "oai"}, {"format": "sglang_native"}]
)
def test_rejects_all_nvext_envelopes(case, selector):
    endpoint, body, _ = case
    body["nvext"] = selector
    info = create_request_info(endpoint.model_endpoint, turns=[Turn(raw_payload=body)])
    with pytest.raises(ValueError, match="nvext"):
        endpoint.get_endpoint_headers(info)


@pytest.mark.parametrize(
    "bad", [None, {}, {"error": "failure"}, {"model": "x", "answers": []}]
)
def test_malformed_http_success_rejected(case, bad):
    endpoint, _, _ = case
    with pytest.raises(ValueError):
        endpoint.parse_response(create_mock_response(json_data=bad))


@pytest.mark.parametrize("value", [-0.1, 1.1, True, "0.8", float("inf"), float("nan")])
def test_invalid_probability_rejected(case, value):
    endpoint, _, body = case
    answers = body["answers"]
    answer = answers[0] if isinstance(answers, list) else answers["q"]
    answer["probability" if isinstance(answers, list) else "noul"] = value
    with pytest.raises(ValueError):
        endpoint.parse_response(create_mock_response(json_data=body))


def test_question_count_mismatch_becomes_record_error(case):
    endpoint, body, response = case
    if isinstance(body["questions"], list):
        body["questions"].append(deepcopy(body["questions"][0]))
    else:
        body["questions"]["q2"] = deepcopy(body["questions"]["q"])
    record = Mock(error=None, _parsed_responses_cache=None)
    record.request_info.payload_bytes = orjson.dumps(body)
    record.responses = [create_mock_response(json_data=response)]
    assert endpoint.extract_response_data(record) == []
    assert record.error is not None


@pytest.mark.parametrize("name", ["decisions", "systemone"])
def test_registry_non_generating_non_streaming(name):
    entry = plugins.get_entry(PluginType.ENDPOINT, name)
    assert entry.metadata["supports_streaming"] is False
    assert entry.metadata["produces_tokens"] is False
    assert entry.metadata["tokenizes_input"] is False


def test_complete_choice_and_score_correlation(case):
    endpoint, body, response = case
    questions = [
        {
            "type": "choice",
            "instructions": "Choose",
            "choices": [{"value": True}, {"value": "true"}],
        },
        {
            "type": "score",
            "instructions": "Rate",
            "levels": [{"label": "low"}, {"label": "high"}],
        },
    ]
    answers = [
        {
            "type": "choice",
            "name": "",
            "choice": True,
            "confidence": 0.4,
            "probabilities": [
                {"value": True, "probability": 0.8},
                {"value": "true", "probability": 0.2},
            ],
        },
        {
            "type": "score",
            "name": "",
            "score": 0.2,
            "confidence": 0.3,
            "probabilities": [
                {"value": 0, "label": "low", "probability": 0.8},
                {"value": 1, "label": "high", "probability": 0.2},
            ],
        },
    ]
    for question in questions:
        question["name"] = ""
    if endpoint.systemone:
        questions = {
            "q1": {"type": "choice", "criteria": {"true": "yes", "other": "no"}},
            "q2": {"type": "score", "criteria": ["low", "high"]},
        }
        answers = {
            "q1": {
                "type": "choice",
                "choice": "true",
                "confidence": 0.4,
                "probabilities": {"true": 0.8, "other": 0.2},
            },
            "q2": {
                "type": "score",
                "score": 0.2,
                "confidence": 0.3,
                "probabilities": {"0": 0.8, "1": 0.2},
                "legend": {"0": "low", "1": "high"},
            },
        }
    body["questions"], response["answers"] = questions, answers
    record = Mock(error=None, _parsed_responses_cache=None)
    record.request_info.payload_bytes = orjson.dumps(body)
    record.responses = [create_mock_response(json_data=response)]
    parsed = endpoint.extract_response_data(record)
    assert record.error is None
    assert len(parsed) == 1
    assert parsed[0].metadata["decision_question_count"] == 2
    assert endpoint.extract_response_data(record) is parsed
    assert endpoint.build_assistant_turn(record) is None
    assert parsed[0].data.get_text() == ""


def test_refusal_and_names_are_not_invented():
    endpoint = OpenAIDecisionsEndpoint(create_model_endpoint("decisions"))
    parsed = endpoint.parse_response(
        create_mock_response(
            json_data={
                "model": "test-model",
                "answers": [{"name": None, "type": "refusal"}],
            }
        )
    )
    assert parsed.metadata["decision_refusal_count"] == 1


def test_mmap_payload_and_user_text_messages():
    endpoint = OpenAIDecisionsEndpoint(create_model_endpoint("decisions"))
    body = {
        "model": "test-model",
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "evidence"}]}
        ],
        "questions": [{"type": "predicate", "instructions": "Useful?"}],
    }
    info = create_request_info(endpoint.model_endpoint)
    info.payload_bytes = orjson.dumps(body)
    assert endpoint.format_payload(info) == body
    assert endpoint.get_endpoint_headers(info) == {}


@pytest.mark.parametrize(
    "key,value", [("input_tokens", -1), ("output_tokens", True), ("total_tokens", 12)]
)
def test_invalid_measured_usage(case, key, value):
    endpoint, _, body = case
    body["usage"] = {
        "input_tokens": 10,
        "output_tokens": 1,
        "total_tokens": 11,
        key: value,
    }
    with pytest.raises(ValueError):
        endpoint.parse_response(create_mock_response(json_data=body))


@pytest.mark.parametrize("wrong", [False, True])
def test_missing_or_wrong_score_labels_rejected(case, wrong):
    endpoint, body, response = case
    if endpoint.systemone:
        body["questions"] = {"q": {"type": "score", "criteria": ["low", "high"]}}
        answer = {
            "type": "score",
            "score": 0.2,
            "confidence": 0.5,
            "probabilities": {"0": 0.8, "1": 0.2},
        }
        if wrong:
            answer["legend"] = {"0": "wrong", "1": "high"}
        response["answers"] = {"q": answer}
    else:
        body["questions"] = [
            {
                "type": "score",
                "instructions": "Rate",
                "levels": [{"label": "low"}, {"label": "high"}],
            }
        ]
        answer = {
            "type": "score",
            "name": None,
            "score": 0.2,
            "confidence": 0.5,
            "probabilities": [
                {"value": 0, "probability": 0.8},
                {"value": 1, "label": "high", "probability": 0.2},
            ],
        }
        if wrong:
            answer["probabilities"][0]["label"] = "wrong"
        response["answers"] = [answer]
    record = Mock(error=None, _parsed_responses_cache=None)
    record.request_info.payload_bytes = orjson.dumps(body)
    record.responses = [create_mock_response(json_data=response)]
    assert endpoint.extract_response_data(record) == []
    assert record.error is not None


def test_omitted_optional_answer_name_is_valid():
    endpoint = OpenAIDecisionsEndpoint(create_model_endpoint("decisions"))
    parsed = endpoint.parse_response(
        create_mock_response(
            json_data={
                "model": "test-model",
                "answers": [{"type": "predicate", "probability": 0.8}],
            }
        )
    )
    assert "name" not in parsed.data.body["answers"][0]


@pytest.mark.parametrize(
    "details",
    [
        {"input_tokens_details": {"cached_tokens": -1}},
        {"input_tokens_details": {"cached_tokens": True}},
        {"input_tokens_details": {"cache_write_tokens": 11}},
        {"output_tokens_details": {"reasoning_tokens": 2}},
    ],
)
def test_invalid_nested_token_usage_rejected(case, details):
    endpoint, _, body = case
    body["usage"] = {"input_tokens": 10, "output_tokens": 1, **details}
    with pytest.raises(ValueError):
        endpoint.parse_response(create_mock_response(json_data=body))
