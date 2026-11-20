# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from typing import Any
from unittest.mock import Mock, patch

import orjson
import pytest
from pytest import param

from aiperf.common.models import Turn
from aiperf.endpoints.decisions import OpenAIDecisionsEndpoint, SystemOneEndpoint
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType
from tests.unit.endpoints.conftest import (
    create_mock_response,
    create_model_endpoint,
    create_request_info,
)

DecisionCase = tuple[
    OpenAIDecisionsEndpoint | SystemOneEndpoint, dict[str, Any], dict[str, Any]
]


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


def create_record(body: dict, response: dict) -> Mock:
    record = Mock(error=None, _parsed_responses_cache=None)
    record.request_info.payload_bytes = orjson.dumps(body)
    record.responses = [create_mock_response(json_data=response)]
    return record


def test_exact_raw_payload_and_no_mutation(case):
    endpoint, body, _ = case
    original = deepcopy(body)
    info = create_request_info(endpoint.model_endpoint, turns=[Turn(raw_payload=body)])
    assert endpoint.format_payload(info) == original
    assert body == original


def test_raw_header_validation_does_not_copy_payload(case: DecisionCase) -> None:
    endpoint, body, _ = case
    info = create_request_info(endpoint.model_endpoint, turns=[Turn(raw_payload=body)])
    with patch(
        "aiperf.endpoints.decisions.deepcopy",
        side_effect=AssertionError("Unexpected payload copy"),
    ):
        assert endpoint.get_endpoint_headers(info) == {}
    copied = endpoint.format_payload(info)
    assert copied == body
    assert copied is not body
    assert copied["questions"] is not body["questions"]


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
    record = create_record(body, response)
    assert endpoint.extract_response_data(record) == []
    assert record.error is not None


@pytest.mark.parametrize("name", ["decisions", "systemone"])
def test_registry_non_generating_non_streaming(name):
    entry = plugins.get_entry(PluginType.ENDPOINT, name)
    assert entry.metadata["supports_streaming"] is False
    assert entry.metadata["produces_tokens"] is False
    assert entry.metadata["tokenizes_input"] is False
    assert entry.metadata["requires_request_payload"] is True


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
    record = create_record(body, response)
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
    record = create_record(body, response)
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


@pytest.mark.parametrize("score", [1.3, 0.65, float("nan"), float("inf")])
def test_three_level_score_is_expected_zero_based_ordinal(case, score):
    endpoint, _, response = case
    labels = ["low", "medium", "high"]
    probabilities = [0.2, 0.3, 0.5]
    answer = {
        "type": "score",
        "score": score,
        "confidence": 0.5,
        "probabilities": [
            {"value": i, "label": label, "probability": probability}
            for i, (label, probability) in enumerate(
                zip(labels, probabilities, strict=True)
            )
        ],
    }
    if endpoint.systemone:
        answer["probabilities"] = {str(i): p for i, p in enumerate(probabilities)}
        answer["legend"] = {str(i): label for i, label in enumerate(labels)}
    response["answers"] = {"q": answer} if endpoint.systemone else [answer]
    wire = create_mock_response(json_data=response)
    if score == 1.3:
        assert endpoint.parse_response(wire).data.body == response
    else:
        with pytest.raises(ValueError, match="expected ordinal"):
            endpoint.parse_response(wire)


@pytest.mark.parametrize(
    "values,choice",
    [
        ([1, "true"], True),
        (["true", "false"], "true"),
        ([True, "true"], 1),
        ([True, True], True),
        ([True], True),
        ([True, "true", "extra"], True),
    ],
)
def test_openai_choice_candidate_mismatch_is_record_error(
    values: list[str | bool | int], choice: str | bool | int
) -> None:
    endpoint = OpenAIDecisionsEndpoint(create_model_endpoint("decisions"))
    body = {
        "model": "test-model",
        "input": "evidence",
        "questions": [
            {"type": "choice", "choices": [{"value": True}, {"value": "true"}]}
        ],
    }
    response = {
        "model": "test-model",
        "answers": [
            {
                "type": "choice",
                "choice": choice,
                "confidence": 0.5,
                "probabilities": [
                    {"value": value, "probability": 0.5} for value in values
                ],
            }
        ],
    }
    record = create_record(body, response)
    assert endpoint.extract_response_data(record) == []
    assert record.error is not None


@pytest.mark.parametrize(
    "fault,error",
    [
        param(None, None, id="reordered"),
        param("position_score", "expected ordinal", id="position-score"),
        param("label", "Score labels", id="wrong-label"),
        param("duplicate", "zero-based score ordinals", id="duplicate"),
        param("gap", "zero-based score ordinals", id="missing-ordinal"),
        param("boolean", "zero-based score ordinals", id="boolean"),
        param("float", "zero-based score ordinals", id="float"),
        param("negative", "zero-based score ordinals", id="negative"),
    ],
)  # fmt: skip
def test_openai_score_distribution_binds_values_to_ordinals(
    fault: str | None, error: str | None
) -> None:
    endpoint = OpenAIDecisionsEndpoint(create_model_endpoint("decisions"))
    labels = ["low", "medium", "high"]
    probabilities = [0.2, 0.3, 0.5]
    values = [
        {"value": i, "label": labels[i], "probability": probabilities[i]}
        for i in (2, 0, 1)
    ]
    body = {
        "model": "test-model",
        "input": "evidence",
        "questions": [
            {"type": "score", "levels": [{"label": label} for label in labels]}
        ],
    }
    answer = {
        "type": "score",
        "score": 0.8 if fault == "position_score" else 1.3,
        "confidence": 0.5,
        "probabilities": values,
    }
    if fault == "label":
        values[0]["label"] = "low"
    elif fault in ("duplicate", "gap", "boolean", "float", "negative"):
        values[1]["value"] = {
            "duplicate": 2,
            "gap": 3,
            "boolean": False,
            "float": 0.0,
            "negative": -1,
        }[fault]
    record = create_record(body, {"model": "test-model", "answers": [answer]})
    parsed = endpoint.extract_response_data(record)
    if error is None:
        assert record.error is None
        assert parsed[0].data.body["answers"][0]["probabilities"] == values
    else:
        assert parsed == []
        assert error in record.error.message


def test_refusal_cannot_contain_manufactured_answer_data():
    endpoint = OpenAIDecisionsEndpoint(create_model_endpoint("decisions"))
    with pytest.raises(ValueError, match="manufactured answer"):
        endpoint.parse_response(
            create_mock_response(
                json_data={
                    "model": "test-model",
                    "answers": [{"type": "refusal", "probability": 0.0}],
                }
            )
        )


def test_refusal_name_must_match_request():
    endpoint = OpenAIDecisionsEndpoint(create_model_endpoint("decisions"))
    body = {
        "model": "test-model",
        "input": "evidence",
        "questions": [{"type": "predicate", "name": "expected"}],
    }
    response = {
        "model": "test-model",
        "answers": [{"type": "refusal", "name": "wrong"}],
    }
    record = create_record(body, response)
    assert endpoint.extract_response_data(record) == []
    assert "names/order" in record.error.message


@pytest.mark.parametrize("field", ["images", "assets"])
def test_image_request_fields_rejected_before_dispatch(
    case: DecisionCase, field: str
) -> None:
    endpoint, body, _ = case
    body[field] = {"photo": {"type": "image", "format": "png", "data_base64": "AA=="}}
    info = create_request_info(endpoint.model_endpoint, turns=[Turn(raw_payload=body)])
    with pytest.raises(ValueError, match="text inputs only"):
        endpoint.get_endpoint_headers(info)


@pytest.mark.parametrize("kind", ["predicate", "choice", "score"])
@pytest.mark.parametrize(
    "instructions,valid",
    [
        param("Useful?", True, id="string"),
        param([{"type": "input_text", "text": "Useful?"}], True, id="text-parts"),
        param([{"type": "input_image", "image_url": "data:image/png;base64,AA=="}], False, id="image"),
        param([{"type": "input_text", "text": "Useful?"}, {"type": "input_image", "image_url": "https://example.com/image.png"}], False, id="mixed"),
    ],
)  # fmt: skip
def test_question_instructions_remain_text_only(
    kind: str, instructions: Any, valid: bool
) -> None:
    endpoint = OpenAIDecisionsEndpoint(create_model_endpoint("decisions"))
    body = {
        "model": "test-model",
        "input": "evidence",
        "questions": [{"type": kind, "instructions": instructions}],
    }
    info = create_request_info(endpoint.model_endpoint, turns=[Turn(raw_payload=body)])
    if valid:
        assert endpoint.get_endpoint_headers(info) == {}
    else:
        with pytest.raises(ValueError, match="text content"):
            endpoint.get_endpoint_headers(info)


@pytest.mark.parametrize("usage", [None, [], "unknown", 0])
def test_explicit_non_object_usage_is_record_error(
    case: DecisionCase, usage: Any
) -> None:
    endpoint, body, response = case
    response["usage"] = usage
    record = create_record(body, response)
    assert endpoint.extract_response_data(record) == []
    assert "Invalid decision usage" in record.error.message


def test_nested_response_copy_failure_is_record_error(case: DecisionCase) -> None:
    endpoint, body, response = case
    wire = orjson.dumps(response)
    wire = wire[:-1] + b',"extra":' + b"[" * 600 + b"0" + b"]" * 600 + b"}"
    decoded = orjson.loads(wire)
    record = create_record(body, response)
    record.responses = [create_mock_response(json_data=decoded)]
    assert endpoint.extract_response_data(record) == []
    assert record.error.type == "RecursionError"
    assert record._parsed_responses_cache == []
    assert record.responses[0].get_json() is decoded


def test_openai_choice_probabilities_can_be_reordered() -> None:
    endpoint = OpenAIDecisionsEndpoint(create_model_endpoint("decisions"))
    body = {
        "model": "test-model",
        "input": "evidence",
        "questions": [
            {"type": "choice", "choices": [{"value": True}, {"value": "true"}]}
        ],
    }
    response = {
        "model": "test-model",
        "answers": [
            {
                "type": "choice",
                "choice": True,
                "confidence": 0.5,
                "probabilities": [
                    {"value": "true", "probability": 0.2},
                    {"value": True, "probability": 0.8},
                ],
            }
        ],
    }
    record = create_record(body, response)
    assert endpoint.extract_response_data(record)[0].data.body == response
    assert record.error is None


def test_explicit_zero_usage_remains_measured(case):
    endpoint, _, response = case
    response["usage"] = {"input_tokens": 0, "output_tokens": 0}
    parsed = endpoint.parse_response(create_mock_response(json_data=response))
    assert parsed.usage.prompt_tokens == 0
    assert parsed.usage.completion_tokens == 0


@pytest.mark.parametrize("missing_context", [False, True])
def test_missing_request_context_or_payload_is_record_error(case, missing_context):
    endpoint, _, response = case
    record = Mock(error=None, _parsed_responses_cache=None)
    record.request_info.payload_bytes = None
    if missing_context:
        record.request_info = None
    record.responses = [create_mock_response(json_data=response)]
    assert endpoint.extract_response_data(record) == []
    assert "retained request payload" in record.error.message
    assert record._parsed_responses_cache == []


@pytest.mark.parametrize(
    "aliases",
    [
        {"prompt_tokens": 100, "completion_tokens": 200},
        {"prompt_tokens": True, "completion_tokens": "wrong"},
        {"usageMetadata": {"prompt_tokens": 100, "completion_tokens": 200}},
        {"cache_read_input_tokens": 100},
        {
            "prompt_tokens_details": {"cached_tokens": 100},
            "completion_tokens_details": {"reasoning_tokens": 200},
        },
    ],
)
def test_only_validated_decision_usage_fields_feed_metrics(case, aliases):
    endpoint, _, response = case
    response["usage"] = {
        "input_tokens": 10,
        "output_tokens": 1,
        "input_tokens_details": {"cached_tokens": 2},
        "output_tokens_details": {"reasoning_tokens": 1},
        **aliases,
    }
    original = deepcopy(response)
    parsed = endpoint.parse_response(create_mock_response(json_data=response))
    assert parsed.usage.prompt_tokens == 10
    assert parsed.usage.completion_tokens == 1
    assert parsed.usage.prompt_cache_read_tokens == 2
    assert parsed.usage.reasoning_tokens == 1
    assert parsed.data.body == original
    assert response == original


@pytest.fixture
def systemone_mixed_case():
    endpoint = SystemOneEndpoint(create_model_endpoint("systemone"))
    body = {
        "model": "test-model",
        "state": "evidence",
        "questions": {
            "z": {"type": "noul", "instructions": "Useful?"},
            "a": {"type": "choice", "criteria": {"yes": "Yes", "no": "No"}},
            "score": {"type": "score", "criteria": ["low", "medium", "high"]},
        },
    }
    response = {
        "model": "test-model",
        "answers": {
            "z": {"type": "noul", "noul": 0.8},
            "a": {
                "type": "choice",
                "choice": "yes",
                "confidence": 0.6,
                "probabilities": {"yes": 0.8, "no": 0.2},
            },
            "score": {
                "type": "score",
                "score": 1.3,
                "confidence": 0.5,
                "probabilities": {"0": 0.2, "1": 0.3, "2": 0.5},
                "legend": {"0": "low", "1": "medium", "2": "high"},
            },
        },
    }
    return endpoint, body, response


@pytest.mark.parametrize(
    "reorder", ["answers", "choice", "probabilities", "legend", "all"]
)
def test_systemone_maps_are_correlated_by_key_not_order(systemone_mixed_case, reorder):
    endpoint, body, response = systemone_mixed_case
    answers = response["answers"]
    if reorder in ("answers", "all"):
        response["answers"] = dict(reversed(answers.items()))
    if reorder in ("choice", "all"):
        answers["a"]["probabilities"] = {"no": 0.2, "yes": 0.8}
    if reorder in ("probabilities", "all"):
        answers["score"]["probabilities"] = {"2": 0.5, "0": 0.2, "1": 0.3}
    if reorder in ("legend", "all"):
        answers["score"]["legend"] = {"1": "medium", "2": "high", "0": "low"}
    record = create_record(body, response)
    parsed = endpoint.extract_response_data(record)
    assert record.error is None
    assert parsed[0].data.body == response


@pytest.mark.parametrize("mismatch", [False, True])
def test_systemone_structured_score_legend_matches_criteria(
    systemone_mixed_case, mismatch
):
    endpoint, body, response = systemone_mixed_case
    levels = [
        "low",
        {"description": "medium", "examples": ["degraded"]},
        ["high", {"blocked": True}],
    ]
    body["questions"]["score"]["criteria"] = levels
    response["answers"]["score"]["legend"] = {
        str(i): level for i, level in enumerate(levels)
    }
    if mismatch:
        response["answers"]["score"]["legend"]["1"] = {"description": "wrong"}
    record = create_record(body, response)
    parsed = endpoint.extract_response_data(record)
    if mismatch:
        assert parsed == []
        assert "Score labels differ" in record.error.message
    else:
        assert record.error is None
        assert parsed[0].data.body == response


@pytest.mark.parametrize(
    ("expected", "actual", "matches"),
    [
        param({"blocked": True}, {"blocked": 1}, False, id="boolean-number"),
        param({"blocked": 1}, {"blocked": True}, False, id="number-boolean"),
        param([{"blocked": False}], [{"blocked": 0}], False, id="nested-array"),
        param({"a": True, "b": [1, None]}, {"b": [1.0, None], "a": True}, True, id="object-order"),
        param(["low", "high"], ["high", "low"], False, id="array-order"),
        param({"blocked": False}, {"blocked": True}, False, id="boolean-value"),
        param({"value": None}, {"value": 0}, False, id="null-number"),
    ],
)  # fmt: skip
def test_systemone_structured_score_labels_preserve_json_types(
    systemone_mixed_case, expected, actual, matches
):
    endpoint, body, response = systemone_mixed_case
    body["questions"]["score"]["criteria"] = ["low", expected, "high"]
    response["answers"]["score"]["legend"] = {"0": "low", "1": actual, "2": "high"}
    record = create_record(body, response)
    parsed = endpoint.extract_response_data(record)
    if matches:
        assert record.error is None
        assert parsed[0].data.body == response
    else:
        assert parsed == []
        assert "Score labels differ" in record.error.message
        assert record._parsed_responses_cache == []


@pytest.mark.parametrize(
    "invalid", ["missing", "extra", "ordinal", "legend", "candidate", "criteria"]
)
def test_systemone_map_changes_still_rejected(systemone_mixed_case, invalid):
    endpoint, body, response = systemone_mixed_case
    if invalid == "missing":
        del response["answers"]["a"]
    elif invalid == "extra":
        response["answers"]["extra"] = {"type": "noul", "noul": 0.5}
    elif invalid == "legend":
        response["answers"]["score"]["legend"]["3"] = response["answers"]["score"][
            "legend"
        ].pop("2")
    elif invalid == "candidate":
        response["answers"]["a"]["choice"] = "other"
        response["answers"]["a"]["probabilities"] = {"other": 0.8, "no": 0.2}
    elif invalid == "criteria":
        body["questions"]["a"]["criteria"] = []
    else:
        response["answers"]["score"]["probabilities"] = {"0": 0.2, "1": 0.3, "3": 0.5}
    record = create_record(body, response)
    assert endpoint.extract_response_data(record) == []
    assert record.error is not None


def test_openai_answer_arrays_remain_positional():
    endpoint = OpenAIDecisionsEndpoint(create_model_endpoint("decisions"))
    body = {
        "model": "test-model",
        "input": "evidence",
        "questions": [{"type": "predicate", "name": name} for name in ("z", "a")],
    }
    response = {
        "model": "test-model",
        "answers": [
            {"type": "predicate", "name": name, "probability": 0.8}
            for name in ("a", "z")
        ],
    }
    record = create_record(body, response)
    assert endpoint.extract_response_data(record) == []
    assert "names/order" in record.error.message
