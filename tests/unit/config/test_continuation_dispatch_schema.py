# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Published continuation policy constraints match Python configuration validation."""

import copy
from pathlib import Path
from typing import Any

import orjson
import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError
from pytest import param

from aiperf.config import AIPerfConfig
from tools.generate_config_schema import ConfigSchemaGenerator


@pytest.fixture(scope="module", params=["generated", "published"])
def schema_validator(request: pytest.FixtureRequest) -> Draft202012Validator:
    """Exercise both the generator and the schema shipped to editors."""
    if request.param == "generated":
        schema = orjson.loads(ConfigSchemaGenerator().generate().files[0].content)
    else:
        schema_path = (
            Path(__file__).resolve().parents[3]
            / "src/aiperf/config/schema/aiperf-config.schema.json"
        )
        schema = orjson.loads(schema_path.read_bytes())
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _assert_policy_parity(
    validator: Draft202012Validator,
    phase: dict[str, Any],
    accepted: bool,
    *,
    form: str = "list",
) -> None:
    """Validate the same payload through the public schema and Python model."""
    benchmark = {
        "models": "test-model",
        "endpoint": {"url": "http://localhost:8000"},
        "dataset": {"type": "synthetic"},
    }
    if form == "list":
        benchmark["phases"] = [phase]
    elif form == "single":
        benchmark["phases"] = phase
    elif form == "profiling":
        benchmark["profiling"] = phase
    else:
        benchmark["warmup"] = phase
        benchmark["profiling"] = {"type": "concurrency", "requests": 1}
    payload = {"benchmark": benchmark}
    if accepted:
        AIPerfConfig.model_validate(copy.deepcopy(payload))
    else:
        with pytest.raises(ValidationError):
            AIPerfConfig.model_validate(copy.deepcopy(payload))
    errors = list(validator.iter_errors(payload))
    assert (not errors) == accepted, [error.message for error in errors]


@pytest.mark.parametrize("phase_type", ["constant", "poisson", "gamma", "user_centric"])
@pytest.mark.parametrize("policy", [None, "rate_slot", "direct"])
@pytest.mark.parametrize("key", ["continuationDispatch", "continuation_dispatch"])
def test_schema_and_python_agree_on_continuation_policy(
    schema_validator: Draft202012Validator,
    phase_type: str,
    policy: str | None,
    key: str,
) -> None:
    phase = {"name": "profiling", "type": phase_type, "requests": 9, "rate": 4}
    if phase_type == "user_centric":
        phase["users"] = 1
    if policy is not None:
        phase[key] = policy
    _assert_policy_parity(
        schema_validator,
        phase,
        not (phase_type == "user_centric" and policy == "direct"),
    )


@pytest.mark.parametrize(
    ("overrides", "accepted"),
    [
        param({"adaptiveScale": False}, True, id="disabled-flat"),
        param({"adaptive_scale": {"enabled": False}}, True, id="disabled-nested-snake"),
        param({"adaptiveScale": {"enabled": "off"}}, True, id="disabled-nested-string"),
        param({"adaptiveScale": True}, False, id="enabled-flat"),
        param({"adaptive_scale": {"enabled": True}}, False, id="enabled-nested-snake"),
        param({"adaptiveScale": {}}, False, id="enabled-nested-default"),
        param({"adaptiveScale": True, "adaptive_scale": False}, True, id="snake-disabled-wins"),
        param({"adaptiveScale": {"enabled": True}, "adaptive_scale": {"enabled": False}}, True, id="snake-nested-disabled-wins"),
        param({"adaptiveScale": False, "adaptive_scale": True}, False, id="snake-enabled-wins"),
        param({"timingMode": None}, True, id="timing-default"),
        param({"timingMode": "request_rate"}, True, id="timing-request-rate"),
        param({"timingMode": "agentic_replay"}, False, id="timing-agentic"),
        param({"timingMode": "adaptive_scale"}, False, id="timing-adaptive"),
        param({"timingMode": "fixed_schedule"}, False, id="timing-fixed"),
        param({"rate": None, "rateSeries": {"points": [{"timeS": 0, "qps": 2}, {"timeS": 10, "qps": 4}]}}, True, id="rate-series"),
    ],
)  # fmt: skip
@pytest.mark.parametrize("form", ["list", "single", "profiling", "warmup"])
def test_schema_direct_constraints_survive_phase_shorthands(
    schema_validator: Draft202012Validator,
    overrides: dict[str, Any],
    accepted: bool,
    form: str,
) -> None:
    phase = {
        "type": "poisson",
        "rate": 4,
        "requests": 9,
        "duration": 10,
        "adaptiveSustainDuration": 1,
        "adaptiveControlVariable": "request_rate",
        "sla": [{"metricTag": "request_latency", "op": "le", "threshold": 1000}],
        "continuationDispatch": "direct",
        **overrides,
    }
    if form == "list":
        phase["name"] = "profiling"
    _assert_policy_parity(schema_validator, phase, accepted, form=form)


def test_schema_and_python_reject_duplicate_policy_aliases(
    schema_validator: Draft202012Validator,
) -> None:
    _assert_policy_parity(
        schema_validator,
        {
            "name": "profiling",
            "type": "constant",
            "rate": 4,
            "requests": 9,
            "continuationDispatch": "rate_slot",
            "continuation_dispatch": "direct",
        },
        False,
    )


@pytest.mark.parametrize("policy", [None, "rate_slot"])
@pytest.mark.parametrize(
    "overrides",
    [
        param({"adaptiveScale": True}, id="adaptive"),
        param({"timingMode": "agentic_replay"}, id="agentic"),
        param({"timingMode": "fixed_schedule"}, id="fixed"),
    ],
)  # fmt: skip
def test_default_policy_retains_existing_phase_combinations(
    schema_validator: Draft202012Validator,
    policy: str | None,
    overrides: dict[str, Any],
) -> None:
    phase = {
        "name": "profiling",
        "type": "poisson",
        "rate": 4,
        "duration": 10,
        "adaptiveSustainDuration": 1,
        "adaptiveControlVariable": "request_rate",
        "sla": [{"metricTag": "request_latency", "op": "le", "threshold": 1000}],
        **overrides,
    }
    if policy is not None:
        phase["continuationDispatch"] = policy
    _assert_policy_parity(schema_validator, phase, True)
