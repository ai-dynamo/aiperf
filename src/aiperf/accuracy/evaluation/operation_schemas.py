# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical typed model-operation schemas shared by stock providers."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any

from aiperf.accuracy.evaluation.canonical import canonical_dumps, canonical_sha256

_SCALAR_OR_OBJECT: dict[str, Any] = {
    "oneOf": [
        {"type": "null"},
        {"type": "boolean"},
        {"type": "integer"},
        {"type": "number"},
        {"type": "string"},
        {"type": "array"},
        {"type": "object"},
    ]
}

_CONTENT_BLOCK: dict[str, Any] = {
    "oneOf": [
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["type", "text"],
            "properties": {"type": {"const": "text"}, "text": {"type": "string"}},
        },
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["type", "reasoning"],
            "properties": {
                "type": {"const": "reasoning"},
                "reasoning": {"type": "string"},
                "signature": {"type": "string"},
            },
        },
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["type", "asset_id"],
            "properties": {
                "type": {"enum": ["image", "audio", "video", "document", "data"]},
                "asset_id": {"type": "string", "minLength": 1},
                "media_type": {"type": "string", "minLength": 1},
                "detail": {"enum": ["auto", "low", "high"]},
            },
        },
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["type", "tool_call_id", "content"],
            "properties": {
                "type": {"const": "tool_result"},
                "tool_call_id": {"type": "string", "minLength": 1},
                "content": _SCALAR_OR_OBJECT,
                "is_error": {"type": "boolean"},
            },
        },
    ]
}

_TOOL_CALL: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["id", "type", "function"],
    "properties": {
        "id": {"type": "string", "minLength": 1},
        "type": {"const": "function"},
        "function": {
            "type": "object",
            "additionalProperties": False,
            "required": ["name", "arguments"],
            "properties": {
                "name": {"type": "string", "minLength": 1},
                "arguments": {"type": "object"},
            },
        },
    },
}

_MESSAGE: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["role", "content"],
    "properties": {
        "role": {"enum": ["system", "developer", "user", "assistant", "tool"]},
        "content": {
            "oneOf": [
                {"type": "string"},
                {"type": "array", "items": _CONTENT_BLOCK},
            ]
        },
        "name": {"type": "string"},
        "tool_call_id": {"type": "string"},
        "tool_calls": {"type": "array", "items": _TOOL_CALL},
    },
}

_GENERATION: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["max_tokens"],
    "properties": {
        "max_tokens": {"type": "integer", "minimum": 1},
        "temperature": {"type": "number"},
        "top_p": {"type": "number"},
        "stop": {
            "oneOf": [
                {"type": "string"},
                {"type": "array", "items": {"type": "string"}},
            ]
        },
    },
}

_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "best_of": {"type": "integer", "minimum": 1},
        "frequency_penalty": {"type": "number"},
        "presence_penalty": {"type": "number"},
        "logit_bias": {"type": "object", "additionalProperties": {"type": "number"}},
        "seed": {"type": "integer"},
        "top_k": {"type": "integer", "minimum": 1},
        "num_choices": {"type": "integer", "minimum": 1},
        "logprobs": {"type": "boolean"},
        "top_logprobs": {"type": "integer", "minimum": 0},
        "parallel_tool_calls": {"type": "boolean"},
        "internal_tools": {"type": "boolean"},
        "max_tool_output": {"type": "integer", "minimum": 1},
        "reasoning_effort": {"enum": ["minimal", "low", "medium", "high"]},
        "reasoning_tokens": {"type": "integer", "minimum": 0},
        "reasoning_summary": {"enum": ["concise", "detailed", "auto"]},
        "reasoning_history": {"enum": ["none", "all", "last", "auto"]},
    },
}

_TOOL: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["type", "function"],
    "properties": {
        "type": {"const": "function"},
        "function": {
            "type": "object",
            "additionalProperties": False,
            "required": ["name", "parameters"],
            "properties": {
                "name": {"type": "string", "minLength": 1},
                "description": {"type": "string"},
                "parameters": {"type": "object"},
            },
        },
    },
}

_USAGE: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "prompt_tokens": {"type": "integer", "minimum": 0},
        "completion_tokens": {"type": "integer", "minimum": 0},
        "reasoning_tokens": {"type": "integer", "minimum": 0},
        "cached_tokens": {"type": "integer", "minimum": 0},
    },
}

_GENERATE_RESPONSE: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["choices", "usage"],
    "properties": {
        "choices": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["message", "stop_reason"],
                "properties": {
                    "message": _MESSAGE,
                    "stop_reason": {
                        "enum": [
                            "stop",
                            "max_tokens",
                            "model_length",
                            "tool_calls",
                            "content_filter",
                            "unknown",
                        ]
                    },
                    "finish_reason": {"type": "string"},
                    "logprobs": {"type": ["object", "null"]},
                },
            },
        },
        "usage": _USAGE,
    },
}

_EMPTY_STREAM_SCHEMA: dict[str, Any] = {"type": "null"}

MODEL_GENERATE_SCHEMA: dict[str, Any] = {
    "schema_version": 1,
    "request": {
        "type": "object",
        "additionalProperties": False,
        "required": ["messages", "generation"],
        "properties": {
            "messages": {"type": "array", "minItems": 1, "items": _MESSAGE},
            "generation": _GENERATION,
            "tools": {"type": "array", "items": _TOOL},
            "tool_choice": {
                "oneOf": [
                    {"enum": ["auto", "none", "required"]},
                    {"type": "object"},
                ]
            },
            "response_format": {"type": "object"},
            "parameters": _PARAMETERS,
        },
    },
    "response": _GENERATE_RESPONSE,
    "stream": {
        "oneOf": [
            _EMPTY_STREAM_SCHEMA,
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["choice_index", "delta"],
                "properties": {
                    "choice_index": {"type": "integer", "minimum": 0},
                    "delta": _MESSAGE,
                },
            },
        ]
    },
}

MODEL_COMPLETE_SCHEMA: dict[str, Any] = {
    "schema_version": 1,
    "request": {
        "type": "object",
        "additionalProperties": False,
        "required": ["prompt", "generation"],
        "properties": {
            "prompt": {
                "oneOf": [
                    {"type": "string"},
                    {"type": "array", "minItems": 1, "items": {"type": "string"}},
                ]
            },
            "generation": _GENERATION,
            "parameters": _PARAMETERS,
        },
    },
    "response": {
        "type": "object",
        "additionalProperties": False,
        "required": ["choices", "usage"],
        "properties": {
            "choices": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["text", "finish_reason"],
                    "properties": {
                        "text": {"type": "string"},
                        "finish_reason": {"type": "string"},
                        "logprobs": {"type": ["object", "null"]},
                    },
                },
            },
            "usage": _USAGE,
        },
    },
    "stream": {
        "oneOf": [
            _EMPTY_STREAM_SCHEMA,
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["choice_index", "text"],
                "properties": {
                    "choice_index": {"type": "integer", "minimum": 0},
                    "text": {"type": "string"},
                },
            },
        ]
    },
}

MODEL_RESPONSES_SCHEMA: dict[str, Any] = {
    "schema_version": 1,
    "request": {
        "type": "object",
        "additionalProperties": False,
        "required": ["input", "generation"],
        "properties": {
            "input": {"type": "array", "minItems": 1, "items": _MESSAGE},
            "instructions": {"type": "string"},
            "generation": _GENERATION,
            "tools": {"type": "array", "items": _TOOL},
            "parameters": _PARAMETERS,
        },
    },
    "response": {
        "type": "object",
        "additionalProperties": False,
        "required": ["output", "usage", "status"],
        "properties": {
            "output": {"type": "array", "items": _MESSAGE},
            "usage": _USAGE,
            "status": {"enum": ["completed", "incomplete", "failed"]},
        },
    },
    "stream": {
        "oneOf": [
            _EMPTY_STREAM_SCHEMA,
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["event_type", "item"],
                "properties": {
                    "event_type": {"type": "string", "minLength": 1},
                    "item": _SCALAR_OR_OBJECT,
                },
            },
        ]
    },
}

MODEL_EMBED_SCHEMA: dict[str, Any] = {
    "schema_version": 1,
    "request": {
        "type": "object",
        "additionalProperties": False,
        "required": ["input"],
        "properties": {
            "input": {
                "oneOf": [
                    {"type": "string"},
                    {"type": "array", "minItems": 1, "items": {"type": "string"}},
                ]
            },
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "dimensions": {"type": "integer", "minimum": 1},
                    "encoding_format": {"enum": ["float"]},
                },
            },
        },
    },
    "response": {
        "type": "object",
        "additionalProperties": False,
        "required": ["embeddings", "usage"],
        "properties": {
            "embeddings": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "number"}},
            },
            "usage": _USAGE,
        },
    },
    "stream": _EMPTY_STREAM_SCHEMA,
}

OPERATION_SCHEMAS = MappingProxyType(
    {
        "model.generate": MODEL_GENERATE_SCHEMA,
        "model.complete": MODEL_COMPLETE_SCHEMA,
        "model.responses": MODEL_RESPONSES_SCHEMA,
        "model.embed": MODEL_EMBED_SCHEMA,
    }
)

OPERATION_SCHEMA_SHA256 = MappingProxyType(
    {name: canonical_sha256(schema) for name, schema in OPERATION_SCHEMAS.items()}
)

OPERATION_DIRECTION_SCHEMA_SHA256 = MappingProxyType(
    {
        name: MappingProxyType(
            {
                "request": canonical_sha256(schema["request"]),
                "response": canonical_sha256(schema["response"]),
                "stream": canonical_sha256(schema["stream"]),
            }
        )
        for name, schema in OPERATION_SCHEMAS.items()
    }
)


def operation_schema_bytes(operation_id: str) -> bytes:
    """Return exact canonical request/response/stream schema bytes."""
    return canonical_dumps(OPERATION_SCHEMAS[operation_id])
