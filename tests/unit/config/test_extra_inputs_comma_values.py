# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""An --extra-inputs value may contain a comma (AIP-2217).

Splitting on every comma before splitting on ``:`` cut a value at its first
comma and parsed each remaining fragment as its own ``key:value`` pair -- so a
multi-field ``payload_template`` lost everything after the first comma AND
silently gained bogus keys in the request payload. Two examples published in
`docs/tutorials/template-endpoint.md` could not work as written.

The failure is quiet where the truncation still parses: the run sends a payload
the user never asked for, which for a benchmarking tool means publishing numbers
that do not match the request.
"""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.config.loader.parsing import _parse_str_as_tuple_list

TEMPLATE = 'payload_template:{"input": {{ texts|tojson }}, "model": {{ model|tojson }}}'


def test_a_multi_field_template_survives_intact() -> None:
    """The docs example that could not work before."""
    assert _parse_str_as_tuple_list(TEMPLATE) == [
        (
            "payload_template",
            '{"input": {{ texts|tojson }}, "model": {{ model|tojson }}}',
        )
    ]


def test_a_truncated_template_does_not_become_an_extra_payload_key() -> None:
    """The quieter half: the surplus fragment used to be merged into the payload."""
    keys = [k for k, _ in _parse_str_as_tuple_list(TEMPLATE)]
    assert keys == ["payload_template"], (
        f"parsed spurious keys {keys[1:]} -- these are merged into the request payload"
    )


@pytest.mark.parametrize(
    "raw, expected",
    [
        param("a:1,b:2,c:3", [("a", 1), ("b", 2), ("c", 3)], id="plain-pairs"),
        param("temperature:0.7, top_p:0.9", [("temperature", 0.7), ("top_p", 0.9)], id="spaced-pairs"),
        param("a:[1,2,3]", [("a", "[1,2,3]")], id="json-array-value"),
        param('prompt:"hello, world"', [("prompt", '"hello, world"')], id="quoted-comma"),
        param(
            'payload_template:{"m": {{ model|tojson }}}, extra:1',
            [("payload_template", '{"m": {{ model|tojson }}}'), ("extra", 1)],
            id="template-then-pair",
        ),
    ],
)  # fmt: skip
def test_values_split_only_on_separating_commas(raw: str, expected) -> None:
    assert _parse_str_as_tuple_list(raw) == expected


def test_json_object_form_is_untouched() -> None:
    """A leading `{` is parsed by orjson before the comma path is reached."""
    assert _parse_str_as_tuple_list('{"a": 1, "b": 2}') == [("a", 1), ("b", 2)]


def test_an_item_without_a_colon_still_errors() -> None:
    """Unbalanced delimiters must surface, not silently swallow the remainder."""
    with pytest.raises(ValueError, match="key:value"):
        _parse_str_as_tuple_list("a:1,bare")


def test_an_escaped_quote_does_not_end_the_value() -> None:
    """A backslash-escaped quote is part of the string, not its terminator.

    Without this, the value would be considered closed at the inner quote and
    the comma after it would split mid-value -- the same truncation this fix
    exists to prevent, just one level deeper.
    """
    raw = 'prompt:"he said \\"hi\\", ok"'
    assert _parse_str_as_tuple_list(raw) == [("prompt", '"he said \\"hi\\", ok"')]


def test_a_separator_after_an_escaped_quote_still_splits() -> None:
    """The escape must not swallow the rest of the input either."""
    raw = 'a:"x\\"y",b:2'
    assert _parse_str_as_tuple_list(raw) == [("a", '"x\\"y"'), ("b", 2)]
