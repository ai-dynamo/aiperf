# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every synthetic-only flag is rejected on file/public datasets, not dropped.

The mechanical test below re-derives the field list from the ``_build_*``
functions, so a newly added synthetic flag fails here rather than quietly
joining the silently-dropped set.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pytest import param

from aiperf.config.flags import _converter_dataset as conv
from aiperf.config.flags.cli_config import CLIConfig


def _cli_fields_referenced(func: Callable[[CLIConfig], dict[str, Any]]) -> set[str]:
    """Every CLIConfig field name the builder reads, however it reads it.

    Covers all four access shapes the builders use: ``"field" in s``,
    ``cli.field``, ``getattr(cli, key)`` over a dict literal, and field names
    passed as string arguments to ``_mean_stddev_pair``.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    known = set(CLIConfig.model_fields)
    found: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value in known
        ):
            found.add(node.value)
        elif (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "cli"
            and node.attr in known
        ):
            found.add(node.attr)
    return found


@pytest.mark.parametrize(
    "builder",
    [
        param(conv._build_prefix_prompts, id="prefix_prompts"),
        param(conv._build_rankings, id="rankings"),
        param(conv._build_audio, id="audio"),
        param(conv._build_images, id="images"),
        param(conv._build_video, id="video"),
    ],
)  # fmt: skip
def test_every_synthetic_only_flag_is_rejected_or_rescued(
    builder: Callable[[CLIConfig], dict[str, Any]],
) -> None:
    """No field of a stripped subtable may be silently accepted.

    ``prompts`` is excluded from this sweep on purpose: it is stripped too, but
    several members are re-attached for FILE/PUBLIC (``_apply_file_osl``,
    ``_apply_corpus_and_cache_bust``, ``_apply_block_size``,
    ``_apply_sequence_distribution``), so --osl/--cache-bust/--prompt-corpus
    legitimately work there.
    """
    rejected = {attr for attr, _ in conv._FILE_DATASET_INCOMPATIBLE_TRIGGERS}
    unhandled = _cli_fields_referenced(builder) - rejected
    assert not unhandled, (
        f"{sorted(unhandled)} are read by {builder.__name__} but are neither in "
        "_FILE_DATASET_INCOMPATIBLE_TRIGGERS nor rescued, so they are silently "
        "dropped on file/public datasets."
    )


def _trace(tmp_path: Path) -> str:
    trace = tmp_path / "t.jsonl"
    trace.write_text('{"text": "hi"}\n')
    return str(trace)


class TestNewlyRejected:
    @pytest.mark.parametrize(
        "field,value,expected_flag",
        [
            param("image_width_mean", 64, "--image-width-mean", id="image_shape"),
            param("image_format", "png", "--image-format", id="image_format"),
            param("audio_length_mean", 5.0, "--audio-length-mean", id="audio_shape"),
            param("video_duration", 3.0, "--video-duration", id="video"),
            param(
                "rankings_passages_mean", 4, "--rankings-passages-mean", id="rankings"
            ),
        ],
    )  # fmt: skip
    def test_rejected_with_input_file(
        self, tmp_path: Path, field: str, value: object, expected_flag: str
    ) -> None:
        cli = CLIConfig(
            model_names=["m"], input_file=_trace(tmp_path), **{field: value}
        )
        with pytest.raises(ValueError, match=expected_flag):
            conv.build_dataset(cli)

    def test_rejected_with_public_dataset(self) -> None:
        cli = CLIConfig(
            model_names=["m"], public_dataset="sharegpt", image_width_mean=64
        )
        with pytest.raises(ValueError, match="--image-width-mean"):
            conv.build_dataset(cli)


class TestUnchangedBehavior:
    """Guardrails: this must not widen beyond the silently-dropped set."""

    def test_synthetic_dataset_still_accepts_them(self) -> None:
        cli = CLIConfig(model_names=["m"], image_width_mean=64, audio_length_mean=5.0)
        out = conv.build_dataset(cli)
        assert out["images"]["width"]["mean"] == 64
        assert out["audio"]["length"]["mean"] == 5.0
