# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Parser for sequence length distribution string formats.

Split out of ``sequence_distribution`` to keep that module under the
ergonomics file-size limit. Public API is re-exported from
``aiperf.common.models.sequence_distribution`` for backward compatibility.
"""

from __future__ import annotations

import re

import orjson

from aiperf.common.models.sequence_distribution_core import (
    SequenceLengthDistribution,
    SequenceLengthPair,
    _validate_probability_sum,
)
from aiperf.common.utils import load_json_str


class DistributionParser:
    """Parser for various sequence length distribution string formats."""

    # Regex patterns for different formats (allow whitespace and optional stddev)
    # Use unambiguous numeric pattern to avoid ReDoS: (?:\d+(?:\.\d+)?|\.\d+)
    # This matches: integers (123), decimals (123.45), or leading-dot decimals (.45)
    _NUM = r"(?:\d+(?:\.\d+)?|\.\d+)"
    SEMICOLON_PATTERN = re.compile(
        rf"(\d+)(?:\|({_NUM}))?\s*,\s*(\d+)(?:\|({_NUM}))?\s*:\s*({_NUM})"
    )
    BRACKET_PATTERN = re.compile(
        rf"\(\s*(\d+)(?:\|({_NUM}))?\s*,\s*(\d+)(?:\|({_NUM}))?\s*\)\s*:\s*({_NUM})"
    )

    @classmethod
    def validate(cls, dist_str: str) -> list[SequenceLengthPair]:
        """
        Validate distribution string format without creating the full distribution object.

        This method parses the string and creates SequenceLengthPair objects to validate
        the format, but does NOT create a SequenceLengthDistribution (which requires RNG).

        Args:
            dist_str: Distribution specification string

        Returns:
            List of validated SequenceLengthPair objects

        Raises:
            ValueError: If string format is invalid or unrecognized
        """
        if not isinstance(dist_str, str) or not dist_str.strip():
            raise ValueError(
                "SequenceLengthDistribution.validate() failed: input string is empty; "
                "expected one of: semicolon ('256,128:40;512,256:60'), "
                "bracket ('[(256,128):40,(512,256):60]'), "
                'or JSON (\'{"pairs": [{"isl": 256, "osl": 128, "prob": 40}, ...]}\')'
            )

        dist_str = dist_str.strip()

        try:
            # Try JSON format first
            if dist_str.startswith("{"):
                pairs = cls._validate_json_format(dist_str)
            # Try bracket format
            elif dist_str.startswith("[") and dist_str.endswith("]"):
                pairs = cls._validate_bracket_format(dist_str[1:-1])
            # Default to semicolon format
            else:
                pairs = cls._validate_semicolon_format(dist_str)

            # Validate probability sum without creating distribution object
            _validate_probability_sum(pairs)
            return pairs

        except Exception as e:
            raise ValueError(
                f"Failed to parse distribution string '{dist_str}': {e}"
            ) from e

    @classmethod
    def parse(cls, dist_str: str) -> SequenceLengthDistribution:
        """
        Parse distribution string in various supported formats.

        Supported formats:
        - Semicolon: "256,128:40;512,256:60" (percentages) or "256,128:0.4;512,256:0.6" (fractions)
        - With stddev: "256|10,128|5:40;512|20,256|10:60" (mean|stddev format)
        - Bracket: "[(256,128):40,(512,256):60]" or "[(256|10,128|5):40,(512|20,256|10):60]"
        - JSON: '{"pairs": [{"isl": 256, "isl_stddev": 10, "osl": 128, "osl_stddev": 5, "prob": 40}, ...]}'

        Args:
            dist_str: Distribution specification string

        Returns:
            SequenceLengthDistribution object

        Raises:
            ValueError: If string format is invalid or unrecognized
        """
        if not isinstance(dist_str, str) or not dist_str.strip():
            raise ValueError(
                "SequenceLengthDistribution.validate() failed: input string is empty; "
                "expected one of: semicolon ('256,128:40;512,256:60'), "
                "bracket ('[(256,128):40,(512,256):60]'), "
                'or JSON (\'{"pairs": [{"isl": 256, "osl": 128, "prob": 40}, ...]}\')'
            )

        dist_str = dist_str.strip()

        try:
            # Try JSON format first
            if dist_str.startswith("{"):
                return cls._parse_json_format(dist_str)

            # Try bracket format
            if dist_str.startswith("[") and dist_str.endswith("]"):
                return cls._parse_bracket_format(dist_str[1:-1])

            # Default to semicolon format
            return cls._parse_semicolon_format(dist_str)

        except Exception as e:
            raise ValueError(
                f"Failed to parse distribution string '{dist_str}': {e}"
            ) from e

    @classmethod
    def _parse_pairs_from_json(cls, json_str: str) -> list[SequenceLengthPair]:
        """Parse JSON format and extract pairs: {"pairs": [{"isl": 256, "isl_stddev": 10, "osl": 128, "osl_stddev": 5, "prob": 40}, ...]}"""
        try:
            data = load_json_str(json_str)
        except orjson.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}") from e

        if "pairs" not in data:
            raise ValueError(
                f"SequenceLengthDistribution JSON parse failed: top-level object "
                f"missing required 'pairs' key; got keys={sorted(data.keys())!r}; "
                'expected \'{"pairs": [{"isl": <int>, "osl": <int>, "prob": <float>}, ...]}\''
            )

        pairs = []
        for i, pair_data in enumerate(data["pairs"]):
            required_keys = {"isl", "osl", "prob"}
            if not required_keys.issubset(pair_data.keys()):
                missing = required_keys - pair_data.keys()
                raise ValueError(f"Pair {i} missing required keys: {missing}")

            pairs.append(
                SequenceLengthPair(
                    input_seq_len=int(pair_data["isl"]),
                    output_seq_len=int(pair_data["osl"]),
                    probability=float(pair_data["prob"]),
                    input_seq_len_stddev=float(pair_data.get("isl_stddev", 0.0)),
                    output_seq_len_stddev=float(pair_data.get("osl_stddev", 0.0)),
                )
            )
        return pairs

    @classmethod
    def _parse_pairs_from_bracket(cls, content: str) -> list[SequenceLengthPair]:
        """Parse bracket format and extract pairs: (256|10,128|5):40,(512|20,256|10):60 or (256,128):40,(512,256):60"""
        pairs = []
        for match in cls.BRACKET_PATTERN.finditer(content):
            isl, isl_stddev, osl, osl_stddev, prob = match.groups()
            pairs.append(
                SequenceLengthPair(
                    input_seq_len=int(isl),
                    output_seq_len=int(osl),
                    probability=float(prob),
                    input_seq_len_stddev=float(isl_stddev) if isl_stddev else 0.0,
                    output_seq_len_stddev=float(osl_stddev) if osl_stddev else 0.0,
                )
            )
        if not pairs:
            raise ValueError("No valid pairs found in bracket format")
        return pairs

    @classmethod
    def _parse_pairs_from_semicolon(cls, dist_str: str) -> list[SequenceLengthPair]:
        """Parse semicolon format and extract pairs: 256|10,128|5:40;512|20,256|10:60 or 256,128:40;512,256:60"""
        pairs = []
        for pair_str in dist_str.split(";"):
            pair_str = pair_str.strip()
            if not pair_str:
                continue

            match = cls.SEMICOLON_PATTERN.fullmatch(pair_str)
            if not match:
                raise ValueError(
                    f"Invalid pair format: '{pair_str}'. Expected 'ISL[|ISL_STDDEV],OSL[|OSL_STDDEV]:PROB'"
                )

            isl, isl_stddev, osl, osl_stddev, prob = match.groups()
            pairs.append(
                SequenceLengthPair(
                    input_seq_len=int(isl),
                    output_seq_len=int(osl),
                    probability=float(prob),
                    input_seq_len_stddev=float(isl_stddev) if isl_stddev else 0.0,
                    output_seq_len_stddev=float(osl_stddev) if osl_stddev else 0.0,
                )
            )
        if not pairs:
            raise ValueError("No valid pairs found in semicolon format")
        return pairs

    @classmethod
    def _validate_json_format(cls, json_str: str) -> list[SequenceLengthPair]:
        """Validate JSON format without creating distribution object."""
        return cls._parse_pairs_from_json(json_str)

    @classmethod
    def _validate_bracket_format(cls, content: str) -> list[SequenceLengthPair]:
        """Validate bracket format without creating distribution object."""
        return cls._parse_pairs_from_bracket(content)

    @classmethod
    def _validate_semicolon_format(cls, dist_str: str) -> list[SequenceLengthPair]:
        """Validate semicolon format without creating distribution object."""
        return cls._parse_pairs_from_semicolon(dist_str)

    @classmethod
    def _parse_json_format(cls, json_str: str) -> SequenceLengthDistribution:
        """Parse JSON format: {"pairs": [{"isl": 256, "isl_stddev": 10, "osl": 128, "osl_stddev": 5, "prob": 40}, ...]}"""
        pairs = cls._parse_pairs_from_json(json_str)
        return SequenceLengthDistribution(pairs)

    @classmethod
    def _parse_bracket_format(cls, content: str) -> SequenceLengthDistribution:
        """Parse bracket format: (256|10,128|5):40,(512|20,256|10):60 or (256,128):40,(512,256):60"""
        pairs = cls._parse_pairs_from_bracket(content)
        return SequenceLengthDistribution(pairs)

    @classmethod
    def _parse_semicolon_format(cls, dist_str: str) -> SequenceLengthDistribution:
        """Parse semicolon format: 256|10,128|5:40;512|20,256|10:60 or 256,128:40;512,256:60"""
        pairs = cls._parse_pairs_from_semicolon(dist_str)
        return SequenceLengthDistribution(pairs)
