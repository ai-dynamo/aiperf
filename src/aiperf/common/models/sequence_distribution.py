# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Sequence length distribution models for AIPerf benchmarking.

This module provides data models and parsers for specifying distributions of input sequence
length (ISL) and output sequence length (OSL) pairs with optional standard deviations,
allowing for more realistic LLM benchmarking scenarios.

The sequence distribution feature allows users to specify multiple ISL/OSL pairs with
different probabilities, enabling simulation of mixed workloads that better represent
production traffic patterns.

        Supported formats (probabilities must be percentages 0-100):
        - Semicolon: "256,128:40;512,256:60" or "256|10,128|5:40;512|20,256|10:60"
        - Bracket: "[(256,128):40,(512,256):60]" or "[(256|10,128|5):40,(512|20,256|10):60]"
        - JSON: '{"pairs": [{"isl": 256, "isl_stddev": 10, "osl": 128, "osl_stddev": 5, "prob": 40}, ...]}'

Note: Probabilities must be specified as percentages (0-100), not fractions (0-1).
This prevents common errors from mixing different probability formats.

Examples:
    Basic usage:
        >>> from aiperf.common.models.sequence_distribution import DistributionParser
        >>> dist = DistributionParser.parse("256,128:60;512,256:40")
        >>> isl, osl = dist.sample()

    With standard deviations:
        >>> dist = DistributionParser.parse("256|10,128|5:60;512|20,256|10:40")
        >>> isl, osl = dist.sample()  # Will vary around means based on stddev
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    ClassVar,
    Protocol,
    Self,
    runtime_checkable,
)

import numpy as np
import orjson
from pydantic import ConfigDict, Field, field_validator, model_validator

from aiperf.common import random_generator as rng
from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.enums import RandomCorpusStyle
from aiperf.common.utils import load_json_str
from aiperf.config.base import BaseConfig

if TYPE_CHECKING:
    from aiperf.common.random_generator import RandomGenerator

logger = AIPerfLogger(__name__)


@runtime_checkable
class SequenceLengthSampler(Protocol):
    """Anything that yields an (ISL, OSL) pair per call."""

    def sample(self) -> tuple[int, int]: ...


def _validate_probability_sum(pairs: list[SequenceLengthPair]) -> None:
    """
    Validate that probabilities sum to approximately 100.0.

    This is a module-level helper used by both SequenceLengthDistribution
    and DistributionParser to avoid code duplication.

    Args:
        pairs: List of SequenceLengthPair objects to validate

    Raises:
        ValueError: If probabilities don't sum to 100.0 (within floating-point tolerance)
    """
    total_prob = sum(pair.probability for pair in pairs)

    # Allow small floating-point errors
    if not np.isclose(total_prob, 100.0, rtol=1e-6, atol=1e-6):
        raise ValueError(
            f"Probabilities must sum to 100.0, got {total_prob:.6f}. "
            f"Pairs: {[str(p) for p in pairs]}"
        )


@dataclass(frozen=True)
class SequenceLengthPair:
    """Immutable representation of an ISL/OSL pair with probability weight and optional stddevs."""

    input_seq_len: int
    output_seq_len: int
    probability: float
    input_seq_len_stddev: float = 0.0
    output_seq_len_stddev: float = 0.0

    def __post_init__(self) -> None:
        """Validate sequence lengths, standard deviations, and probability on construction."""
        if self.input_seq_len <= 0:
            raise ValueError(
                f"Input sequence length must be positive, got {self.input_seq_len}"
            )
        if self.output_seq_len <= 0:
            raise ValueError(
                f"Output sequence length must be positive, got {self.output_seq_len}"
            )
        if not 0.0 <= self.probability <= 100.0:
            raise ValueError(f"Probability must be in [0,100], got {self.probability}")
        if self.input_seq_len_stddev < 0.0:
            raise ValueError(
                f"Input sequence length stddev must be non-negative, got {self.input_seq_len_stddev}"
            )
        if self.output_seq_len_stddev < 0.0:
            raise ValueError(
                f"Output sequence length stddev must be non-negative, got {self.output_seq_len_stddev}"
            )

    def __str__(self) -> str:
        if self.input_seq_len_stddev > 0 or self.output_seq_len_stddev > 0:
            return f"({self.input_seq_len}|{self.input_seq_len_stddev},{self.output_seq_len}|{self.output_seq_len_stddev}):{self.probability}%"
        else:
            return f"({self.input_seq_len},{self.output_seq_len}):{self.probability}%"


class SequenceLengthDistribution:
    """
    Manages probability distributions of ISL/OSL pairs for benchmark sampling.

    Supports efficient O(log n) sampling using binary search on cumulative
    probability distribution.
    """

    def __init__(self, pairs: list[SequenceLengthPair]) -> None:
        """
        Initialize distribution from list of sequence length pairs.

        Args:
            pairs: List of SequenceLengthPair objects. Probabilities must sum to 100.0.

        Raises:
            ValueError: If pairs is empty or probabilities don't sum to 100.0.
        """
        if not pairs:
            raise ValueError(
                "Distribution must contain at least one sequence length pair"
            )

        # RNG is derived lazily on first sample so that parsing/construction
        # stays pure and works before bootstrap calls rng.init().
        self._rng: RandomGenerator | None = None
        self._pairs = tuple(pairs)  # Immutable copy
        _validate_probability_sum(list(self._pairs))
        self._cumulative_probs = self._compute_cumulative_probabilities()

        logger.debug(f"Created distribution with {len(self._pairs)} pairs: {self}")

    def _get_rng(self) -> RandomGenerator:
        """Lazily derive and cache the RNG on first use."""
        if self._rng is None:
            self._rng = rng.derive("models.sequence.distribution")
        return self._rng

    def _compute_cumulative_probabilities(self) -> np.ndarray:
        """Compute cumulative probability distribution for efficient sampling."""
        # Convert percentages to fractions for internal calculation
        probs = [pair.probability / 100.0 for pair in self._pairs]
        return np.cumsum(probs, dtype=np.float64)

    def sample(self) -> tuple[int, int]:
        """
        Sample an (ISL, OSL) pair according to the distribution.

        Returns:
            Tuple of (input_seq_len, output_seq_len)
        """
        rand_val = self._get_rng().random()

        # Binary search for efficiency with large distributions
        idx = np.searchsorted(self._cumulative_probs, rand_val, side="right")
        idx = min(idx, len(self._pairs) - 1)  # Handle edge case

        pair = self._pairs[idx]

        # Sample from normal distribution if stddev is specified
        if pair.input_seq_len_stddev > 0:
            isl = self._get_rng().sample_positive_normal_integer(
                pair.input_seq_len, pair.input_seq_len_stddev
            )
        else:
            isl = pair.input_seq_len

        if pair.output_seq_len_stddev > 0:
            osl = self._get_rng().sample_positive_normal_integer(
                pair.output_seq_len, pair.output_seq_len_stddev
            )
        else:
            osl = pair.output_seq_len

        return (isl, osl)

    def sample_batch(self, batch_size: int) -> list[tuple[int, int]]:
        """
        Sample multiple (ISL, OSL) pairs efficiently.

        Args:
            batch_size: Number of pairs to sample

        Returns:
            List of (input_seq_len, output_seq_len) tuples
        """
        if batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {batch_size}")

        rng_inst = self._get_rng()
        rand_vals = rng_inst.random_batch(batch_size)
        indices = np.searchsorted(self._cumulative_probs, rand_vals, side="right")
        indices = np.clip(indices, 0, len(self._pairs) - 1)

        samples: list[tuple[int, int]] = []
        for idx in indices:
            pair = self._pairs[idx]
            if pair.input_seq_len_stddev > 0:
                isl = rng_inst.sample_positive_normal_integer(
                    pair.input_seq_len, pair.input_seq_len_stddev
                )
            else:
                isl = pair.input_seq_len

            if pair.output_seq_len_stddev > 0:
                osl = rng_inst.sample_positive_normal_integer(
                    pair.output_seq_len, pair.output_seq_len_stddev
                )
            else:
                osl = pair.output_seq_len

            samples.append((isl, osl))

        return samples

    @property
    def pairs(self) -> tuple[SequenceLengthPair, ...]:
        """Get immutable view of sequence length pairs."""
        return self._pairs

    def get_statistics(self) -> dict[str, int | float | list[tuple[int, int, float]]]:
        """
        Get comprehensive statistics about the distribution.

        Returns:
            Dictionary with distribution statistics including expected values,
            variance, and individual pair information.
        """
        # Expected values (convert percentages to fractions for calculation)
        exp_isl = sum(p.input_seq_len * (p.probability / 100.0) for p in self._pairs)
        exp_osl = sum(p.output_seq_len * (p.probability / 100.0) for p in self._pairs)

        # Variance calculations
        var_isl = sum(
            (p.probability / 100.0) * (p.input_seq_len - exp_isl) ** 2
            for p in self._pairs
        )
        var_osl = sum(
            (p.probability / 100.0) * (p.output_seq_len - exp_osl) ** 2
            for p in self._pairs
        )

        return {
            "num_pairs": len(self._pairs),
            "expected_isl": exp_isl,
            "expected_osl": exp_osl,
            "variance_isl": var_isl,
            "variance_osl": var_osl,
            "std_isl": np.sqrt(var_isl),
            "std_osl": np.sqrt(var_osl),
            "pairs": [
                (p.input_seq_len, p.output_seq_len, p.probability) for p in self._pairs
            ],
            "total_probability": sum(p.probability for p in self._pairs),
        }

    def __str__(self) -> str:
        """String representation showing all pairs."""
        pairs_str = ";".join(str(pair) for pair in self._pairs)
        return f"SequenceLengthDistribution[{pairs_str}]"

    def __repr__(self) -> str:
        return f"SequenceLengthDistribution({list(self._pairs)})"


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
            raise ValueError("Distribution string cannot be empty")

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
            raise ValueError("Distribution string cannot be empty")

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
            raise ValueError("JSON format must contain 'pairs' key")

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


def create_uniform_distribution(isl: int, osl: int) -> SequenceLengthDistribution:
    """
    Create a uniform distribution with a single ISL/OSL pair.

    Args:
        isl: Input sequence length
        osl: Output sequence length

    Returns:
        SequenceLengthDistribution with single pair at 100% probability
    """
    return SequenceLengthDistribution([SequenceLengthPair(isl, osl, 100.0)])


def create_balanced_distribution(
    pairs: list[tuple[int, int]],
) -> SequenceLengthDistribution:
    """
    Create a balanced distribution where all pairs have equal probability.

    Args:
        pairs: List of (isl, osl) tuples

    Returns:
        SequenceLengthDistribution with equal probabilities
    """
    if not pairs:
        raise ValueError("Cannot create distribution from empty pairs list")

    prob_per_pair = 100.0 / len(pairs)
    seq_pairs = [SequenceLengthPair(isl, osl, prob_per_pair) for isl, osl in pairs]

    return SequenceLengthDistribution(seq_pairs)


def _parse_vllm_ratio_string(value: str) -> tuple[float, float]:
    """Parse a vLLM-style CLI ratio string.

    Accepts a plain float string (``"0.3"``) applied to both dimensions, or a
    JSON object (``'{"input": 0.3, "output": 0.5}'``) for independent values.
    """
    value = value.strip()
    if not value:
        raise ValueError("--random-range-ratio value cannot be empty")
    try:
        ratio = float(value)
        return ratio, ratio
    except ValueError:
        pass
    try:
        data = orjson.loads(value)
    except orjson.JSONDecodeError as e:
        raise ValueError(
            f"--random-range-ratio must be a float or a JSON object with "
            f"'input' and 'output' keys, got: {value!r} ({e})"
        ) from e
    if not isinstance(data, dict):
        raise ValueError(
            f"--random-range-ratio must be a float or a JSON object with "
            f"'input' and 'output' keys, got: {value!r}"
        )
    missing = {"input", "output"} - data.keys()
    if missing:
        raise ValueError(
            f"--random-range-ratio JSON object missing keys: {sorted(missing)}"
        )
    extra = data.keys() - {"input", "output"}
    if extra:
        raise ValueError(
            f"--random-range-ratio JSON object has unexpected keys: {sorted(extra)}"
        )
    return float(data["input"]), float(data["output"])


def _parse_sglang_ratio_string(value: str) -> tuple[float, float]:
    """Parse an SGLang-style CLI ratio string.

    Accepts only a plain float string (``"0.3"``); independent input/output
    values via a JSON dict are rejected because SGLang applies a single ratio
    to both dimensions.
    """
    value = value.strip()
    if not value:
        raise ValueError("--random-range-ratio value cannot be empty")
    try:
        ratio = float(value)
        return ratio, ratio
    except ValueError:
        pass
    try:
        data = orjson.loads(value)
        if isinstance(data, dict):
            raise ValueError(
                "SGLang corpus style applies a single ratio to both ISL and OSL. "
                "Independent input/output values are not supported; "
                "provide a plain float instead (e.g. 0.3)."
            )
    except orjson.JSONDecodeError:
        pass
    raise ValueError(
        f"--random-range-ratio must be a plain float for SGLang corpus style, got: {value!r}"
    )


def _coerce_ratio_input(v: Any, parser: Any) -> tuple[float, float] | Any:
    """Coerce ``range_ratio`` field input to a ``(isl_ratio, osl_ratio)`` tuple.

    Handles float/int (symmetric), string (via ``parser``), and 2-element
    list/tuple.  Returns ``v`` unchanged for other types so Pydantic's own type
    error fires.
    """
    if isinstance(v, (int, float)):
        return float(v), float(v)
    if isinstance(v, str):
        return parser(v)
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return float(v[0]), float(v[1])
    return v


class VLLMRatioConfig(BaseConfig):
    """Config for vLLM-style range-ratio sampling.

    ``range_ratio`` accepts:

    - a plain float (``0.3``) — applied to both ISL and OSL
    - a CLI string (``"0.3"`` or ``'{"input": 0.3, "output": 0.5}'``)
    - a 2-tuple ``(isl_ratio, osl_ratio)``

    Ratios must be in ``[0.0, 1.0)``.
    """

    model_config = ConfigDict(extra="forbid")

    isl_mean: Annotated[int, Field(ge=1, description="Mean input sequence length.")]
    osl_mean: Annotated[int, Field(ge=1, description="Mean output sequence length.")]
    isl_stddev: Annotated[
        float,
        Field(
            default=0.0,
            ge=0.0,
            description="Must be 0; asserts caller resolved a fixed ISL mean.",
        ),
    ] = 0.0
    osl_stddev: Annotated[
        float,
        Field(
            default=0.0,
            ge=0.0,
            description="Must be 0; asserts caller resolved a fixed OSL mean.",
        ),
    ] = 0.0
    num_special_tokens: Annotated[
        int,
        Field(
            default=0,
            ge=0,
            description="Special tokens subtracted from isl_mean before bounds.",
        ),
    ] = 0
    chat_template_len: Annotated[
        int,
        Field(
            default=0,
            ge=0,
            description="Present for API symmetry with SGLangRatioConfig; unused in VLLM-style bounds.",
        ),
    ] = 0
    range_ratio: Annotated[
        tuple[float, float],
        Field(
            description="(isl_ratio, osl_ratio); symmetric window [floor(mean*(1-r)), ceil(mean*(1+r))]."
        ),
    ]

    @model_validator(mode="after")
    def _no_stddev(self) -> Self:
        if self.isl_stddev != 0.0:
            raise ValueError(
                "--isl-stddev cannot be combined with --random-range-ratio; "
                "the ratio window already controls ISL variance."
            )
        if self.osl_stddev != 0.0:
            raise ValueError(
                "--osl-stddev cannot be combined with --random-range-ratio; "
                "the ratio window already controls OSL variance."
            )
        return self

    @field_validator("range_ratio", mode="before")
    @classmethod
    def _coerce_and_validate(cls, v: Any) -> tuple[float, float]:
        ir, or_ = _coerce_ratio_input(v, _parse_vllm_ratio_string)
        if not math.isfinite(ir) or not (0.0 <= ir < 1.0):
            raise ValueError(f"ISL range_ratio must be in [0.0, 1.0), got {ir}")
        if not math.isfinite(or_) or not (0.0 <= or_ < 1.0):
            raise ValueError(f"OSL range_ratio must be in [0.0, 1.0), got {or_}")
        return ir, or_

    def compute_input_bounds(self) -> tuple[int, int]:
        adjusted = max(0, self.isl_mean - self.num_special_tokens)
        r = self.range_ratio[0]
        return max(0, math.floor(adjusted * (1 - r))), math.ceil(adjusted * (1 + r))

    def compute_output_bounds(self) -> tuple[int, int]:
        r = self.range_ratio[1]
        return (
            max(1, math.floor(self.osl_mean * (1 - r))),
            max(1, math.ceil(self.osl_mean * (1 + r))),
        )


class SGLangRatioConfig(BaseConfig):
    """Config for SGLang-style range-ratio sampling.

    ``range_ratio`` accepts:

    - a plain float (``0.3``) — applied to both ISL and OSL
    - a plain-float CLI string (``"0.3"``); JSON dict form is rejected
    - a 2-tuple ``(r, r)`` where both elements must be equal

    Ratios must be in ``[0.0, 1.0]``. ``chat_template_len`` is subtracted from
    ``isl_mean`` before bounds are computed.
    """

    model_config = ConfigDict(extra="forbid")

    isl_mean: Annotated[int, Field(ge=1, description="Mean input sequence length.")]
    osl_mean: Annotated[int, Field(ge=1, description="Mean output sequence length.")]
    isl_stddev: Annotated[
        float,
        Field(
            default=0.0,
            ge=0.0,
            description="Must be 0; asserts caller resolved a fixed ISL mean.",
        ),
    ] = 0.0
    osl_stddev: Annotated[
        float,
        Field(
            default=0.0,
            ge=0.0,
            description="Must be 0; asserts caller resolved a fixed OSL mean.",
        ),
    ] = 0.0
    num_special_tokens: Annotated[
        int,
        Field(
            default=0,
            ge=0,
            description="Ignored; present for API symmetry with VLLMRatioConfig.",
        ),
    ] = 0
    chat_template_len: Annotated[
        int,
        Field(
            default=0,
            ge=0,
            description="Chat template token overhead subtracted from isl_mean.",
        ),
    ] = 0
    range_ratio: Annotated[
        tuple[float, float],
        Field(
            description="(isl_ratio, osl_ratio); lower-bounded window [max(1, int(mean*r)), mean]."
        ),
    ]

    @model_validator(mode="after")
    def _no_stddev(self) -> Self:
        if self.isl_stddev != 0.0:
            raise ValueError(
                "--isl-stddev cannot be combined with --random-range-ratio; "
                "the ratio window already controls ISL variance."
            )
        if self.osl_stddev != 0.0:
            raise ValueError(
                "--osl-stddev cannot be combined with --random-range-ratio; "
                "the ratio window already controls OSL variance."
            )
        return self

    @field_validator("range_ratio", mode="before")
    @classmethod
    def _coerce_and_validate(cls, v: Any) -> tuple[float, float]:
        ir, or_ = _coerce_ratio_input(v, _parse_sglang_ratio_string)
        if not math.isfinite(ir) or not (0.0 <= ir <= 1.0):
            raise ValueError(f"range_ratio must be in [0.0, 1.0], got {ir}")
        if ir != or_:
            raise ValueError(
                f"SGLang corpus style requires equal ISL and OSL ratios, got ({ir}, {or_})"
            )
        return ir, or_

    def compute_input_bounds(self) -> tuple[int, int]:
        adjusted = max(1, self.isl_mean - self.chat_template_len)
        r = self.range_ratio[0]
        return max(1, int(adjusted * r)), adjusted

    def compute_output_bounds(self) -> tuple[int, int]:
        r = self.range_ratio[1]
        return max(1, int(self.osl_mean * r)), self.osl_mean


class RangeRatioDistribution:
    """Uniform ISL/OSL sampling in a ratio-defined integer window around configured means.

    Instantiate via the registry for mode-driven dispatch::

        DistClass = _CLASS_FOR_MODE[mode]
        config = DistClass.get_config_class()(isl_mean=512, osl_mean=128, range_ratio="0.3")
        dist = DistClass(config)

    Two concrete styles:

    - :class:`RangeRatioDistribution` (VLLM): symmetric window
      ``[floor(mean*(1-r)), ceil(mean*(1+r))]``, ``r ∈ [0.0, 1.0)``.
    - :class:`SGLangRangeRatioDistribution` (SGLANG): lower-bounded window
      ``[max(1, int(mean*r)), mean]``, ``r ∈ [0.0, 1.0]``.
    """

    _style: ClassVar[RandomCorpusStyle] = RandomCorpusStyle.VLLM

    @classmethod
    def get_config_class(cls) -> type[VLLMRatioConfig]:
        return VLLMRatioConfig

    def __init__(self, config: VLLMRatioConfig) -> None:
        self._rng = rng.derive("models.range_ratio.distribution")
        self._isl_mean = config.isl_mean
        self._osl_mean = config.osl_mean
        self._config = config

        self._input_low, self._input_high = config.compute_input_bounds()
        self._output_low, self._output_high = config.compute_output_bounds()

    def preseed(self, n: int, seed: int | None) -> None:
        """Pre-generate all ISL then all OSL values using vLLM's PCG64 draw order.

        Creates ``numpy.random.default_rng(seed)`` internally so the RNG
        algorithm and seeding are encapsulated here rather than in the caller.
        Stores the generator after ISL/OSL draws as ``_preseed_rng`` so that
        :meth:`PromptGenerator.preseed` can continue drawing offsets from the
        same stream without the caller needing to manage generator state.

        Prefix prompts do not participate: they are additive and prepended to the
        body after generation, so the cached ISLs describe the body alone.

        Subclasses override this method to use a different RNG algorithm
        (e.g. :class:`SGLangRangeRatioDistribution` uses MT19937 to match
        SGLang's ``benchmark_serving.py``).
        """
        g = np.random.default_rng(seed)
        self._isl_cache = g.integers(
            self._input_low, self._input_high + 1, size=n
        ).tolist()
        self._osl_cache = g.integers(
            self._output_low, self._output_high + 1, size=n
        ).tolist()
        self._cache_idx = 0
        self._preseed_rng: object = g

    def sample(self) -> tuple[int, int]:
        """Sample a single (ISL, OSL) pair with independent uniform integers."""
        cache = getattr(self, "_isl_cache", None)
        if cache is not None:
            idx = self._cache_idx
            self._cache_idx = idx + 1
            return self._isl_cache[idx], self._osl_cache[idx]
        isl = int(self._rng.integers(self._input_low, self._input_high + 1))
        osl = int(self._rng.integers(self._output_low, self._output_high + 1))
        return isl, osl

    @property
    def input_bounds(self) -> tuple[int, int]:
        """Inclusive [low, high] integer bounds for ISL sampling."""
        return self._input_low, self._input_high

    @property
    def output_bounds(self) -> tuple[int, int]:
        """Inclusive [low, high] integer bounds for OSL sampling."""
        return self._output_low, self._output_high

    @property
    def mode(self) -> RandomCorpusStyle:
        return type(self)._style

    def __repr__(self) -> str:
        return (
            f"RangeRatioDistribution(mode={type(self)._style}, isl_mean={self._isl_mean}, "
            f"osl_mean={self._osl_mean}, range_ratio={self._config.range_ratio})"
        )


class _LegacyRNG:
    """Thin wrapper over ``numpy.random`` (MT19937 global state) exposing the
    same ``.integers()`` interface as ``numpy.random.Generator`` (PCG64).

    Used by :class:`SGLangRangeRatioDistribution` so that preseed callers
    (including :meth:`PromptGenerator.preseed`) can treat both RNG backends
    identically without branching on algorithm.
    """

    def integers(
        self,
        low: int,
        high: int | None = None,
        size: int | None = None,
    ) -> np.ndarray:
        if high is None:
            return np.random.randint(0, low, size=size)
        return np.random.randint(low, high, size=size)


class SGLangRangeRatioDistribution(RangeRatioDistribution):
    """RangeRatioDistribution with SGLang-compatible MT19937 preseed.

    Also mirrors SGLang's ``use_chat_template`` bound adjustment: when
    ``chat_template_len > 0``, it is subtracted from ``isl_mean`` before
    the sampling window is computed — matching::

        chat_template_len = len(tokenizer.encode(_apply_chat_template("a"))) - 1
        input_len = input_len - chat_template_len      # SGLang adjusts mean first
        lower = int(input_len * ratio)
        upper = input_len

    Overrides :meth:`preseed` to use ``numpy.random`` (MT19937 global state)
    instead of ``numpy.random.default_rng`` (PCG64), matching the draw order
    in SGLang's ``benchmark_serving.py``::

        input_lens  = np.random.randint(lower, upper + 1, size=n)
        output_lens = np.random.randint(lower, upper + 1, size=n)
        offsets     = np.random.randint(0, vocab_size, size=n)

    When ``seed`` is provided, the global RNG is seeded before the draws so
    that aiperf runs are reproducible even though SGLang itself never seeds
    it before sampling. The seed is folded through
    :func:`~aiperf.common.random_generator.fold_seed_to_uint32` first, since
    ``numpy.random.seed`` rejects the 64-bit seeds that adaptive sweeps and
    ``multi_run.vary_seed_per_trial`` produce.
    """

    _style: ClassVar[RandomCorpusStyle] = RandomCorpusStyle.SGLANG

    @classmethod
    def get_config_class(cls) -> type[SGLangRatioConfig]:
        return SGLangRatioConfig

    def __init__(self, config: SGLangRatioConfig) -> None:
        super().__init__(config)

    def preseed(self, n: int, seed: int | None) -> None:
        if seed is not None:
            # numpy's legacy global seeder caps at 2**32-1, but run seeds are
            # 64-bit on the adaptive-sweep and vary_seed_per_trial paths (and
            # --random-seed is only bounded ge=0), so fold before seeding.
            # The PCG64 parent path takes 64-bit seeds directly and needs no fold.
            np.random.seed(rng.fold_seed_to_uint32(seed))
        g = _LegacyRNG()
        self._isl_cache = g.integers(
            self._input_low, self._input_high + 1, size=n
        ).tolist()
        self._osl_cache = g.integers(
            self._output_low, self._output_high + 1, size=n
        ).tolist()
        self._cache_idx = 0
        self._preseed_rng: object = g


_CLASS_FOR_MODE: dict[RandomCorpusStyle, type[RangeRatioDistribution]] = {
    RandomCorpusStyle.VLLM: RangeRatioDistribution,
    RandomCorpusStyle.SGLANG: SGLangRangeRatioDistribution,
}
