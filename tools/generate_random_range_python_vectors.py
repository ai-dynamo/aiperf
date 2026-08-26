# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate NumPy reference vectors for native random-range parity tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "rust/runtime/tests/fixtures/random_range_python_vectors.json"
VOCAB_SIZE = 16
SPECIAL_TOKEN_IDS = frozenset({3, 7, 15})
VLLM_POOL = [token for token in range(VOCAB_SIZE) if token not in SPECIAL_TOKEN_IDS]


CASES: tuple[dict[str, Any], ...] = (
    {
        "name": "vllm_scalar_zero_seed_0",
        "style": "vllm",
        "seed": 0,
        "entries": 8,
        "input_mean": 8,
        "output_mean": 4,
        "input_ratio": 0.0,
        "output_ratio": 0.0,
        "special_tokens": 0,
    },
    {
        "name": "vllm_split_seed_42_with_specials",
        "style": "vllm",
        "seed": 42,
        "entries": 8,
        "input_mean": 12,
        "output_mean": 6,
        "input_ratio": 0.25,
        "output_ratio": 0.5,
        "special_tokens": 2,
    },
    {
        "name": "vllm_near_upper_boundary_wide_seed",
        "style": "vllm",
        "seed": 18_446_744_073_709_551_613,
        "entries": 8,
        "input_mean": 12,
        "output_mean": 5,
        "input_ratio": 0.999_999,
        "output_ratio": 0.999_999,
        "special_tokens": 1,
    },
    {
        "name": "sglang_lower_boundary_seed_0",
        "style": "sglang",
        "seed": 0,
        "entries": 8,
        "input_mean": 9,
        "output_mean": 5,
        "input_ratio": 0.0,
        "output_ratio": 0.0,
        "special_tokens": 0,
    },
    {
        "name": "sglang_midpoint_seed_42_with_specials",
        "style": "sglang",
        "seed": 42,
        "entries": 8,
        "input_mean": 12,
        "output_mean": 6,
        "input_ratio": 0.5,
        "output_ratio": 0.5,
        "special_tokens": 2,
    },
    {
        "name": "sglang_upper_boundary_wide_seed",
        "style": "sglang",
        "seed": 4_294_967_300,
        "entries": 8,
        "input_mean": 11,
        "output_mean": 7,
        "input_ratio": 1.0,
        "output_ratio": 1.0,
        "special_tokens": 1,
    },
)


def fold_seed(seed: int) -> int:
    """Match AIPerf's wide-seed fold for NumPy's legacy MT19937 seeder."""
    return (seed & 0xFFFF_FFFF) ^ (seed >> 32)


def bounds(case: dict[str, Any]) -> tuple[tuple[int, int], tuple[int, int]]:
    """Compute inclusive reference bounds for one authored case."""
    if case["style"] == "vllm":
        adjusted = max(0, case["input_mean"] - case["special_tokens"])
        input_bounds = (
            max(0, int(np.floor(adjusted * (1.0 - case["input_ratio"])))),
            int(np.ceil(adjusted * (1.0 + case["input_ratio"]))),
        )
        output_bounds = (
            max(1, int(np.floor(case["output_mean"] * (1.0 - case["output_ratio"])))),
            max(1, int(np.ceil(case["output_mean"] * (1.0 + case["output_ratio"])))),
        )
        return input_bounds, output_bounds
    return (
        (max(1, int(case["input_mean"] * case["input_ratio"])), case["input_mean"]),
        (max(1, int(case["output_mean"] * case["output_ratio"])), case["output_mean"]),
    )


def generate_case(authored: dict[str, Any]) -> dict[str, Any]:
    """Run the actual NumPy reference stream and compose exact request bodies."""
    case = dict(authored)
    input_bounds, output_bounds = bounds(case)
    if case["style"] == "vllm":
        generator = np.random.default_rng(case["seed"])
        algorithm = "numpy.random.default_rng/PCG64"
        pool = VLLM_POOL
    else:
        generator = np.random.RandomState(fold_seed(case["seed"]))
        algorithm = "numpy.random.RandomState/MT19937"
        pool = list(range(VOCAB_SIZE))

    # The three vectorized calls, in this order, are the compatibility contract.
    inputs = (
        generator.integers(input_bounds[0], input_bounds[1] + 1, size=case["entries"])
        if case["style"] == "vllm"
        else generator.randint(
            input_bounds[0], input_bounds[1] + 1, size=case["entries"]
        )
    )
    outputs = (
        generator.integers(output_bounds[0], output_bounds[1] + 1, size=case["entries"])
        if case["style"] == "vllm"
        else generator.randint(
            output_bounds[0], output_bounds[1] + 1, size=case["entries"]
        )
    )
    offsets = (
        generator.integers(0, VOCAB_SIZE, size=case["entries"])
        if case["style"] == "vllm"
        else generator.randint(0, VOCAB_SIZE, size=case["entries"])
    )

    if case["style"] == "sglang":
        inputs = np.maximum(1, inputs - case["special_tokens"])

    requests = []
    for request_index, (input_len, output_len, offset) in enumerate(
        zip(inputs.tolist(), outputs.tolist(), offsets.tolist(), strict=True)
    ):
        token_ids = [
            pool[(offset + request_index + token_index) % len(pool)]
            for token_index in range(input_len)
        ]
        payload = {"max_tokens": output_len, "prompt_token_ids": token_ids}
        requests.append(
            {
                "token_ids": token_ids,
                "request_utf8": json.dumps(
                    payload, sort_keys=True, separators=(",", ":")
                ),
            }
        )

    case.update(
        {
            "algorithm": algorithm,
            "input_bounds": list(input_bounds),
            "output_bounds": list(output_bounds),
            "inputs": inputs.tolist(),
            "outputs": outputs.tolist(),
            "offsets": offsets.tolist(),
            "token_pool": pool,
            "requests": requests,
        }
    )
    return case


def rendered_fixture() -> str:
    """Return the stable, human-reviewable fixture representation."""
    document = {
        "provenance": {
            "generator": "tools/generate_random_range_python_vectors.py",
            "numpy_version": np.__version__,
            "draw_order": "all_inputs_then_all_outputs_then_all_offsets",
            "token_formula": "pool[(offset + request_index + token_index) % len(pool)]",
            "request_encoding": "json.dumps(sort_keys=True,separators=(',',':')).encode('utf-8')",
            "vocab_size": VOCAB_SIZE,
            "vllm_special_token_ids": sorted(SPECIAL_TOKEN_IDS),
        },
        "cases": [generate_case(case) for case in CASES],
    }
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


def main() -> int:
    """Write the fixture, or prove that it still matches NumPy."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check", action="store_true", help="fail when the fixture differs"
    )
    args = parser.parse_args()
    rendered = rendered_fixture()
    if args.check:
        actual = FIXTURE.read_text(encoding="utf-8")
        if actual != rendered:
            raise SystemExit(
                f"{FIXTURE.relative_to(ROOT)} differs; regenerate with {Path(__file__).name}"
            )
    else:
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
