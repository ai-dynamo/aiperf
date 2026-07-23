# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import orjson

from aiperf.dataset.loader.weka_trace_models import WekaNormalRequest


def _write_trace(tmp_path, data, name="t.json"):
    p = tmp_path / name
    p.write_bytes(orjson.dumps(data))
    return p


def _req(
    t: float, hash_ids: list[int], api_time: float = 1.0, model: str = "m"
) -> WekaNormalRequest:
    return WekaNormalRequest(
        type="n",
        t=t,
        model=model,
        input_length=len(hash_ids) * 64,
        output_length=10,
        hash_ids=hash_ids,
        api_time=api_time,
    )


def _normals(*reqs: WekaNormalRequest) -> list[tuple[int, WekaNormalRequest]]:
    return list(enumerate(reqs))


def _chain_outer_indices(result, chain_index: int) -> list[int]:
    return [oi for oi, _ in result.chains[chain_index].requests]
