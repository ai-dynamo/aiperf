# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real AIPerf processes against CPU-only loopback decision fixtures."""

import asyncio
import os
import sys
from pathlib import Path

import orjson
import pytest
from aiohttp import web


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("dialect", ["decisions", "systemone"])
async def test_structured_replay_errors_and_non_token_metrics(
    tmp_path: Path, dialect: str
) -> None:
    body = {
        "model": "fixture",
        "input": "evidence",
        "questions": [{"type": "predicate", "instructions": "Useful?"}],
    }
    result = {
        "model": "fixture",
        "answers": [{"type": "predicate", "name": None, "probability": 0.8}],
        "usage": {"input_tokens": 10, "output_tokens": 0},
    }
    if dialect == "systemone":
        body = {
            "model": "fixture",
            "state": "evidence",
            "questions": {"q": {"type": "noul", "instructions": "Useful?"}},
        }
        result["answers"] = {"q": {"type": "noul", "noul": 0.8}}
    seen = []

    async def handle(request: web.Request) -> web.Response:
        seen.append(await request.json(loads=orjson.loads))
        index = len(seen)
        return web.json_response(
            {} if index == 2 else result, status=429 if index == 3 else 200
        )

    app = web.Application()
    app.router.add_post(f"/v1/{dialect}", handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]
    dataset = tmp_path / "payload.json"
    dataset.write_bytes(
        orjson.dumps({"data": [{"session_id": "fixture", "payloads": [body]}]})
    )
    artifact = tmp_path / "artifacts"
    command = [
        sys.executable,
        "-m",
        "aiperf",
        "profile",
        "--model",
        "fixture",
        "--endpoint-type",
        dialect,
        "--url",
        f"http://127.0.0.1:{port}",
        "--input-file",
        str(dataset),
        "--custom-dataset-type",
        "inputs_json",
        "--request-count",
        "4",
        "--concurrency",
        "1",
        "--workers-max",
        "1",
        "--ui",
        "simple",
        "--no-gpu-telemetry",
        "--artifact-dir",
        str(artifact),
        "--export-level",
        "raw",
    ]
    process = None
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        )
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=120)
        (tmp_path / "aiperf.log").write_bytes(stdout)
        assert process.returncode == 0, stdout.decode()[-12000:]
    finally:
        if process is not None and process.returncode is None:
            process.kill()
            await process.wait()
        await runner.cleanup()
    assert seen == [body] * 4
    records = [
        orjson.loads(line)
        for line in (artifact / "profile_export.jsonl").read_bytes().splitlines()
    ]
    assert len(records) == 4
    assert sum(bool(record.get("error")) for record in records) == 2
    for record in records:
        assert not {
            "time_to_first_token",
            "inter_token_latency",
            "output_token_throughput",
        }.intersection(record.get("metrics", {}))
