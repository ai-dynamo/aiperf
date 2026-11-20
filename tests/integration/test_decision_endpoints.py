# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real AIPerf processes against CPU-only loopback decision fixtures."""

import asyncio
from copy import deepcopy
from pathlib import Path

import orjson
import pytest
from aiohttp import web

from tests.harness.utils import AIPerfRunnerFn


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("dialect", ["decisions", "systemone"])
@pytest.mark.parametrize("export_level", [None, "raw"])
async def test_structured_replay_errors_and_non_token_metrics(
    tmp_path: Path,
    dialect: str,
    export_level: str | None,
    aiperf_runner: AIPerfRunnerFn,
    monkeypatch: pytest.MonkeyPatch,
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
    mismatched = deepcopy(result)
    if dialect == "systemone":
        mismatched["answers"] = {"wrong": result["answers"]["q"]}
    else:
        mismatched["answers"][0]["name"] = "wrong"
    seen = []

    async def handle(request: web.Request) -> web.Response:
        seen.append(await request.json(loads=orjson.loads))
        index = len(seen)
        return web.Response(
            body=orjson.dumps({2: {}, 4: mismatched}.get(index, result)),
            content_type="application/json",
            status=429 if index == 3 else 200,
        )

    app = web.Application()
    app.router.add_post(f"/v1/{dialect}", handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]
    try:
        dataset = tmp_path / "payload.json"
        await asyncio.to_thread(
            dataset.write_bytes,
            orjson.dumps({"data": [{"session_id": "fixture", "payloads": [body]}]}),
        )
        command = [
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
            "5",
            "--concurrency",
            "1",
            "--workers-max",
            "1",
            "--ui",
            "simple",
            "--no-gpu-telemetry",
        ]
        if export_level is not None:
            command.extend(["--export-level", export_level])
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
        monkeypatch.delenv("AIPERF_RECORD_STRIP_PAYLOAD_BYTES", raising=False)
        run = await aiperf_runner(command, 120)
        output = run.stdout + run.stderr
        await asyncio.to_thread((tmp_path / "aiperf.log").write_text, output)
        assert run.exit_code == 0, output[-12000:]
    finally:
        await runner.cleanup()
    assert seen == [body] * 5
    exported = await asyncio.to_thread(
        (run.output_dir / "profile_export.jsonl").read_bytes
    )
    records = [orjson.loads(line) for line in exported.splitlines()]
    assert len(records) == 5
    errors = [record["error"] for record in records if record.get("error")]
    assert len(errors) == 3
    assert any("Missing decision model" in error["message"] for error in errors)
    mismatch_message = (
        "Question identifiers differ"
        if dialect == "systemone"
        else "Question names/order differ"
    )
    assert any(mismatch_message in error["message"] for error in errors)
    assert any(error.get("code") == 429 for error in errors)
    for record in records:
        if not record.get("error"):
            assert record["metrics"]["request_latency"]["value"] > 0
        assert not {
            "time_to_first_token",
            "inter_token_latency",
        }.intersection(record.get("metrics", {}))
    summary = orjson.loads(
        await asyncio.to_thread(
            (run.output_dir / "profile_export_aiperf.json").read_bytes
        )
    )
    assert "output_token_throughput" not in summary
    assert summary["request_count"]["avg"] == 2
    assert summary["error_request_count"]["avg"] == 3
    if export_level == "raw":
        raw_exported = await asyncio.to_thread(
            (run.output_dir / "profile_export_raw.jsonl").read_bytes
        )
        raw_records = [orjson.loads(line) for line in raw_exported.splitlines()]
        assert len(raw_records) == 5
        assert all(record["payload"] == body for record in raw_records)
        mismatch_record = next(
            record
            for record in raw_records
            if mismatch_message in record.get("error", {}).get("message", "")
        )
        assert mismatch_record["status"] == 200
        assert orjson.loads(mismatch_record["responses"][0]["text"]) == mismatched
