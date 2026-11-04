# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real HTTP speech responses cross ZMQ and reach audio metrics and error counts."""

import io
import wave
from collections.abc import AsyncIterator
from pathlib import Path

import orjson
import pytest
from aiohttp import web
from pytest import param

from aiperf.common.models import BinaryResponse
from tests.harness.utils import AIPerfCLI

_PCM = b"\xff\x80" * 96000  # one second of 48 kHz stereo signed 16-bit PCM


def _wav() -> bytes:
    """Return half a second of 16 kHz mono PCM in a finite WAV container."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(b"\xff\x80" * 8000)
    return buffer.getvalue()


@pytest.fixture
async def speech_server() -> AsyncIterator[tuple[str, list[dict]]]:
    """Serve valid audio, malformed audio, HTTP rejection and JSON errors."""
    requests: list[dict] = []

    async def handle(request: web.Request) -> web.Response:
        """Select deterministic fixtures by input text rather than arrival order."""
        payload = await request.json()
        requests.append(payload)
        assert "pcm_output" not in payload and "pcmOutput" not in payload
        assert payload["voice"] == "fixture-voice"
        match payload["input"]:
            case "wav":
                return web.Response(body=_wav(), content_type="audio/wav")
            case "pcm":
                return web.Response(body=_PCM, content_type="audio/pcm")
            case "truncated":
                return web.Response(body=_wav()[:-2], content_type="audio/wav")
            case "partial":
                return web.Response(body=b"\xff\x80", content_type="audio/pcm")
            case "rejected":
                return web.Response(status=429, text="busy")
            case _:
                return web.json_response({"error": "no audio"})

    app = web.Application()
    app.router.add_post("/v1/audio/speech", handle)
    runner = web.AppRunner(app)
    await runner.setup()
    try:
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        yield f"http://127.0.0.1:{runner.addresses[0][1]}", requests
    finally:
        await runner.cleanup()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("export_level", [param("records", id="records"), param("raw", id="raw")])  # fmt: skip
async def test_speech_http_metrics_and_error_accounting(
    cli: AIPerfCLI,
    speech_server: tuple[str, list[dict]],
    tmp_path: Path,
    export_level: str,
) -> None:
    """Only complete audio contributes duration, while every request is accounted for."""
    url, requests = speech_server
    dataset = tmp_path / "speech.jsonl"
    cases = ("wav", "pcm", "truncated", "partial", "rejected", "json-error")
    dataset.write_bytes(
        b"\n".join(
            orjson.dumps(
                {
                    "text": name,
                    "extra": {
                        "voice": "fixture-voice",
                        "response_format": "pcm"
                        if name in {"pcm", "partial"}
                        else "wav",
                    },
                }
            )
            for name in cases
        )
    )
    config = tmp_path / "speech.json"
    config.write_bytes(
        orjson.dumps(
            {
                "schemaVersion": "2.0",
                "benchmark": {
                    "model": "fixture-tts",
                    "dataset": {
                        "type": "file",
                        "format": "single_turn",
                        "path": str(dataset),
                    },
                    "profiling": {
                        "type": "concurrency",
                        "concurrency": 1,
                        "requests": 6,
                    },
                    "endpoint": {
                        "type": "speech",
                        "url": url,
                        "pcmOutput": {"sampleRate": 48000, "channels": 2},
                    },
                },
            }
        )
    )
    result = await cli.run(
        f"""
        aiperf profile --config {config} --endpoint-type speech --url {url}
            --input-file {dataset} --custom-dataset-type single_turn
            --dataset-sampling-strategy sequential
            --request-count 6 --concurrency 1 --workers-max 1
            --export-level {export_level} --ui none
        """,
        timeout=60,
    )
    assert result.exit_code == 0
    assert len(requests) == 6
    assert result.json is not None
    assert result.json.request_count.avg == 2
    assert result.json.error_request_count.avg == 4
    assert result.json.output_audio_duration.sum == pytest.approx(1.5)
    assert result.json.total_output_audio_duration.avg == pytest.approx(1.5)
    assert result.json.output_audio_rtf.count == 2
    assert result.json.output_audio_throughput.avg > 0
    assert result.json.time_to_first_token is None
    assert result.json.output_token_throughput is None
    assert result.json.effective_concurrency is not None
    exported = result.json.model_dump()
    assert exported.get("effective_decode_throughput") is None
    assert exported.get("active_decode_throughput") is None
    assert exported.get("tokens_in_flight") is None
    assert result.jsonl is not None and len(result.jsonl) == 6
    for record in result.jsonl:
        if record.error is not None:
            assert "output_audio_duration" not in record.metrics
        else:
            assert record.metrics["output_audio_duration"].value in (0.5, 1.0)
            assert "time_to_first_audio" not in record.metrics
            assert "audio_duration" not in record.metrics
            assert "rtfx" not in record.metrics
    if export_level == "raw":
        assert result.raw_records is not None and len(result.raw_records) == 6
        bodies = [
            response.raw_bytes
            for record in result.raw_records
            for response in record.responses
            if isinstance(response, BinaryResponse)
        ]
        assert _PCM in bodies and _wav() in bodies
