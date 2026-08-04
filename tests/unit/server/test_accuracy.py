# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wire tests for mock-server accuracy-dataset mode."""

from __future__ import annotations

from pathlib import Path

import pytest
from aiperf_mock_server.accuracy import (
    AccuracyDataset,
    AccuracyFormat,
    AccuracySettings,
    set_accuracy_state,
)
from aiperf_mock_server.config import MockServerConfig, server_config
from fastapi.testclient import TestClient

pytestmark = pytest.mark.server_unit


def _load_accuracy(
    tmp_path: Path,
    body: str,
    *,
    correct_rate: float = 1.0,
    cot_rate: float = 0.0,
    adversarial_rate: float = 0.0,
    reasoning_field: bool = True,
    random_seed: int = 7,
) -> None:
    path = tmp_path / "accuracy.jsonl"
    path.write_text(body, encoding="utf-8")
    settings = AccuracySettings(
        default_format=AccuracyFormat.MMLU,
        correct_rate=correct_rate,
        cot_rate=cot_rate,
        adversarial_rate=adversarial_rate,
        reasoning_field=reasoning_field,
        random_seed=random_seed,
    )
    dataset = AccuracyDataset.from_jsonl(body, settings)
    set_accuracy_state(dataset, settings)
    # Keep global config seed aligned for any other RNG uses.
    server_config.random_seed = random_seed
    server_config.accuracy_dataset = str(path)
    server_config.accuracy_format = "mmlu"
    server_config.accuracy_correct_rate = correct_rate
    server_config.accuracy_cot_rate = cot_rate
    server_config.accuracy_adversarial_rate = adversarial_rate
    server_config.accuracy_reasoning_field = reasoning_field


def _chat(client: TestClient, prompt: str, *, stream: bool = False) -> dict | str:
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4",
            "messages": [{"role": "user", "content": prompt}],
            "stream": stream,
        },
    )
    assert resp.status_code == 200
    if stream:
        return resp.text
    return resp.json()


def test_correct_answer_is_wired_through(
    test_client: TestClient, tmp_path: Path
) -> None:
    _load_accuracy(
        tmp_path, '{"text": "What is 2+2?", "ground_truth": "B"}', correct_rate=1.0
    )
    data = _chat(test_client, "What is 2+2?")
    assert isinstance(data, dict)
    msg = data["choices"][0]["message"]
    assert msg["content"] == "The answer is (B)"
    assert msg.get("reasoning_content") is None


def test_wrong_answer_at_zero_rate(test_client: TestClient, tmp_path: Path) -> None:
    _load_accuracy(
        tmp_path, '{"text": "What is 2+2?", "ground_truth": "B"}', correct_rate=0.0
    )
    data = _chat(test_client, "What is 2+2?")
    assert isinstance(data, dict)
    content = data["choices"][0]["message"]["content"]
    assert content.startswith("The answer is (")
    assert content != "The answer is (B)"


def test_cot_populates_reasoning_content(
    test_client: TestClient, tmp_path: Path
) -> None:
    _load_accuracy(
        tmp_path,
        '{"text": "Q", "ground_truth": "B"}',
        correct_rate=1.0,
        cot_rate=1.0,
        reasoning_field=True,
    )
    data = _chat(test_client, "Q")
    assert isinstance(data, dict)
    msg = data["choices"][0]["message"]
    assert msg["content"] == "The answer is (B)"
    assert "The answer is (B)" in (msg.get("reasoning_content") or "")


def test_unmatched_prompt_falls_through_to_corpus(
    test_client: TestClient, tmp_path: Path
) -> None:
    _load_accuracy(
        tmp_path, '{"text": "known prompt", "ground_truth": "B"}', correct_rate=1.0
    )
    data = _chat(test_client, "an entirely different prompt string")
    assert isinstance(data, dict)
    assert data["choices"][0]["message"]["content"] != "The answer is (B)"


def test_live_accuracy_endpoint_and_prometheus_reflect_served_requests(
    test_client: TestClient, tmp_path: Path
) -> None:
    body = "\n".join(
        [
            '{"text": "p one", "ground_truth": "B", "task": "demo"}',
            '{"text": "p two", "ground_truth": "B", "task": "demo"}',
            '{"text": "p three", "ground_truth": "B", "task": "demo"}',
        ]
    )
    _load_accuracy(tmp_path, body, correct_rate=1.0)
    for prompt in ("p one", "p two", "p three"):
        _chat(test_client, prompt)
    _chat(test_client, "unknown")

    status = test_client.get("/accuracy").json()
    assert status["enabled"] is True
    assert status["matched"] == 3
    assert status["correct"] == 3
    assert status["incorrect"] == 0
    assert status["accuracy"] == 1.0
    assert status["unmatched"] == 1
    assert status["tasks"]["demo"]["matched"] == 3
    assert status["tasks"]["demo"]["correct"] == 3

    metrics = test_client.get("/metrics").text
    assert "aiperf_mock_accuracy_matched_total 3.0" in metrics
    assert "aiperf_mock_accuracy_correct_total 3.0" in metrics
    assert "aiperf_mock_accuracy_ratio 1.0" in metrics
    assert 'aiperf_mock_accuracy_task_correct_total{task="demo"} 3.0' in metrics

    set_accuracy_state(None, None)
    metrics = test_client.get("/metrics").text
    assert 'aiperf_mock_accuracy_task_correct_total{task="demo"}' not in metrics


def test_accuracy_endpoint_disabled_without_dataset(test_client: TestClient) -> None:
    status = test_client.get("/accuracy").json()
    assert status == {"enabled": False}


def test_accuracy_reset_zeroes_the_tally_and_keeps_the_dataset(
    test_client: TestClient, tmp_path: Path
) -> None:
    """A caller can scope the tally to one phase without restarting the server."""
    _load_accuracy(tmp_path, '{"text": "the q", "ground_truth": "B"}', correct_rate=1.0)
    _chat(test_client, "the q")
    _chat(test_client, "unmatched prompt")
    assert test_client.get("/accuracy").json()["matched"] == 1

    assert test_client.post("/accuracy/reset").json() == {
        "enabled": True,
        "reset": True,
    }

    status = test_client.get("/accuracy").json()
    assert status["enabled"] is True
    assert status["matched"] == 0
    assert status["correct"] == 0
    assert status["unmatched"] == 0
    assert status["tasks"] == {}
    assert status["config"]["dataset_rows"] == 1  # dataset survives the reset

    # and the mock still serves from the dataset afterwards
    data = _chat(test_client, "the q")
    assert isinstance(data, dict)
    assert data["choices"][0]["message"]["content"] == "The answer is (B)"
    assert test_client.get("/accuracy").json()["matched"] == 1


def test_accuracy_reset_without_dataset_is_a_noop(test_client: TestClient) -> None:
    set_accuracy_state(None, None)
    assert test_client.post("/accuracy/reset").json() == {
        "enabled": False,
        "reset": False,
    }


def test_adversarial_null_object_frame_is_served_in_stream(
    test_client: TestClient, tmp_path: Path
) -> None:
    lines = [f'{{"text": "prompt number {i}", "ground_truth": "B"}}' for i in range(40)]
    _load_accuracy(tmp_path, "\n".join(lines), correct_rate=1.0, adversarial_rate=1.0)
    saw_null = False
    for i in range(40):
        body = _chat(test_client, f"prompt number {i}", stream=True)
        assert isinstance(body, str)
        assert "[DONE]" in body
        if '"object":null' in body or '"object": null' in body:
            saw_null = True
    assert saw_null


def test_accuracy_does_not_override_embeddings(
    test_client: TestClient, tmp_path: Path
) -> None:
    _load_accuracy(
        tmp_path, '{"text": "known prompt", "ground_truth": "B"}', correct_rate=1.0
    )
    response = test_client.post(
        "/v1/embeddings",
        json={"model": "text-embedding", "input": "known prompt"},
    )
    assert response.status_code == 200
    assert test_client.get("/accuracy").json()["matched"] == 0


def test_accuracy_dataset_forces_workers_one() -> None:
    cfg = MockServerConfig(accuracy_dataset="/tmp/x.jsonl", workers=4)
    assert cfg.workers == 1
