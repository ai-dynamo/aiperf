# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native execution preserves documented server arguments and shell commands."""

from __future__ import annotations

import asyncio
import os
import shlex
import shutil
import signal
import sys
import zipfile
from collections.abc import Generator
from contextlib import suppress
from pathlib import Path

import psutil
import pytest
from pytest import param

from aiperf.common.constants import IS_WINDOWS

REPO_ROOT = Path(__file__).resolve().parents[3]
HARNESS = REPO_ROOT / "tests/ci/test_docs_end_to_end"


@pytest.fixture(autouse=True)
def _contain_import_side_effects() -> Generator[None]:
    saved_path = list(sys.path)
    saved_modules = set(sys.modules)
    sys.path.insert(0, str(HARNESS))
    try:
        yield
    finally:
        sys.path[:] = saved_path
        for name in set(sys.modules) - saved_modules:
            del sys.modules[name]


@pytest.mark.parametrize(
    "server_name,model",
    [
        param("vllm-default-openai", "Qwen/Qwen3-0.6B", id="chat"),
        param("vllm-audio-openai", "Qwen/Qwen2.5-Omni-3B", id="audio"),
        param("vllm-embeddings-openai", "BAAI/bge-small-en-v1.5", id="embeddings"),
        param("vllm-vision-openai", "Qwen/Qwen2-VL-2B-Instruct", id="vision"),
        param("vllm-video-openai", "Qwen/Qwen2-VL-2B-Instruct", id="video"),
    ],
)  # fmt: skip
def test_native_setup_uses_documented_model(server_name: str, model: str) -> None:
    from native_setup import parse_native_setup
    from parser import MarkdownParser

    server = MarkdownParser().parse_directory(str(REPO_ROOT))[server_name]
    setup = parse_native_setup(server.setup_command.command)

    assert setup.image == "vllm/vllm-openai:latest"
    assert setup.args[setup.args.index("--model") + 1] == model
    if server_name == "vllm-audio-openai":
        assert setup.requirements == ("vllm[audio]",)
        assert "--trust-remote-code" in setup.args
    if server_name == "vllm-video-openai":
        assert setup.env == {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
        assert setup.args[setup.args.index("--media-io-kwargs") + 1] == (
            '{"video": {"num_frames": 16}}'
        )
        assert setup.args[setup.args.index("--max-num-seqs") + 1] == "1"


@pytest.mark.parametrize(
    "command",
    [
        param("echo unhandled-setup\ndocker run vllm/vllm-openai:latest --model m", id="prelude"),
        param("docker run --privileged vllm/vllm-openai:latest --model m", id="unknown-option"),
        param("docker run -p 9000:8000 vllm/vllm-openai:latest --model m", id="port-remapping"),
        param("docker run vllm/vllm-openai:latest --model m\necho ignored", id="trailing-command"),
        param("docker run vllm/vllm-openai:latest --model m && false", id="shell-operator"),
        param("docker pull other/image\ndocker run vllm/vllm-openai:latest --model m", id="image-mismatch"),
        param('docker run vllm/vllm-openai:latest --model "$MODEL"', id="variable-expansion"),
        param('docker run vllm/vllm-openai:latest --model "${MODEL}"', id="braced-expansion"),
        param('docker run vllm/vllm-openai:latest --model "$(printf model)"', id="command-expansion"),
        param('docker run vllm/vllm-openai:latest --model "`printf model`"', id="backtick-expansion"),
        param('docker run -e HF_TOKEN="$HF_TOKEN" vllm/vllm-openai:latest --model m', id="environment-expansion"),
    ],
)  # fmt: skip
def test_native_setup_rejects_unsupported_semantics(command: str) -> None:
    from native_setup import parse_native_setup

    with pytest.raises(ValueError, match="Unsupported"):
        parse_native_setup(command)


def test_audio_constraints_allow_python_dependency_repairs() -> None:
    from native_setup import compiled_server_constraints

    constraints = compiled_server_constraints(
        {
            "vllm": "0.34.0",
            "torch": "2.11.0+cu132",
            "nvidia-cublas-cu13": "13.0.0",
            "numpy": "2.2.6",
            "opentelemetry-exporter-otlp": "1.40.0",
            "librosa": "0.10.0",
        }
    )

    assert "vllm==0.34.0" in constraints
    assert "torch==2.11.0+cu132" in constraints
    assert "nvidia-cublas-cu13==13.0.0" in constraints
    assert "numpy==2.2.6" in constraints
    assert not any("opentelemetry" in constraint for constraint in constraints)
    assert not any("librosa" in constraint for constraint in constraints)


@pytest.mark.skipif(IS_WINDOWS, reason="The documentation harness requires Bash.")
@pytest.mark.asyncio
async def test_native_shell_preserves_quotes_and_fixture_files(tmp_path: Path) -> None:
    from test_runner import run_script

    script = """cat <<'EOF' > input.jsonl
{"text": "it's quoted", "value": "$literal"}
EOF
"""
    assert await run_script(script, cwd=tmp_path, prefix="TEST", timeout=5) == 0
    assert (
        await run_script(
            "cp input.jsonl result.jsonl", cwd=tmp_path, prefix="TEST", timeout=5
        )
        == 0
    )
    assert (tmp_path / "result.jsonl").read_text() == (
        '{"text": "it\'s quoted", "value": "$literal"}\n'
    )


@pytest.mark.skipif(IS_WINDOWS, reason="The documentation harness requires Bash.")
@pytest.mark.asyncio
async def test_native_shell_stops_after_failed_fixture_creation(tmp_path: Path) -> None:
    from test_runner import run_script

    assert (
        await run_script(
            "false\ntouch must-not-exist", cwd=tmp_path, prefix="TEST", timeout=5
        )
        != 0
    )
    assert not (tmp_path / "must-not-exist").exists()


@pytest.mark.skipif(IS_WINDOWS, reason="The documentation harness requires Bash.")
@pytest.mark.asyncio
async def test_native_shell_timeout_kills_descendants(tmp_path: Path) -> None:
    from test_runner import run_script

    child = "import os,time;open('pid','w').write(str(os.getpid()));time.sleep(60)"
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(child)} & wait"
    with pytest.raises(TimeoutError):
        await run_script(command, cwd=tmp_path, prefix="TEST", timeout=1)

    pid = int((tmp_path / "pid").read_text())
    assert (
        not psutil.pid_exists(pid)
        or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    )


@pytest.mark.skipif(IS_WINDOWS, reason="The documentation harness requires Bash.")
@pytest.mark.asyncio
async def test_native_shell_cancellation_stops_running_command(tmp_path: Path) -> None:
    from test_runner import run_script

    command = "echo $$ > pid; exec sleep 60"
    task = asyncio.create_task(
        run_script(command, cwd=tmp_path, prefix="TEST", timeout=60)
    )
    async with asyncio.timeout(5):
        while not (tmp_path / "pid").exists():
            await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert not psutil.pid_exists(int((tmp_path / "pid").read_text()))


@pytest.mark.skipif(IS_WINDOWS, reason="The documentation harness requires Bash.")
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cancel",
    [param(False, id="timeout"), param(True, id="cancellation")],
)  # fmt: skip
async def test_native_shell_stops_separate_child_process_group(
    tmp_path: Path, cancel: bool
) -> None:
    from test_runner import run_script

    child = (
        "import os,signal,time; os.setpgrp(); "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "open('child-pid','w').write(str(os.getpid())); time.sleep(60)"
    )
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(child)} & wait"
    task = asyncio.create_task(
        run_script(
            command, cwd=tmp_path, prefix="TEST", timeout=2 if not cancel else 60
        )
    )
    pid = None
    try:
        async with asyncio.timeout(5):
            while not (tmp_path / "child-pid").exists():
                await asyncio.sleep(0)
        pid = int((tmp_path / "child-pid").read_text())
        if cancel:
            task.cancel()
        with pytest.raises(asyncio.CancelledError if cancel else TimeoutError):
            await task

        async with asyncio.timeout(5):
            while psutil.pid_exists(pid):
                with suppress(psutil.NoSuchProcess):
                    if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                        break
                await asyncio.sleep(0)
    finally:
        task.cancel()
        if pid is not None:
            with suppress(ProcessLookupError, PermissionError):
                os.killpg(pid, signal.SIGKILL)
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.skipif(IS_WINDOWS, reason="The documentation harness requires Bash.")
@pytest.mark.asyncio
@pytest.mark.parametrize("health_passes", [True, False])
async def test_native_runner_keeps_fixtures_and_stops_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, health_passes: bool
) -> None:
    from data_types import Command, Server
    from test_runner import EndToEndTestRunner

    executable = tmp_path / "server-python"
    pid_file = shlex.quote(str(tmp_path / "server-pid"))
    directory_file = shlex.quote(str(tmp_path / "work-dir"))
    executable.write_text(
        f"#!/usr/bin/env bash\necho $$ > {pid_file}\npwd > {directory_file}\n"
        "touch server-started\nexec sleep 60\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("AIPERF_DOCS_VLLM_PYTHON", str(executable))
    monkeypatch.setenv("AIPERF_DOCS_SERVER_IMAGE", "vllm/vllm-openai:latest")

    def command(text: str) -> Command:
        return Command("test", text, "guide.md", 1, 1)

    health = "while [ ! -f server-started ]; do sleep 0.01; done"
    if not health_passes:
        health += "\nfalse"
    server = Server(
        "test",
        command("docker run vllm/vllm-openai:latest --model test"),
        command(health),
        [
            command("printf 'fixture data' > input.txt"),
            command(
                f"test -d ./fixtures/audio && cp input.txt {shlex.quote(str(tmp_path / 'result.txt'))}"
            ),
        ],
    )
    runner = EndToEndTestRunner()
    try:
        assert await runner.run_tests({server.name: server}) is health_passes
        assert not psutil.pid_exists(int((tmp_path / "server-pid").read_text()))
        assert not Path((tmp_path / "work-dir").read_text().strip()).exists()
        assert runner.work_dir is None
        if health_passes:
            assert (tmp_path / "result.txt").read_text() == "fixture data"
        else:
            assert not (tmp_path / "result.txt").exists()
    finally:
        if runner.work_dir:
            shutil.rmtree(runner.work_dir)


@pytest.mark.asyncio
async def test_native_runner_rejects_mismatched_container_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from data_types import Command, Server
    from test_runner import EndToEndTestRunner

    monkeypatch.setenv("AIPERF_DOCS_SERVER_IMAGE", "vllm/vllm-openai:different")
    setup = Command(
        "test", "docker run vllm/vllm-openai:latest --model m", "guide.md", 1, 1
    )
    command = Command("test", "true", "guide.md", 1, 1)
    server = Server("test", setup, command, [command])

    assert await EndToEndTestRunner().run_tests({server.name: server}) is False


@pytest.mark.skipif(
    IS_WINDOWS or shutil.which("uv") is None,
    reason="The native dependency installer requires Bash and uv.",
)
@pytest.mark.asyncio
@pytest.mark.parametrize("compatible", [True, False])
async def test_native_audio_overlay_reuses_compiled_packages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, compatible: bool
) -> None:
    from native_setup import NativeServerSetup
    from test_runner import EndToEndTestRunner

    wheels = tmp_path / "wheels"
    wheels.mkdir()
    base = tmp_path / "base-site"
    base.mkdir()

    def wheel(name: str, requirements: tuple[str, ...] = ()) -> Path:
        filename = wheels / f"{name}-1.0-py3-none-any.whl"
        metadata = f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n"
        metadata += "Provides-Extra: audio\n"
        metadata += "".join(
            f"Requires-Dist: {requirement}\n" for requirement in requirements
        )
        with zipfile.ZipFile(filename, "w") as archive:
            archive.writestr(f"{name}/__init__.py", "")
            archive.writestr(f"{name}-1.0.dist-info/METADATA", metadata)
            archive.writestr(
                f"{name}-1.0.dist-info/WHEEL",
                "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
            )
            archive.writestr(f"{name}-1.0.dist-info/RECORD", "")
        return filename

    for filename in (
        wheel("torch"),
        wheel("vllm", ("torch==1.0", "probe-audio==1.0; extra == 'audio'")),
    ):
        with zipfile.ZipFile(filename) as archive:
            archive.extractall(base)
    wheel("probe_audio", ("torch==1.0" if compatible else "torch>=2.0",))
    monkeypatch.setenv("PYTHONPATH", str(base))
    monkeypatch.setenv("AIPERF_DOCS_VLLM_PYTHON", sys._base_executable)
    monkeypatch.setenv("UV_NO_INDEX", "1")
    monkeypatch.setenv("UV_FIND_LINKS", wheels.as_uri())
    monkeypatch.delenv("UV_OVERRIDE", raising=False)
    runner = EndToEndTestRunner()
    runner.work_dir = tmp_path / "work"
    runner.work_dir.mkdir()
    setup = NativeServerSetup("vllm/vllm-openai:test", (), {}, ("vllm[audio]",))

    if not compatible:
        with pytest.raises(ValueError, match="dependencies failed to install"):
            await runner._server_python(setup)
        return

    python = await runner._server_python(setup)
    script = (
        "import pathlib,sys,torch,vllm,probe_audio; "
        f"base=pathlib.Path({str(base)!r}); "
        "assert pathlib.Path(torch.__file__).is_relative_to(base); "
        "assert pathlib.Path(vllm.__file__).is_relative_to(base); "
        "assert pathlib.Path(probe_audio.__file__).is_relative_to(pathlib.Path(sys.prefix))"
    )
    process = await asyncio.create_subprocess_exec(python, "-c", script)
    assert await process.wait() == 0
    assert not list(
        runner.work_dir.glob("server-venv/lib/python*/site-packages/torch*")
    )
    assert not list(runner.work_dir.glob("server-venv/lib/python*/site-packages/vllm*"))
