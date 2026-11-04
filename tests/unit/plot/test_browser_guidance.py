# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import Mock

import plotly.graph_objects as go
import plotly.io as pio
import pytest
from choreographer.browsers import chromium
from choreographer.browsers.chromium import Chromium
from kaleido.errors import ChromeNotFoundError
from pytest import param

from aiperf.common.exceptions import ConfigurationError
from aiperf.plot import browser
from aiperf.plot.exporters.png.base import BasePNGExporter


class _Exporter(BasePNGExporter):
    def export(self, *args: object, **kwargs: object) -> list[Path]:
        return []


def _plotly_missing_browser(*args: object, **kwargs: object) -> None:
    try:
        raise ChromeNotFoundError("browser not found")
    except ChromeNotFoundError:
        raise RuntimeError("Install with plotly_get_chrome")  # noqa: B904 - match Plotly's implicit chain


@pytest.mark.parametrize(
    "machine",
    [
        param("aarch64", id="aarch64"),
        param("arm64", id="arm64"),
        param("ARM64", id="uppercase"),
    ],
)  # fmt: skip
def test_png_missing_browser_linux_arm64_suggests_system_chromium(
    machine: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(browser, "IS_LINUX", True)
    monkeypatch.setattr(browser.platform, "machine", lambda: machine)
    monkeypatch.setattr("aiperf.plot.exporters.png.base.IS_WINDOWS_ARM", False)
    fig = Mock()
    fig.write_image.side_effect = _plotly_missing_browser

    with pytest.raises(ConfigurationError) as raised:
        _Exporter(tmp_path)._export_figure(fig, tmp_path / "out.png")
    message = str(raised.value)
    assert "Chromium" in message and "BROWSER_PATH" in message
    assert "plotly_get_chrome" not in message


def test_png_missing_browser_other_platform_preserves_original_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(browser, "IS_LINUX", True)
    monkeypatch.setattr(browser.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr("aiperf.plot.exporters.png.base.IS_WINDOWS_ARM", False)
    fig = Mock()
    fig.write_image.side_effect = _plotly_missing_browser

    with pytest.raises(RuntimeError, match="plotly_get_chrome"):
        _Exporter(tmp_path)._export_figure(fig, tmp_path / "out.png")


def test_png_unrelated_failure_on_linux_arm64_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(browser, "IS_LINUX", True)
    monkeypatch.setattr(browser.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr("aiperf.plot.exporters.png.base.IS_WINDOWS_ARM", False)
    fig = Mock()
    error = RuntimeError("rendering failed for another reason")
    fig.write_image.side_effect = error

    with pytest.raises(RuntimeError) as raised:
        _Exporter(tmp_path)._export_figure(fig, tmp_path / "out.png")
    assert raised.value is error


def test_png_export_with_browser_on_linux_arm64_still_renders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(browser, "IS_LINUX", True)
    monkeypatch.setattr(browser.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr("aiperf.plot.exporters.png.base.IS_WINDOWS_ARM", False)
    fig = Mock()
    path = tmp_path / "out.png"
    _Exporter(tmp_path)._export_figure(fig, path)
    fig.write_image.assert_called_once()
    assert fig.write_image.call_args.args[0] == str(path)


@pytest.mark.parametrize(
    "linux,machine,expected,absent",
    [
        param(True, "aarch64", "BROWSER_PATH", "plotly_get_chrome", id="linux-arm64"),
        param(True, "x86_64", "plotly_get_chrome", "BROWSER_PATH", id="linux-x86"),
        param(False, "arm64", "plotly_get_chrome", "BROWSER_PATH", id="other-arm64"),
    ],
)  # fmt: skip
def test_dashboard_fallback_readme_uses_platform_guidance(
    linux: bool,
    machine: str,
    expected: str,
    absent: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(browser, "IS_LINUX", linux)
    monkeypatch.setattr(browser.platform, "machine", lambda: machine)
    result = browser.png_fallback_readme()
    assert "HTML" in result and expected in result and absent not in result


def test_png_real_plotly_missing_browser_chain_on_linux_arm64(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(browser, "IS_LINUX", True)
    monkeypatch.setattr(browser.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr("aiperf.plot.exporters.png.base.IS_WINDOWS_ARM", False)
    monkeypatch.setattr(Chromium, "find_browser", lambda *args, **kwargs: None)
    fig = Mock()
    fig.write_image.side_effect = lambda *args, **kwargs: pio.to_image(
        go.Figure(), format="png"
    )

    with pytest.raises(ConfigurationError) as raised:
        _Exporter(tmp_path)._export_figure(fig, tmp_path / "out.png")
    assert "BROWSER_PATH" in str(raised.value)
    assert "plotly_get_chrome" not in str(raised.value)


def test_png_invalid_browser_path_linux_arm64_suggests_correction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(browser, "IS_LINUX", True)
    monkeypatch.setattr(browser.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr("aiperf.plot.exporters.png.base.IS_WINDOWS_ARM", False)
    monkeypatch.setattr(chromium, "get_chrome_download_path", lambda **kwargs: None)
    monkeypatch.setattr(chromium, "get_old_chrome_download_path", lambda: None)
    monkeypatch.setenv("BROWSER_PATH", str(tmp_path / "missing"))
    fig = Mock()
    fig.write_image.side_effect = lambda *args, **kwargs: pio.to_image(
        go.Figure(), format="png"
    )

    with pytest.raises(ConfigurationError) as raised:
        _Exporter(tmp_path)._export_figure(fig, tmp_path / "out.png")
    assert "correct or unset" in str(raised.value)
    assert "BROWSER_PATH" in str(raised.value)


@pytest.mark.skipif(not browser.IS_LINUX, reason="uses a POSIX executable")
def test_png_failed_chrome_linux_arm64_preserves_cause_and_guides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(browser, "IS_LINUX", True)
    monkeypatch.setattr(browser.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr("aiperf.plot.exporters.png.base.IS_WINDOWS_ARM", False)
    fake_chrome = tmp_path / "chrome"
    fake_chrome.write_text("#!/bin/sh\nexit 126\n", encoding="utf-8")
    fake_chrome.chmod(0o700)
    monkeypatch.setattr(
        chromium, "get_chrome_download_path", lambda **kwargs: fake_chrome
    )
    system_chromium = tmp_path / "system-chromium"
    system_chromium.touch()
    monkeypatch.setenv("BROWSER_PATH", str(system_chromium))
    assert Chromium.find_browser(skip_local=False) == str(fake_chrome)
    fig = Mock()
    fig.write_image.side_effect = lambda *args, **kwargs: pio.to_image(
        go.Figure(), format="png"
    )

    with pytest.raises(ConfigurationError) as raised:
        _Exporter(tmp_path)._export_figure(fig, tmp_path / "out.png")
    assert "could not start" in str(raised.value)
    assert "Choreographer" in str(raised.value)
    assert "incompatible" in str(raised.value)
    assert raised.value.__cause__ is not None
    assert browser.is_browser_start_failure(raised.value.__cause__)
