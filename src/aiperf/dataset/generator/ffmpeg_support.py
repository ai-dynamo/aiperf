# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FFmpeg availability checks and platform-specific install instructions."""

from __future__ import annotations

import platform
import shutil


def check_ffmpeg_availability() -> bool:
    """Check if FFmpeg binary is available in the system."""
    return shutil.which("ffmpeg") is not None


def _linux_install_instructions() -> str:
    try:
        with open("/etc/os-release") as f:
            os_info = f.read().lower()
    except (FileNotFoundError, PermissionError, OSError):
        os_info = ""

    if "ubuntu" in os_info or "debian" in os_info:
        return "sudo apt update && sudo apt install ffmpeg"
    if "fedora" in os_info or "rhel" in os_info or "centos" in os_info:
        return "sudo dnf install ffmpeg  # or: sudo yum install ffmpeg"
    if "arch" in os_info:
        return "sudo pacman -S ffmpeg"
    return "sudo apt install ffmpeg  # (Ubuntu/Debian) or use your distribution's package manager"


def _macos_install_instructions() -> str:
    if shutil.which("brew"):
        return "brew install ffmpeg"
    if shutil.which("port"):
        return "sudo port install ffmpeg"
    return "brew install ffmpeg  # (install Homebrew first: https://brew.sh)"


def _windows_install_instructions() -> str:
    if shutil.which("choco"):
        return "choco install ffmpeg"
    if shutil.which("winget"):
        return "winget install ffmpeg"
    return (
        "Download from https://ffmpeg.org/download.html or use 'choco install ffmpeg'"
    )


def get_ffmpeg_install_instructions() -> str:
    """Get platform-specific FFmpeg installation instructions."""
    system = platform.system().lower()
    if system == "linux":
        return _linux_install_instructions()
    if system == "darwin":
        return _macos_install_instructions()
    if system == "windows":
        return _windows_install_instructions()
    return "Install FFmpeg using your system's package manager or download from https://ffmpeg.org"
