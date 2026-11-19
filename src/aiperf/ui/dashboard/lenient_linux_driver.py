# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from textual.drivers.linux_driver import LinuxDriver

from aiperf.common.aiperf_logger import AIPerfLogger

_logger = AIPerfLogger(__name__)


class LenientLinuxDriver(LinuxDriver):
    """LinuxDriver that survives non-UTF-8 bytes on stdin.

    Textual decodes stdin strictly: an invalid byte makes it panic the app and
    stop reading input. Some terminals and container runtimes (e.g. podman -it)
    send such bytes, so input reading restarts instead. Valid keys in the same
    read as the invalid byte are dropped; reading resumes with the next read.
    """

    _undecodable_input_warned: bool = False

    def run_input_thread(self) -> None:
        while True:
            try:
                super().run_input_thread()
                return
            except UnicodeDecodeError:
                if not self._undecodable_input_warned:
                    self._undecodable_input_warned = True
                    _logger.warning("Dropped undecodable (non-UTF-8) terminal input")
                if self.exit_event.is_set():
                    return
