# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from aiperf.common.enums import CreditPhase
from aiperf.common.messages import (
    Message,
)
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.progress.progress_models import BenchmarkSuiteProgress, ProfileRunProgress


class ProgressTracker(AIPerfLoggerMixin):
    """A progress tracker that tracks the progress of the entire benchmark suite."""

    def __init__(self):
        super().__init__()
        self.suite: BenchmarkSuiteProgress | None = None

    def configure(
        self, suite: BenchmarkSuiteProgress, current_profile_run: ProfileRunProgress
    ):
        """Configure the progress tracker with a benchmark suite."""
        self.suite = suite
        self.suite.current_profile_run = current_profile_run

    @property
    def current_profile_run(self) -> ProfileRunProgress | None:
        if self.suite is None:
            return None
        return self.suite.current_profile_run

    @property
    def active_credit_phase(self) -> CreditPhase | None:
        if self.current_profile_run is None:
            return None
        return self.current_profile_run.active_phase

    @active_credit_phase.setter
    def active_credit_phase(self, value: CreditPhase):
        if self.current_profile_run is None:
            return
        self.current_profile_run.active_phase = value

    def on_message(self, message: Message):
        """Update the progress from a message."""
        if self.current_profile_run is None:
            self.debug(
                lambda: f"Received {message.message_type} message before profile run is started"
            )
            return
        self.current_profile_run.on_message(message)
