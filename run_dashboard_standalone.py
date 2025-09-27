#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Standalone script to run the AIPerf Dashboard UI for testing mouse and keyboard interactions.

This script uses real AIPerf modules by properly loading all modules and registrations.
"""

import asyncio
import contextlib
import multiprocessing
import random
import sys
import time
import warnings
from contextlib import suppress
from datetime import datetime
from pathlib import Path

from aiperf.common.enums.timing_enums import CreditPhase
from aiperf.common.enums.worker_enums import WorkerStatus
from aiperf.common.models.credit_models import ProcessingStats
from aiperf.common.models.progress_models import (
    RecordsStats,
    RequestsStats,
    WorkerStats,
)
from aiperf.common.models.worker_models import WorkerTaskStats

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Suppress warnings for cleaner testing
warnings.filterwarnings("ignore")


# Mock only the LogConsumer to avoid complex ZMQ queue setup
class MockLogConsumer:
    """Mock log consumer that doesn't require ZMQ queue setup."""

    def __init__(self, log_queue=None, app=None, **kwargs):
        self.app = app

    async def initialize(self):
        pass

    async def start(self):
        pass

    async def stop(self):
        pass


def patch_log_consumer():
    """Patch the LogConsumer to avoid complex queue setup."""
    import aiperf.ui.dashboard.rich_log_viewer as rich_log_viewer_module

    rich_log_viewer_module.LogConsumer = MockLogConsumer


class StandaloneDashboardRunner:
    """Runner for the standalone dashboard using real AIPerf modules."""

    def __init__(self):
        self.app = None
        self._demo_task = None

    async def run(self):
        """Run the dashboard with real aiperf modules and demo data."""
        try:
            print("🚀 Starting AIPerf Dashboard UI (Standalone Mode)")
            print("🎬 EPIC DEMO MODE: Showcasing full profiling workflow!")
            print("")
            print("📋 Key bindings:")
            print(
                "   - 1: Overview   - 2: Progress   - 3: Metrics   - 4: Workers   - 5: Logs"
            )
            print("   - ESC: Restore view   - L: Toggle logs   - Ctrl+C: Quit")
            print("   - Double-click: Maximize/restore panels")
            print("")
            print("🎭 Demo Features:")
            print("   ✨ Live progress bars (Warmup → Profiling → Records)")
            print("   ⚡ Real-time worker status updates (8 demo workers)")
            print(
                "   📊 8 REAL AIPerf metrics: TTFT, Request Latency, Token Throughput, etc."
            )
            print(
                "   📈 Phase-based performance: 180ms→85ms latency, 15→45 req/sec throughput"
            )
            print("   📝 Contextual log messages for each phase")
            print("   🔄 Continuous cycle to show all features")
            print("=" * 70)

            # Import and load all modules to register ZMQ implementations
            from aiperf.module_loader import ensure_modules_loaded

            print("📦 Loading all AIPerf modules...")
            ensure_modules_loaded()
            print("✅ Modules loaded successfully!")

            # Patch only the LogConsumer to avoid queue complexity
            patch_log_consumer()

            # Import real aiperf modules
            from aiperf.common.config.endpoint_config import EndpointConfig
            from aiperf.common.config.service_config import ServiceConfig
            from aiperf.common.config.user_config import UserConfig
            from aiperf.controller.system_controller import SystemController
            from aiperf.ui.dashboard.aiperf_textual_app import AIPerfTextualApp

            # Create real config objects with minimal setup
            service_config = ServiceConfig()

            # UserConfig requires an endpoint, so provide a minimal mock endpoint
            user_config = UserConfig(
                endpoint=EndpointConfig(model_names=["gpt-3.5-turbo"])
            )
            controller = SystemController(
                user_config=user_config, service_config=service_config
            )

            # Create the textual app
            self.app = AIPerfTextualApp(
                service_config=service_config, controller=controller
            )

            # Start demo data generation
            self._demo_task = asyncio.create_task(self._generate_demo_data())

            # Run the app
            await self.app.run_async()

        except ImportError as e:
            print(f"❌ Import error: {e}")
            print(
                "🔧 Make sure you're in the aiperf project directory and virtual environment is activated"
            )
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error running dashboard: {e}")
            print(
                "🐛 This might be due to missing dependencies or configuration issues"
            )
            import traceback

            traceback.print_exc()
            sys.exit(1)
        finally:
            if self._demo_task:
                self._demo_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._demo_task

    async def _generate_demo_data(self):
        """Generate comprehensive demo data with separate async tasks for lightning-fast updates!"""
        await asyncio.sleep(2)  # Wait for UI to initialize

        if not self.app:
            return

        print("🎬 Starting LIGHTNING-FAST epic demo data generation!")
        print("⚡ 5% progress per second with separate async tasks!")
        print("🚀 Live updating: Progress + Metrics + Workers + Logs simultaneously!")
        print("📊 Phase durations:")
        print("   🔥 Warmup: 100 requests over 20 seconds (5% per second)")
        print("   ⚡ Profiling: 1000 requests over 20 seconds (5% per second)")
        print("   📊 Records: 1000 records over 20 seconds (5% per second)")
        print("=" * 60)

        # Shared state for all tasks
        self.demo_state = {
            "phase": "warmup",
            "time": 0,
            "start_time_ns": time.time_ns(),
            "total_requests": 1000,
            "total_warmup": 100,
            "worker_ids": [f"worker-{i:02d}" for i in range(1, 9)],
        }

        # Start all concurrent tasks for maximum responsiveness!
        tasks = [
            asyncio.create_task(self._progress_update_task()),
            asyncio.create_task(self._metrics_update_task()),
            asyncio.create_task(self._worker_status_task()),
            asyncio.create_task(self._log_messages_task()),
            asyncio.create_task(self._phase_manager_task()),
        ]

        try:
            # Run all tasks concurrently - MAXIMUM SPEED! 🚀
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            # Cancel all tasks gracefully
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _phase_manager_task(self):
        """Manage demo phases and timing - the master coordinator!"""
        import time

        try:
            while True:
                await asyncio.sleep(0.1)  # Check phase every 100ms for responsiveness
                self.demo_state["time"] += 0.1

                current_time = self.demo_state["time"]

                # Phase transitions every 20 seconds (5% per second = 20 seconds total)
                if self.demo_state["phase"] == "warmup" and current_time > 20:
                    self.demo_state["phase"] = "profiling"
                    print(f"🔥 → ⚡ PROFILING phase! (time={current_time:.1f}s)")
                elif self.demo_state["phase"] == "profiling" and current_time > 40:
                    self.demo_state["phase"] = "records"
                    print(f"⚡ → 📊 RECORDS phase! (time={current_time:.1f}s)")
                elif self.demo_state["phase"] == "records" and current_time > 60:
                    self.demo_state["phase"] = "completed"
                    print(f"📊 → ✅ COMPLETED phase! (time={current_time:.1f}s)")
                elif self.demo_state["phase"] == "completed" and current_time > 70:
                    # Reset for new cycle
                    self.demo_state["phase"] = "warmup"
                    self.demo_state["time"] = 0
                    self.demo_state["start_time_ns"] = time.time_ns()
                    print("🔄 RESTARTING cycle - back to WARMUP! 🔥")

        except asyncio.CancelledError:
            pass

    async def _progress_update_task(self):
        """Super fast progress bar updates - 5% per second!"""
        import time

        from aiperf.common.enums import CreditPhase
        from aiperf.common.models import RecordsStats, RequestsStats

        try:
            while True:
                await asyncio.sleep(
                    0.2
                )  # Update progress 5 times per second for smooth bars!

                phase = self.demo_state["phase"]
                current_time = self.demo_state["time"]

                if phase == "warmup":
                    # 5% per second over 20 seconds
                    progress_pct = min(current_time / 20.0, 1.0)
                    completed = int(progress_pct * self.demo_state["total_warmup"])
                    sent = min(completed + 5, self.demo_state["total_warmup"])

                    warmup_stats = RequestsStats(
                        type=CreditPhase.WARMUP,
                        start_ns=self.demo_state["start_time_ns"],
                        total_expected_requests=self.demo_state["total_warmup"],
                        sent=sent,
                        completed=completed,
                        errors=max(0, int(completed * 0.02)),  # 2% error rate
                        sent_end_ns=self.demo_state["start_time_ns"]
                        + int(current_time * 1_000_000_000)
                        if completed >= self.demo_state["total_warmup"]
                        else None,
                        end_ns=self.demo_state["start_time_ns"]
                        + int(current_time * 1_000_000_000)
                        if completed >= self.demo_state["total_warmup"]
                        else None,
                        per_second=round(completed / max(current_time, 0.1), 1),
                        eta=round(
                            (self.demo_state["total_warmup"] - completed)
                            / max(completed / max(current_time, 0.1), 0.1),
                            1,
                        )
                        if completed < self.demo_state["total_warmup"]
                        else None,
                        last_update_ns=time.time_ns(),
                    )
                    await self.app.on_warmup_progress(warmup_stats)

                elif phase == "profiling":
                    # 5% per second over 20 seconds (time 20-40)
                    profiling_time = current_time - 20
                    progress_pct = min(profiling_time / 20.0, 1.0)
                    completed = int(progress_pct * self.demo_state["total_requests"])
                    sent = min(completed + 25, self.demo_state["total_requests"])

                    profiling_stats = RequestsStats(
                        type=CreditPhase.PROFILING,
                        start_ns=self.demo_state["start_time_ns"]
                        + (20 * 1_000_000_000),
                        total_expected_requests=self.demo_state["total_requests"],
                        sent=sent,
                        completed=completed,
                        errors=max(0, int(completed * 0.003)),  # 0.3% error rate
                        sent_end_ns=self.demo_state["start_time_ns"]
                        + int(current_time * 1_000_000_000)
                        if completed >= self.demo_state["total_requests"]
                        else None,
                        end_ns=self.demo_state["start_time_ns"]
                        + int(current_time * 1_000_000_000)
                        if completed >= self.demo_state["total_requests"]
                        else None,
                        per_second=round(completed / max(profiling_time, 0.1), 1),
                        eta=round(
                            (self.demo_state["total_requests"] - completed)
                            / max(completed / max(profiling_time, 0.1), 0.1),
                            1,
                        )
                        if completed < self.demo_state["total_requests"]
                        else None,
                        last_update_ns=time.time_ns(),
                    )
                    await self.app.on_profiling_progress(profiling_stats)

                elif phase == "records":
                    # 5% per second over 20 seconds (time 40-60)
                    records_time = current_time - 40
                    progress_pct = min(records_time / 20.0, 1.0)
                    records_processed = int(
                        progress_pct * self.demo_state["total_requests"]
                    )

                    records_stats = RecordsStats(
                        start_ns=self.demo_state["start_time_ns"]
                        + (40 * 1_000_000_000),
                        total_expected_requests=self.demo_state["total_requests"],
                        processed=records_processed,
                        errors=max(
                            0, int(records_processed * 0.001)
                        ),  # 0.1% error rate
                        total_records=records_processed
                        + max(0, int(records_processed * 0.001)),
                        per_second=round(records_processed / max(records_time, 0.1), 1),
                        eta=round(
                            (self.demo_state["total_requests"] - records_processed)
                            / max(records_processed / max(records_time, 0.1), 0.1),
                            1,
                        )
                        if records_processed < self.demo_state["total_requests"]
                        else None,
                        last_update_ns=time.time_ns(),
                    )
                    await self.app.on_records_progress(records_stats)

        except asyncio.CancelledError:
            pass

    async def _metrics_update_task(self):
        """Lightning-fast metrics updates - multiple times per second!"""

        time.time_ns()
        self.demo_state["total_warmup"]
        self.demo_state["total_requests"]
        self.demo_state["worker_ids"]

        try:
            while True:
                await asyncio.sleep(
                    0.3
                )  # Update metrics ~3 times per second for fluid changes

                phase = self.demo_state["phase"]
                current_time = self.demo_state["time"]

                # if phase == "warmup":
                #     # 5% per second over 20 seconds (time 0-20)
                #     warmup_time = current_time
                #     progress_pct = min(warmup_time / 20.0, 1.0)
                #     completed = int(progress_pct * self.demo_state['total_warmup'])
                #     sent = min(completed + 5, self.demo_state['total_warmup'])

                #     warmup_stats = RequestsStats(
                #         type=CreditPhase.WARMUP,
                #         start_ns=self.demo_state['start_time_ns'],
                #         total_expected_requests=self.demo_state['total_warmup'],
                #         sent=sent,
                #         completed=completed,
                #         errors=max(0, int(completed * 0.02)),  # 2% error rate
                #         sent_end_ns=self.demo_state['start_time_ns'] + int(current_time * 1_000_000_000) if completed >= self.demo_state['total_warmup'] else None,
                #         end_ns=self.demo_state['start_time_ns'] + int(current_time * 1_000_000_000) if completed >= self.demo_state['total_warmup'] else None,
                #         per_second=round(completed / max(current_time, 0.1), 1),
                #         eta=round((self.demo_state['total_warmup'] - completed) / max(completed / max(current_time, 0.1), 0.1), 1) if completed < self.demo_state['total_warmup'] else None,
                #         last_update_ns=time.time_ns()
                #     )
                #     await self.app.on_warmup_progress(warmup_stats)

                # elif phase == "profiling":
                #     # 5% per second over 20 seconds (time 20-40)
                #     profiling_time = current_time - 20
                #     progress_pct = min(profiling_time / 20.0, 1.0)
                #     completed = int(progress_pct * self.demo_state['total_requests'])
                #     sent = min(completed + 25, self.demo_state['total_requests'])

                #     profiling_stats = RequestsStats(
                #         type=CreditPhase.PROFILING,
                #         start_ns=self.demo_state['start_time_ns'] + (20 * 1_000_000_000),
                #         total_expected_requests=self.demo_state['total_requests'],
                #         sent=sent,
                #         completed=completed,
                #         errors=max(0, int(completed * 0.003)),  # 0.3% error rate
                #         sent_end_ns=self.demo_state['start_time_ns'] + int(current_time * 1_000_000_000) if completed >= self.demo_state['total_requests'] else None,
                #         end_ns=self.demo_state['start_time_ns'] + int(current_time * 1_000_000_000) if completed >= self.demo_state['total_requests'] else None,
                #         per_second=round(completed / max(profiling_time, 0.1), 1),
                #         eta=round((self.demo_state['total_requests'] - completed) / max(completed / max(profiling_time, 0.1), 0.1), 1) if completed < self.demo_state['total_requests'] else None,
                #         last_update_ns=time.time_ns()
                #     )
                #     await self.app.on_profiling_progress(profiling_stats)

                # elif phase == "records" and current_time > 80:
                #     records_time = current_time - 40
                #     progress_pct = min(records_time / 20.0, 1.0)
                #     records_processed = int(progress_pct * self.demo_state['total_requests'])

                #     records_stats = RecordsStats(
                #         start_ns=self.demo_state['start_time_ns'] + (40 * 1_000_000_000),
                #         total_expected_requests=self.demo_state['total_requests'],
                #         processed=records_processed,
                #         errors=max(0, int(records_processed * 0.001)),  # 0.1% error rate
                #         total_records=records_processed + max(0, int(records_processed * 0.001)),
                #         per_second=round(records_processed / max(records_time, 0.1), 1),
                #         eta=round((self.demo_state['total_requests'] - records_processed) / max(records_processed / max(records_time, 0.1), 0.1), 1) if records_processed < self.demo_state['total_requests'] else None,
                #         last_update_ns=time.time_ns()
                #     )
                #     await self.app.on_records_progress(records_stats)

                # elif phase == "completed" and current_time > 90:
                #     records_stats = RecordsStats(
                #         start_ns=self.demo_state['start_time_ns'] + (60 * 1_000_000_000),
                #         total_expected_requests=self.demo_state['total_requests'],
                #         processed=records_processed,
                #         errors=max(0, int(records_processed * 0.001)),  # 0.1% error rate
                #         total_records=records_processed + max(0, int(records_processed * 0.001)),
                #         per_second=round(records_processed / max(records_time, 0.1), 1),
                #         eta=round((self.demo_state['total_requests'] - records_processed) / max(records_processed / max(records_time, 0.1), 0.1), 1) if records_processed < self.demo_state['total_requests'] else None,
                #         last_update_ns=time.time_ns()
                #     )
                #     await self.app.on_records_progress(records_stats)

                # elif phase == "completed":
                #     records_stats = RecordsStats(
                #         start_ns=self.demo_state['start_time_ns'] + (40 * 1_000_000_000),
                #         total_expected_requests=self.demo_state['total_requests'],
                #         processed=records_processed,
                #         errors=max(0, int(records_processed * 0.001)),  # 0.1% error rate
                #         total_records=records_processed + max(0, int(records_processed * 0.001)),
                #         per_second=round(records_processed / max(records_time, 0.1), 1),
                #         eta=round((self.demo_state['total_requests'] - records_processed) / max(records_processed / max(records_time, 0.1), 0.1), 1) if records_processed < self.demo_state['total_requests'] else None,
                #         last_update_ns=time.time_ns()
                #     )
                #     await self.app.on_records_progress(records_stats)

                # elif phase == "completed":
                #     records_stats = RecordsStats(
                #         start_ns=self.demo_state['start_time_ns'] + (60 * 1_000_000_000),
                #         total_expected_requests=self.demo_state['total_requests'],
                #         processed=self.demo_state['total_requests'],
                #         errors=max(0, int(self.demo_state['total_requests'] * 0.001)),  # 0.1% error rate
                #         total_records=self.demo_state['total_requests'] + max(0, int(self.demo_state['total_requests'] * 0.001)),
                #         per_second=round(self.demo_state['total_requests'] / max(records_time, 0.1), 1),
                #         eta=round((self.demo_state['total_requests'] - self.demo_state['total_requests']) / max(self.demo_state['total_requests'] / max(records_time, 0.1), 0.1), 1) if self.demo_state['total_requests'] < self.demo_state['total_requests'] else None,
                #         last_update_ns=time.time_ns(),
                #     )
                #     await self.app.on_records_progress(records_stats)

                # elif phase == "completed":
                #     # Completed phase (time 70-90)
                #     pass

                # Debug output every 5 seconds
                if current_time % 5 == 0:
                    if phase == "warmup":
                        completed = min(
                            int(
                                (current_time / 12.0) * self.demo_state["total_warmup"]
                            ),
                            self.demo_state["total_warmup"],
                        )
                        print(
                            f"🔥 Warmup Progress: {completed}/{self.demo_state['total_warmup']} ({current_time}s)"
                        )
                    elif phase == "profiling":
                        profiling_time = current_time - 20
                if current_time % 5 == 0:
                    if phase == "warmup":
                        completed = min(
                            int(
                                (current_time / 12.0) * self.demo_state["total_warmup"]
                            ),
                            self.demo_state["total_warmup"],
                        )
                        print(
                            f"🔥 Warmup Progress: {completed}/{self.demo_state['total_warmup']} ({current_time}s)"
                        )
                    elif phase == "profiling":
                        profiling_time = current_time - 15
                        completed = min(
                            int(
                                (profiling_time / 40.0)
                                * self.demo_state["total_requests"]
                            ),
                            self.demo_state["total_requests"],
                        )
                        print(
                            f"⚡ Profiling Progress: {completed}/{self.demo_state['total_requests']} ({profiling_time}s into profiling)"
                        )
                    elif phase == "records":
                        records_time = current_time - 60
                        processed = min(
                            int(
                                (records_time / 15.0)
                                * self.demo_state["total_requests"]
                            ),
                            self.demo_state["total_requests"],
                        )
                        print(
                            f"📊 Records Progress: {processed}/{self.demo_state['total_requests']} ({records_time}s into records)"
                        )

                # === PROGRESS UPDATES ===
                if phase == "warmup":
                    # 5% per second: 20 seconds to complete (5% * 20 = 100%)
                    completed = min(
                        int((current_time / 20.0) * self.demo_state["total_warmup"]),
                        self.demo_state["total_warmup"],
                    )
                    sent = min(completed + 5, self.demo_state["total_warmup"])
                    warmup_stats = RequestsStats(
                        type=CreditPhase.WARMUP,
                        start_ns=self.demo_state["start_time_ns"],
                        total_expected_requests=self.demo_state["total_warmup"],
                        sent=sent,
                        completed=completed,
                        errors=random.randint(0, 2),
                        sent_end_ns=self.demo_state["start_time_ns"]
                        + int(current_time * 1_000_000_000)
                        if sent >= self.demo_state["total_warmup"]
                        else None,
                        end_ns=self.demo_state["start_time_ns"]
                        + int(current_time * 1_000_000_000)
                        if completed >= self.demo_state["total_warmup"]
                        else None,
                        per_second=round(completed / max(current_time, 0.1), 1),
                        eta=round(
                            (self.demo_state["total_warmup"] - completed)
                            / max(completed / max(current_time, 0.1), 0.1),
                            1,
                        )
                        if completed < self.demo_state["total_warmup"]
                        else None,
                        last_update_ns=time.time_ns(),
                    )
                    await self.app.on_warmup_progress(warmup_stats)

                elif phase == "profiling":
                    # 5% per second: 20 seconds to complete 1000 requests (time 20-40)
                    profiling_time = current_time - 20
                    completed = min(
                        int(
                            (profiling_time / 20.0) * self.demo_state["total_requests"]
                        ),
                        self.demo_state["total_requests"],
                    )
                    sent = min(completed + 25, self.demo_state["total_requests"])
                    profiling_stats = RequestsStats(
                        type=CreditPhase.PROFILING,
                        start_ns=self.demo_state["start_time_ns"]
                        + (20 * 1_000_000_000),
                        total_expected_requests=self.demo_state["total_requests"],
                        sent=sent,
                        completed=completed,
                        errors=random.randint(0, 5),
                        sent_end_ns=self.demo_state["start_time_ns"]
                        + (int(current_time * 1_000_000_000))
                        if sent >= self.demo_state["total_requests"]
                        else None,
                        end_ns=self.demo_state["start_time_ns"]
                        + (int(current_time * 1_000_000_000))
                        if completed >= self.demo_state["total_requests"]
                        else None,
                        per_second=round(completed / max(profiling_time, 0.1), 1),
                        eta=round(
                            (self.demo_state["total_requests"] - completed)
                            / max(completed / max(profiling_time, 0.1), 0.1),
                            1,
                        )
                        if completed < self.demo_state["total_requests"]
                        else None,
                        last_update_ns=time.time_ns(),
                    )
                    await self.app.on_profiling_progress(profiling_stats)

                elif phase == "records":
                    # 5% per second: 20 seconds to complete 1000 records (time 40-60)
                    records_time = current_time - 40
                    records_processed = min(
                        int((records_time / 20.0) * self.demo_state["total_requests"]),
                        self.demo_state["total_requests"],
                    )
                    records_stats = RecordsStats(
                        start_ns=self.demo_state["start_time_ns"]
                        + (40 * 1_000_000_000),
                        total_expected_requests=self.demo_state["total_requests"],
                        processed=records_processed,
                        errors=random.randint(0, 3),
                        total_records=records_processed + random.randint(0, 3),
                        per_second=round(records_processed / max(records_time, 0.1), 1),
                        eta=round(
                            (self.demo_state["total_requests"] - records_processed)
                            / max(records_processed / max(records_time, 0.1), 0.1),
                            1,
                        )
                        if records_processed < self.demo_state["total_requests"]
                        else None,
                        last_update_ns=time.time_ns(),
                    )
                    await self.app.on_records_progress(records_stats)

        except asyncio.CancelledError:
            pass

    async def _metrics_update_task(self):
        """Lightning-fast metrics updates - multiple times per second!"""

        from aiperf.common.models import MetricResult

        try:
            while True:
                await asyncio.sleep(
                    0.3
                )  # Update metrics ~3 times per second for fluid changes

                phase = self.demo_state["phase"]
                current_time = self.demo_state["time"]

                # Define phase-based performance characteristics
                perf_characteristics = {
                    "warmup": {
                        "ttft_ms": 120.0,
                        "request_latency_ms": 180.0,
                        "request_throughput": 15.0,
                        "output_token_throughput": 250.0,
                        "output_token_throughput_per_user": 45.0,
                        "output_sequence_length": 85,
                        "error_count": 2,
                    },
                    "profiling": {
                        "ttft_ms": 45.0,
                        "request_latency_ms": 85.0,
                        "request_throughput": 45.0,
                        "output_token_throughput": 1250.0,
                        "output_token_throughput_per_user": 120.0,
                        "output_sequence_length": 150,
                        "error_count": 3,
                    },
                    "records": {
                        "ttft_ms": 35.0,
                        "request_latency_ms": 65.0,
                        "request_throughput": 25.0,
                        "output_token_throughput": 800.0,
                        "output_token_throughput_per_user": 95.0,
                        "output_sequence_length": 140,
                        "error_count": 1,
                    },
                    "completed": {
                        "ttft_ms": 25.0,
                        "request_latency_ms": 45.0,
                        "request_throughput": 5.0,
                        "output_token_throughput": 100.0,
                        "output_token_throughput_per_user": 20.0,
                        "output_sequence_length": 60,
                        "error_count": 0,
                    },
                }

                current_perf = perf_characteristics[phase]

                # Create realistic metrics using actual AIPerf metric tags with required fields
                demo_metrics = [
                    # TTFT (display_order=100) - VALUES IN NANOSECONDS!
                    MetricResult(
                        tag="ttft",
                        unit="ns",
                        header="Time to First Token",
                        avg=round(
                            current_perf["ttft_ms"]
                            * 1_000_000
                            * random.uniform(0.85, 1.15)
                        ),
                        min=round(
                            current_perf["ttft_ms"]
                            * 1_000_000
                            * 0.4
                            * random.uniform(0.9, 1.1)
                        ),
                        max=round(
                            current_perf["ttft_ms"]
                            * 1_000_000
                            * 2.2
                            * random.uniform(0.9, 1.3)
                        ),
                        p99=round(
                            current_perf["ttft_ms"]
                            * 1_000_000
                            * 1.8
                            * random.uniform(0.9, 1.1)
                        ),
                        p90=round(
                            current_perf["ttft_ms"]
                            * 1_000_000
                            * 1.4
                            * random.uniform(0.9, 1.1)
                        ),
                        p50=round(
                            current_perf["ttft_ms"]
                            * 1_000_000
                            * 0.9
                            * random.uniform(0.95, 1.05)
                        ),
                        std=round(
                            current_perf["ttft_ms"]
                            * 1_000_000
                            * 0.25
                            * random.uniform(0.8, 1.2)
                        ),
                    ),
                    # Request Latency (display_order=300) - VALUES IN NANOSECONDS!
                    MetricResult(
                        tag="request_latency",
                        unit="ns",
                        header="Request Latency",
                        avg=round(
                            current_perf["request_latency_ms"]
                            * 1_000_000
                            * random.uniform(0.8, 1.2)
                        ),
                        min=round(
                            current_perf["request_latency_ms"]
                            * 1_000_000
                            * 0.5
                            * random.uniform(0.9, 1.1)
                        ),
                        max=round(
                            current_perf["request_latency_ms"]
                            * 1_000_000
                            * 2.1
                            * random.uniform(0.9, 1.3)
                        ),
                        p99=round(
                            current_perf["request_latency_ms"]
                            * 1_000_000
                            * 1.7
                            * random.uniform(0.9, 1.1)
                        ),
                        p90=round(
                            current_perf["request_latency_ms"]
                            * 1_000_000
                            * 1.3
                            * random.uniform(0.9, 1.1)
                        ),
                        p50=round(
                            current_perf["request_latency_ms"]
                            * 1_000_000
                            * 0.85
                            * random.uniform(0.95, 1.05)
                        ),
                        std=round(
                            current_perf["request_latency_ms"]
                            * 1_000_000
                            * 0.3
                            * random.uniform(0.8, 1.2)
                        ),
                    ),
                    # Output Token Throughput Per User (display_order=500)
                    MetricResult(
                        tag="output_token_throughput_per_user",
                        unit="tokens/s/user",
                        header="Output Token Throughput Per User",
                        avg=round(
                            current_perf["output_token_throughput_per_user"]
                            * random.uniform(0.8, 1.2),
                            1,
                        ),
                        min=round(
                            current_perf["output_token_throughput_per_user"]
                            * 0.6
                            * random.uniform(0.9, 1.1),
                            1,
                        ),
                        max=round(
                            current_perf["output_token_throughput_per_user"]
                            * 1.4
                            * random.uniform(0.9, 1.2),
                            1,
                        ),
                        p99=round(
                            current_perf["output_token_throughput_per_user"]
                            * 1.3
                            * random.uniform(0.9, 1.1),
                            1,
                        ),
                        p90=round(
                            current_perf["output_token_throughput_per_user"]
                            * 1.1
                            * random.uniform(0.95, 1.05),
                            1,
                        ),
                        p50=round(
                            current_perf["output_token_throughput_per_user"]
                            * 0.95
                            * random.uniform(0.95, 1.05),
                            1,
                        ),
                        std=round(
                            current_perf["output_token_throughput_per_user"]
                            * 0.2
                            * random.uniform(0.8, 1.2),
                            1,
                        ),
                    ),
                    # Output Sequence Length (display_order=600)
                    MetricResult(
                        tag="output_sequence_length",
                        unit="tokens",
                        header="Output Sequence Length",
                        avg=round(
                            current_perf["output_sequence_length"]
                            * random.uniform(0.9, 1.1)
                        ),
                        min=round(
                            current_perf["output_sequence_length"]
                            * 0.3
                            * random.uniform(0.8, 1.2)
                        ),
                        max=round(
                            current_perf["output_sequence_length"]
                            * 2.5
                            * random.uniform(0.9, 1.3)
                        ),
                        p99=round(
                            current_perf["output_sequence_length"]
                            * 2.0
                            * random.uniform(0.9, 1.1)
                        ),
                        p90=round(
                            current_perf["output_sequence_length"]
                            * 1.5
                            * random.uniform(0.95, 1.05)
                        ),
                        p50=round(
                            current_perf["output_sequence_length"]
                            * 0.85
                            * random.uniform(0.95, 1.05)
                        ),
                        std=round(
                            current_perf["output_sequence_length"]
                            * 0.4
                            * random.uniform(0.8, 1.2)
                        ),
                    ),
                    # Output Token Throughput (display_order=800)
                    MetricResult(
                        tag="output_token_throughput",
                        unit="tokens/s",
                        header="Output Token Throughput",
                        avg=round(
                            current_perf["output_token_throughput"]
                            * random.uniform(0.8, 1.2),
                            1,
                        ),
                        min=round(
                            current_perf["output_token_throughput"]
                            * 0.6
                            * random.uniform(0.9, 1.1),
                            1,
                        ),
                        max=round(
                            current_perf["output_token_throughput"]
                            * 1.5
                            * random.uniform(0.9, 1.3),
                            1,
                        ),
                        p99=round(
                            current_perf["output_token_throughput"]
                            * 1.3
                            * random.uniform(0.9, 1.1),
                            1,
                        ),
                        p90=round(
                            current_perf["output_token_throughput"]
                            * 1.1
                            * random.uniform(0.95, 1.05),
                            1,
                        ),
                        p50=round(
                            current_perf["output_token_throughput"]
                            * 0.9
                            * random.uniform(0.95, 1.05),
                            1,
                        ),
                        std=round(
                            current_perf["output_token_throughput"]
                            * 0.25
                            * random.uniform(0.8, 1.2),
                            1,
                        ),
                    ),
                    # Request Throughput (display_order=900)
                    MetricResult(
                        tag="request_throughput",
                        unit="req/s",
                        header="Request Throughput",
                        avg=round(
                            current_perf["request_throughput"]
                            * random.uniform(0.8, 1.2),
                            2,
                        ),
                        min=round(
                            current_perf["request_throughput"]
                            * 0.7
                            * random.uniform(0.9, 1.1),
                            2,
                        ),
                        max=round(
                            current_perf["request_throughput"]
                            * 1.4
                            * random.uniform(0.9, 1.2),
                            2,
                        ),
                        p99=round(
                            current_perf["request_throughput"]
                            * 1.2
                            * random.uniform(0.9, 1.1),
                            2,
                        ),
                        p90=round(
                            current_perf["request_throughput"]
                            * 1.05
                            * random.uniform(0.95, 1.05),
                            2,
                        ),
                        p50=round(
                            current_perf["request_throughput"]
                            * 0.95
                            * random.uniform(0.95, 1.05),
                            2,
                        ),
                        std=round(
                            current_perf["request_throughput"]
                            * 0.15
                            * random.uniform(0.8, 1.2),
                            2,
                        ),
                    ),
                    # Request Count (display_order=1000) - shows current completed count
                    MetricResult(
                        tag="request_count",
                        unit="requests",
                        header="Request Count",
                        avg=self._get_current_request_count(phase, current_time),
                        min=self._get_current_request_count(phase, current_time),
                        max=self._get_current_request_count(phase, current_time),
                        p99=self._get_current_request_count(phase, current_time),
                        p90=self._get_current_request_count(phase, current_time),
                        p50=self._get_current_request_count(phase, current_time),
                        std=0,
                    ),
                    # Error Request Count (ERROR_ONLY flag)
                    MetricResult(
                        tag="error_request_count",
                        unit="requests",
                        header="Error Request Count",
                        avg=current_perf["error_count"] + random.randint(-1, 2)
                        if current_perf["error_count"] > 0
                        else random.randint(0, 1),
                        min=0,
                        max=current_perf["error_count"] + random.randint(0, 3),
                        p99=current_perf["error_count"] + random.randint(0, 2),
                        p90=max(0, current_perf["error_count"] + random.randint(-1, 1)),
                        p50=max(0, current_perf["error_count"]),
                        std=random.uniform(0, 1.5),
                    ),
                ]

                # Send metrics to UI
                await self.app.on_realtime_metrics(demo_metrics)

        except asyncio.CancelledError:
            pass

    def _get_current_request_count(self, phase: str, current_time: float) -> int:
        """Get the current request count based on phase and time."""
        if phase == "warmup":
            return min(
                int((current_time / 20.0) * self.demo_state["total_warmup"]),
                self.demo_state["total_warmup"],
            )
        elif phase == "profiling":
            profiling_time = current_time - 20
            return min(
                int((profiling_time / 20.0) * self.demo_state["total_requests"]),
                self.demo_state["total_requests"],
            )
        elif phase in ["records", "completed"]:
            return self.demo_state["total_requests"]
        return 0

    async def _worker_status_task(self):
        """Super responsive worker status updates!"""

        try:
            while True:
                await asyncio.sleep(0.4)  # Update workers ~2.5 times per second

                phase = self.demo_state["phase"]
                current_time = self.demo_state["time"]
                worker_ids = self.demo_state["worker_ids"]

                # Update individual workers with realistic activity patterns
                for i, worker_id in enumerate(worker_ids):
                    if phase == "warmup":
                        status = (
                            WorkerStatus.IDLE
                            if i > (current_time / 3)
                            else WorkerStatus.HEALTHY
                        )
                        completed_tasks = max(
                            0, int((current_time - i * 2.5) * random.uniform(2, 5))
                        )
                        total_tasks = completed_tasks + random.randint(0, 3)
                    elif phase == "profiling":
                        # Mix of healthy and high load during intense profiling
                        if random.random() > 0.15:
                            status = (
                                WorkerStatus.HEALTHY
                                if random.random() > 0.3
                                else WorkerStatus.HIGH_LOAD
                            )
                        else:
                            status = WorkerStatus.IDLE
                        profiling_time = current_time - 20
                        completed_tasks = max(
                            0, int(profiling_time * random.uniform(15, 25))
                        )
                        total_tasks = completed_tasks + random.randint(0, 8)
                    elif phase == "records":
                        status = WorkerStatus.HEALTHY if i < 5 else WorkerStatus.IDLE
                        records_time = current_time - 40
                        completed_tasks = max(
                            0, int(records_time * random.uniform(10, 20))
                        )
                        total_tasks = completed_tasks + random.randint(0, 5)
                    else:  # completed
                        status = WorkerStatus.IDLE
                        completed_tasks = self.demo_state[
                            "total_requests"
                        ] + random.randint(-20, 20)
                        total_tasks = completed_tasks + random.randint(0, 3)

                    # Inject realistic errors
                    task_errors = random.randint(0, 3) if random.random() > 0.85 else 0
                    processing_errors = (
                        random.randint(0, 2) if random.random() > 0.9 else 0
                    )

                    worker_stats = WorkerStats(
                        worker_id=worker_id,
                        task_stats=WorkerTaskStats(
                            total=max(0, total_tasks),
                            completed=max(0, completed_tasks),
                            failed=task_errors,
                        ),
                        processing_stats=ProcessingStats(
                            processed=max(0, completed_tasks), errors=processing_errors
                        ),
                        status=status,
                        last_update_ns=time.time_ns(),
                    )
                    await self.app.on_worker_update(worker_id, worker_stats)

                # Worker status summary for overview
                worker_status_summary = {
                    worker_id: (
                        WorkerStatus.HEALTHY
                        if random.random() > 0.25
                        else WorkerStatus.IDLE
                    )
                    if phase in ["profiling", "records"]
                    else WorkerStatus.IDLE
                    for worker_id in worker_ids
                }
                await self.app.on_worker_status_summary(worker_status_summary)

        except asyncio.CancelledError:
            pass

    async def _log_messages_task(self):
        """Dynamic contextual log messages!"""

        try:
            while True:
                await asyncio.sleep(0.8)  # Generate logs every ~1.25 times per second

                phase = self.demo_state["phase"]
                worker_ids = self.demo_state["worker_ids"]

                if not hasattr(self.app, "log_viewer") or not self.app.log_viewer:
                    continue

                timestamp = datetime.now().strftime("%H:%M:%S")

                phase_messages = {
                    "warmup": [
                        "🔥 Initializing warmup phase",
                        f"Worker {random.choice(worker_ids)} spinning up",
                        "Connection pool warming...",
                        "Model cache loading",
                        "Calibrating request timing",
                        "Pre-flight validation complete",
                    ],
                    "profiling": [
                        f"🚀 Processing batch #{random.randint(100, 999)}",
                        f"Worker {random.choice(worker_ids)} completed {random.randint(20, 80)} requests",
                        "Peak throughput achieved!",
                        "Load balancer optimizing routes",
                        f"Response time: {random.randint(40, 120)}ms avg",
                        "Credit pool management active",
                    ],
                    "records": [
                        "📊 Aggregating performance records",
                        "Computing percentile statistics",
                        f"Processing {random.randint(50, 200)} records/sec",
                        "Validating metric calculations",
                        "Generating summary statistics",
                        "Record integrity checks passed",
                    ],
                    "completed": [
                        "✅ Profiling session completed successfully",
                        "Final metrics computed",
                        "Cleaning up worker resources",
                        "Session data archived",
                        "System ready for next run",
                    ],
                }

                log_level = random.choices(
                    ["INFO", "DEBUG", "SUCCESS", "WARNING"], weights=[55, 25, 15, 5]
                )[0]

                message = random.choice(phase_messages[phase])
                log_msg = f"[{timestamp}] {log_level} {message}"

                with contextlib.suppress(Exception):
                    self.app.log_viewer.write(log_msg)

        except asyncio.CancelledError:
            pass


def main():
    """Main entry point for standalone dashboard."""

    # Set multiprocessing start method (required for some systems)
    if sys.platform.startswith("darwin"):  # macOS
        multiprocessing.set_start_method("spawn", force=True)

    runner = StandaloneDashboardRunner()

    try:
        asyncio.run(runner.run())
    except KeyboardInterrupt:
        print("\n👋 Dashboard UI stopped by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
