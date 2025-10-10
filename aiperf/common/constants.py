# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

NANOS_PER_SECOND = 1_000_000_000
NANOS_PER_MILLIS = 1_000_000
MILLIS_PER_SECOND = 1000
BYTES_PER_MIB = 1024 * 1024

STAT_KEYS = [
    "avg",
    "min",
    "max",
    "p1",
    "p5",
    "p10",
    "p25",
    "p50",
    "p75",
    "p90",
    "p95",
    "p99",
    "std",
]


GRACEFUL_SHUTDOWN_TIMEOUT = 5.0
"""Default timeout for shutting down services in seconds."""

DEFAULT_SHUTDOWN_ACK_TIMEOUT = 5.0
"""Default timeout for waiting for a shutdown command response in seconds."""

DEFAULT_PROFILE_CANCEL_TIMEOUT = 10.0
"""Default timeout for cancelling a profile run in seconds."""

TASK_CANCEL_TIMEOUT_SHORT = 2.0
"""Maximum time to wait for simple tasks to complete when cancelling them."""

DEFAULT_COMMS_REQUEST_TIMEOUT = 90.0
"""Default timeout for requests from req_clients to rep_clients in seconds."""

DEFAULT_PULL_CLIENT_MAX_CONCURRENCY = 100_000
"""Default maximum concurrency for pull clients."""

DEFAULT_SERVICE_REGISTRATION_TIMEOUT = 30.0
"""Default timeout for service registration in seconds."""

DEFAULT_SERVICE_START_TIMEOUT = 30.0
"""Default timeout for service start in seconds."""

DEFAULT_COMMAND_RESPONSE_TIMEOUT = 30.0
"""Default timeout for command responses in seconds."""

DEFAULT_CONNECTION_PROBE_INTERVAL = 0.1
"""Default interval for connection probes in seconds until a response is received."""

DEFAULT_CONNECTION_PROBE_TIMEOUT = 30.0
"""Maximum amount of time to wait for connection probe response."""

DEFAULT_PROFILE_CONFIGURE_TIMEOUT = 300.0
"""Default timeout for profile configure command in seconds."""

DEFAULT_PROFILE_START_TIMEOUT = 60.0
"""Default timeout for profile start command in seconds."""

DEFAULT_MAX_REGISTRATION_ATTEMPTS = 10
"""Default maximum number of registration attempts for component services before giving up."""

DEFAULT_REGISTRATION_INTERVAL = 1.0
"""Default interval between registration attempts in seconds for component services."""

DEFAULT_HEARTBEAT_INTERVAL = 5.0
"""Default interval between heartbeat messages in seconds for component services."""

AIPERF_DEV_MODE = os.getenv("AIPERF_DEV_MODE", "false").lower() in ("true", "1")

DEFAULT_UI_MIN_UPDATE_PERCENT = 1.0
"""Default minimum percentage difference from the last update to trigger a UI update (for non-dashboard UIs)."""

DEFAULT_WORKER_CHECK_INTERVAL = 1.0
"""Default interval between worker checks in seconds for the WorkerManager."""

DEFAULT_WORKER_HIGH_LOAD_CPU_USAGE = 75.0
"""Default CPU usage threshold for a worker to be considered high load."""

DEFAULT_WORKER_HIGH_LOAD_RECOVERY_TIME = 5.0
"""Default time in seconds from the last time a worker was in high load before it is considered healthy again."""

DEFAULT_WORKER_ERROR_RECOVERY_TIME = 3.0
"""Default time in seconds from the last time a worker had an error before it is considered healthy again."""

DEFAULT_WORKER_STALE_TIME = 10.0
"""Default time in seconds from the last time a worker reported any status before it is considered stale."""

DEFAULT_WORKER_STATUS_SUMMARY_INTERVAL = 0.5
"""Default interval in seconds between worker status summary messages."""

DEFAULT_REALTIME_METRICS_INTERVAL = 5.0
"""Default interval in seconds between real-time metrics messages."""

DEFAULT_CREDIT_PROGRESS_REPORT_INTERVAL = 2.0
"""Default interval in seconds between credit progress report messages."""

DEFAULT_RECORDS_PROGRESS_REPORT_INTERVAL = 2.0
"""Default interval in seconds between records progress report messages."""

DEFAULT_WORKER_HEALTH_CHECK_INTERVAL = 2.0
"""Default interval in seconds between worker health check messages."""

DEFAULT_RECORD_PROCESSOR_SCALE_FACTOR = 4
"""Default scale factor for the number of record processors to spawn based on the number of workers.
This will spawn 1 record processor for every X workers."""

DEFAULT_MAX_WORKERS_CAP = 32
"""Default absolute maximum number of workers to spawn, regardless of the number of CPU cores.
Only applies if the user does not specify a max workers value."""

DEFAULT_ZMQ_CONTEXT_TERM_TIMEOUT = 10.0
"""Default timeout for terminating the ZMQ context in seconds."""

AIPERF_HTTP_CONNECTION_LIMIT = int(os.environ.get("AIPERF_HTTP_CONNECTION_LIMIT", 2500))
"""Maximum number of concurrent connections for HTTP clients."""

GOOD_REQUEST_COUNT_TAG = "good_request_count"
"""GoodRequestCount metric tag"""

DEFAULT_RECORD_EXPORT_BATCH_SIZE = 100
"""Default batch size for record export results processor."""
