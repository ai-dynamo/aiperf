# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

NANOS_PER_SECOND = 1_000_000_000
NANOS_PER_MILLIS = 1_000_000
BYTES_PER_MIB = 1024 * 1024

GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS = 5.0

TASK_CANCEL_TIMEOUT_SHORT = 2.0
"""Maximum time to wait for simple tasks to complete when cancelling them."""

TASK_CANCEL_TIMEOUT_LONG = 5.0
"""Maximum time to wait for complex tasks to complete when cancelling them (like parent tasks)."""

DEFAULT_COMMS_REQUEST_TIMEOUT = 10.0
"""Default timeout for requests from req_clients to rep_clients in seconds."""

DEFAULT_PULL_CLIENT_MAX_CONCURRENCY = 100_000
"""Default maximum concurrency for pull clients."""

DEFAULT_SERVICE_REGISTRATION_TIMEOUT = 5.0
"""Default timeout for service registration in seconds."""

DEFAULT_SERVICE_START_TIMEOUT = 5.0
"""Default timeout for service start in seconds."""

DEFAULT_STREAMING_MAX_QUEUE_SIZE = 100_000
"""Default maximum queue size for streaming post processors."""
