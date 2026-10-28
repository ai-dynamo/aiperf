# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Constants for the end-to-end testing framework.
"""

# Tag patterns
SETUP_TAG_PREFIX = "setup-"
SETUP_FILE_TAG_PREFIX = "setup-file-"
HEALTH_CHECK_TAG_PREFIX = "health-check-"
AIPERF_RUN_TAG_PREFIX = "aiperf-run-"
TAG_SUFFIX = "endpoint-server"

# Tag lengths for parsing
SETUP_TAG_PREFIX_LEN = len(SETUP_TAG_PREFIX)
SETUP_FILE_TAG_PREFIX_LEN = len(SETUP_FILE_TAG_PREFIX)
HEALTH_CHECK_TAG_PREFIX_LEN = len(HEALTH_CHECK_TAG_PREFIX)
AIPERF_RUN_TAG_PREFIX_LEN = len(AIPERF_RUN_TAG_PREFIX)
TAG_SUFFIX_LEN = len(TAG_SUFFIX)

# Attributes each tag kind reads. Anything else on a harness tag is rejected:
# an unread attribute is nearly always a misspelling of one of these, and
# silently ignoring it leaves the guide author believing it applied.
RUN_TAG_ATTRS = frozenset({"weight", "timeout"})
FILE_TAG_ATTRS = frozenset({"path"})
# Lifecycle tags read no attributes: a server boots under the setup monitor
# and the guide's own health-check deadline, so a `weight=` or `timeout=` here
# would be accepted and never read.
LIFECYCLE_TAG_ATTRS: frozenset[str] = frozenset()

# The only fence language a command tag may introduce. The tagged block is a
# shell command; a guide labelling it anything else is not describing one, and
# the read fails loudly rather than dropping the command.
BASH_INFO_STRING = "bash"

# AIPerf execution
AIPERF_UI_TYPE = "simple"

# Timeouts
SETUP_MONITOR_TIMEOUT = 30  # seconds to monitor setup output
CONTAINER_BUILD_TIMEOUT = 600  # seconds for Docker build
CONTAINER_START_TIMEOUT = 60  # seconds for container startup
CONTAINER_DISCOVERY_TIMEOUT = 30  # seconds for a `docker ps` snapshot
FIXTURE_WRITE_TIMEOUT = 30  # seconds to pipe one fixture into the container
AIPERF_COMMAND_TIMEOUT = 1200  # seconds for AIPerf commands
# Grace between the in-container SIGTERM at the deadline and the SIGKILL that
# follows it, for a command that traps or ignores SIGTERM.
CONTAINER_TIMEOUT_KILL_GRACE = 10  # seconds
