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

# The exact fence a command block opens and closes with. Literal strings, not
# CommonMark matching: see ``MarkdownParser._extract_bash_block``.
BASH_FENCE_OPENER = "```bash"
FENCE_CLOSER = "```"

# AIPerf execution
AIPERF_UI_TYPE = "simple"

# Timeouts
SETUP_MONITOR_TIMEOUT = 30  # seconds to monitor setup output
CONTAINER_BUILD_TIMEOUT = 600  # seconds for Docker build
CONTAINER_START_TIMEOUT = 60  # seconds for container startup
CONTAINER_DISCOVERY_TIMEOUT = 30  # seconds for a `docker ps` snapshot
FIXTURE_WRITE_TIMEOUT = 30  # seconds to pipe one fixture into the container
AIPERF_COMMAND_TIMEOUT = 1200  # seconds for AIPerf commands
# The in-container deadline sits this far past the host watchdog's, so the
# watchdog always fires first and keeps ownership of the timeout diagnostic
# ("exceeded Ns timeout"). Reaching the in-container deadline at the same
# moment would race it: the process would already be gone, the watchdog's
# guard would short-circuit, and the run would report a bare exit 124.
CONTAINER_DEADLINE_SLACK = 15  # seconds
# Grace between the in-container SIGTERM and the SIGKILL that follows it, for
# a command that traps or ignores SIGTERM.
CONTAINER_TIMEOUT_KILL_GRACE = 10  # seconds
