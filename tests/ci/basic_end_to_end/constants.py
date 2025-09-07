"""
Constants for the end-to-end testing framework.
"""

# Tag patterns
SETUP_TAG_PREFIX = "setup-"
HEALTH_CHECK_TAG_PREFIX = "health-check-"
AIPERF_RUN_TAG_PREFIX = "aiperf-run-"
TAG_SUFFIX = "endpoint-server"

# Tag lengths for parsing
SETUP_TAG_PREFIX_LEN = len(SETUP_TAG_PREFIX)
HEALTH_CHECK_TAG_PREFIX_LEN = len(HEALTH_CHECK_TAG_PREFIX)
AIPERF_RUN_TAG_PREFIX_LEN = len(AIPERF_RUN_TAG_PREFIX)
TAG_SUFFIX_LEN = len(TAG_SUFFIX)

# AIPerf execution
AIPERF_UI_TYPE = "simple"

# Timeouts
SETUP_MONITOR_TIMEOUT = 30  # seconds to monitor setup output
CONTAINER_BUILD_TIMEOUT = 600  # seconds for Docker build
CONTAINER_START_TIMEOUT = 60   # seconds for container startup
AIPERF_COMMAND_TIMEOUT = 300   # seconds for AIPerf commands
