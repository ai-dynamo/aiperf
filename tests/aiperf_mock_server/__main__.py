# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AIPerf Mock Server entry point."""

import json
import logging
import os
import sys

import cyclopts
import uvicorn
from aiperf_mock_server.app import set_server_config
from aiperf_mock_server.config import MockServerConfig

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = cyclopts.App(name="aiperf-mock-server", help="AIPerf Mock Server")


@app.default
def serve(config: MockServerConfig | None = None) -> None:
    """Start the AIPerf Mock Server.

    Configuration priority (highest to lowest):
    1. CLI arguments
    2. Environment variables (MOCK_SERVER_* prefix)
    3. Default values
    """
    if config is None:
        config = MockServerConfig()

    # Propagate config to environment variables for worker processes
    for key, value in config.model_dump().items():
        if value is not None:
            env_key = f"MOCK_SERVER_{key.upper()}"
            env_value = (
                json.dumps(value) if isinstance(value, list | dict) else str(value)
            )
            os.environ[env_key] = env_value

    logging.root.setLevel(getattr(logging, config.log_level.upper()))
    set_server_config(config)

    logger.info("Starting AIPerf Mock Server")
    logger.info("Config: %s", config.model_dump())

    uvicorn.run(
        "aiperf_mock_server.app:app",
        host=config.host,
        port=config.port,
        log_level=config.log_level.lower(),
        access_log=config.access_logs or config.log_level.lower() == "debug",
        workers=config.workers,
    )


def main() -> None:
    sys.exit(app())


if __name__ == "__main__":
    main()
