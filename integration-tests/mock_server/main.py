# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Main entry point for the integration test server."""

import logging

import cyclopts
import uvicorn

from .app import set_server_config
from .config import MockServerConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

app = cyclopts.App(name="aiperf-mock-server", help="AIPerf Integration Test Server")


@app.default
def main(
    config: MockServerConfig,
) -> None:
    """Start the AIPerf Integration Test Server.

    Configuration priority (highest to lowest):
    1. CLI arguments
    2. Environment variables (prefixed with MOCK_SERVER_)
    3. Default values
    """

    # Set logging level
    logging.root.setLevel(getattr(logging, config.log_level.upper()))

    # Set the global server configuration
    set_server_config(config)

    logger.info("Starting AIPerf Integration Test Server")
    logger.info("Server configuration: %s", config.model_dump())

    # Start the server
    uvicorn.run(
        "mock_server.app:app",
        host=config.host,
        port=config.port,
        log_level=config.log_level.lower(),
        access_log=config.access_logs or config.log_level.lower() == "debug",
        workers=config.workers,
    )


if __name__ == "__main__":
    app()
