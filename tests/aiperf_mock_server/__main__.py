# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AIPerf Mock Server entry point."""

import logging
import sys

import cyclopts
import uvicorn
from aiperf_mock_server.config import MockServerConfig, set_server_config

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
