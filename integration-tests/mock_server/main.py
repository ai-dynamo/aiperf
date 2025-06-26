# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Main entry point for the integration test server."""

import logging

import typer
import uvicorn

from .app import set_server_config
from .config import ConfigDefaults, MockServerConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

app = typer.Typer()


@app.command()
def main(
    port: int = typer.Option(
        ConfigDefaults.PORT,
        "--port",
        "-p",
        help="Port to run the server on (env: SERVER_PORT)",
    ),
    host: str = typer.Option(
        ConfigDefaults.HOST,
        "--host",
        "-h",
        help="Host to bind the server to (env: SERVER_HOST)",
    ),
    ttft: float = typer.Option(
        ConfigDefaults.TTFT_MS,
        "--ttft",
        "--time-to-first-token-ms",
        help="Time to first token latency in milliseconds (env: TTFT_MS)",
    ),
    itl: float = typer.Option(
        ConfigDefaults.ITL_MS,
        "--itl",
        "--inter-token-latency-ms",
        help="Inter-token latency in milliseconds (env: ITL_MS)",
    ),
    workers: int = typer.Option(
        ConfigDefaults.WORKERS,
        "--workers",
        "-w",
        help="Number of worker processes (env: SERVER_WORKERS)",
    ),
    log_level: str = typer.Option(
        ConfigDefaults.LOG_LEVEL,
        "--log-level",
        help="Set the logging level",
    ),
    tokenizer_models: list[str] = typer.Option(
        ConfigDefaults.TOKENIZER_MODELS,
        "--tokenizer-models",
        "-t",
        help="List of tokenizer models to pre-load (e.g., --tokenizer-models gpt2 --tokenizer-models bert-base-uncased)",
    ),
):
    """Start the AI Performance Integration Test Server."""

    # Set logging level
    logging.root.setLevel(getattr(logging, log_level))

    # Create server configuration from CLI arguments and environment variables
    config = MockServerConfig(
        port=port,
        host=host,
        ttft_ms=ttft,
        itl_ms=itl,
        workers=workers,
        tokenizer_models=tokenizer_models,
    )

    # Set the global server configuration
    set_server_config(config)

    logger.info("Starting AI Performance Integration Test Server")
    logger.info(f"Server configuration: {config.model_dump()}")

    # Start the server
    uvicorn.run(
        "mock_server.app:app",
        host=config.host,
        port=config.port,
        log_level=log_level.lower(),
        access_log=log_level.lower() == "debug",
        workers=config.workers,
    )


if __name__ == "__main__":
    app()
