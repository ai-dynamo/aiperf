#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Simple end-to-end test tool for AIPerf documentation.

Parses markdown files for server setup and AIPerf run commands,
builds AIPerf container, and executes tests.
"""

import logging
import sys

from parser import MarkdownParser
from test_runner import EndToEndTestRunner
from utils import get_repo_root, setup_logging

# Configure logging using centralized utility
setup_logging()
logger = logging.getLogger(__name__)


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run end-to-end tests from markdown documentation"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show discovered commands without executing",
    )
    parser.add_argument(
        "--all-servers",
        action="store_true",
        help="Run tests for all discovered servers",
    )
    args = parser.parse_args()

    # Get repository root using centralized function
    repo_root = get_repo_root()

    # Parse markdown files
    md_parser = MarkdownParser()
    servers = md_parser.parse_directory(str(repo_root))

    if not servers:
        logger.warning("No servers found")
        return 0

    logger.info(f"Discovered {len(servers)} servers:")
    for name, server in servers.items():
        setup_file = (
            server.setup_command.file_path if server.setup_command else "MISSING"
        )
        health_file = (
            server.health_check_command.file_path
            if server.health_check_command
            else "MISSING"
        )
        aiperf_count = len(server.aiperf_commands)
        logger.info(
            f"  {name}: setup={setup_file}, health={health_file}, aiperf_commands={aiperf_count}"
        )

    if args.dry_run:
        logger.info("Dry run completed")
        return 0

    # Run tests
    runner = EndToEndTestRunner()
    success = runner.run_tests(servers)

    if success:
        logger.info("All tests passed!")
        return 0
    else:
        logger.error("Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
