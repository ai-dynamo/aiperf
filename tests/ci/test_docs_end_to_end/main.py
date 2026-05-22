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
    target = parser.add_mutually_exclusive_group()
    target.add_argument(
        "--all-servers",
        action="store_true",
        help="Run tests for all discovered servers (default behavior)",
    )
    target.add_argument(
        "--server",
        type=str,
        default=None,
        help="Run tests only for the named server (matrix-shard entry point)",
    )
    # Shard params let a single server's command list be split across runners.
    # Useful for the big chat server (40+ commands) — without sharding it
    # dominates the wall-clock of the whole GPU job.
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="0-based index of this shard (requires --shard-total and --server)",
    )
    parser.add_argument(
        "--shard-total",
        type=int,
        default=1,
        help="Total shard count for the selected server's commands",
    )
    args = parser.parse_args()

    if args.shard_total < 1:
        logger.error("--shard-total must be >= 1")
        return 2
    if args.shard_total > 1:
        if args.server is None:
            logger.error("--shard-total > 1 requires --server")
            return 2
        if not 0 <= args.shard_index < args.shard_total:
            logger.error(
                f"--shard-index {args.shard_index} out of range for "
                f"--shard-total {args.shard_total}"
            )
            return 2

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

    if args.server is not None:
        if args.server not in servers:
            logger.error(
                f"--server '{args.server}' not found among discovered servers: "
                f"{sorted(servers.keys())}"
            )
            return 1
        servers = {args.server: servers[args.server]}

    if args.shard_total > 1:
        server = servers[args.server]
        n = len(server.aiperf_commands)
        # LPT (Longest Processing Time) bin-packing: sort commands by their
        # author-annotated ``weight=`` hint (seconds), then greedily place
        # each into the currently-lightest shard. Within a shard, restore
        # docs order so the runner's per-test logs read top-to-bottom.
        # Pure greedy LPT has a 4/3 - 1/(3m) makespan approximation —
        # close enough to optimal for the ~50-command scale here, and
        # immune to docs reordering (unlike contiguous-chunk slicing).
        shard_bins: list[list] = [[] for _ in range(args.shard_total)]
        shard_load: list[int] = [0] * args.shard_total
        # Tuple key: heaviest first, then (file_path, start_line) for a
        # stable, OS-independent secondary sort. ``Path.rglob`` returns
        # files in filesystem-dependent order (macOS vs Linux differ), so
        # without the tuple tie-break the same code would assign tests to
        # different shards locally vs in CI — a real bug we hit on the
        # 2026-05-19 rebalance pass.
        sorted_cmds = sorted(
            server.aiperf_commands,
            key=lambda c: (-c.weight, c.file_path, c.start_line),
        )
        for cmd in sorted_cmds:
            target = min(range(args.shard_total), key=lambda i: shard_load[i])
            shard_bins[target].append(cmd)
            shard_load[target] += cmd.weight
        my_bin = shard_bins[args.shard_index]
        my_bin.sort(key=lambda c: (c.file_path, c.start_line))
        server.aiperf_commands = my_bin
        logger.info(
            f"Shard {args.shard_index + 1}/{args.shard_total} of "
            f"'{args.server}': {len(my_bin)} of {n} commands, "
            f"estimated weight {shard_load[args.shard_index]}s "
            f"(shard weights: {shard_load})"
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
