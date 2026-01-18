# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Markdown parser for extracting server setup and AIPerf run commands.
"""

import logging
import re
import sys
from pathlib import Path

from constants import (
    AIPERF_RUN_TAG_PREFIX,
    AIPERF_RUN_TAG_PREFIX_LEN,
    HEALTH_CHECK_TAG_PREFIX,
    HEALTH_CHECK_TAG_PREFIX_LEN,
    SETUP_TAG_PREFIX,
    SETUP_TAG_PREFIX_LEN,
    TAG_SUFFIX,
    TAG_SUFFIX_LEN,
)
from data_types import Command, Server

logger = logging.getLogger(__name__)


class MarkdownParser:
    """Parses markdown files for server setup and aiperf run commands"""

    def __init__(self):
        self.servers: dict[str, Server] = {}

    def parse_directory(self, directory: str) -> dict[str, Server]:
        """Parse all markdown files in directory and extract commands"""
        logger.info(f"Parsing markdown files in {directory}")

        for file_path in Path(directory).rglob("*.md"):
            logger.info(f"Parsing file: {file_path}")
            self._parse_file(str(file_path))

        return self.servers

    def _parse_file(self, file_path: str):
        """Parse a single markdown file for tagged commands"""
        try:
            with open(file_path, encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Look for HTML comment tags
            if line.startswith("<!--") and line.endswith("-->"):
                tag_match = re.match(r"<!--\s*([^-\s]+.*?)\s*-->", line)
                if tag_match:
                    tag_name = tag_match.group(1).strip()

                    # Check for setup or aiperf-run tags ending with endpoint-server
                    if self._is_target_tag(tag_name):
                        logger.info(f"Found target tag: {tag_name}")

                        # Extract the bash command
                        bash_content = self._extract_bash_block(lines, i + 1)

                        if bash_content:
                            command = Command(
                                tag_name=tag_name,
                                command=bash_content,
                                file_path=file_path,
                                start_line=i + 1,
                                end_line=i + len(bash_content.split("\n")) + 2,
                            )

                            self._categorize_command(command)
                        else:
                            logger.warning(f"No bash block found after tag {tag_name}")
            i += 1

    def _is_target_tag(self, tag_name: str) -> bool:
        """Check if tag is a setup, health-check, or aiperf-run command for endpoint servers"""
        return (
            (
                tag_name.startswith(SETUP_TAG_PREFIX)
                or tag_name.startswith(HEALTH_CHECK_TAG_PREFIX)
                or tag_name.startswith(AIPERF_RUN_TAG_PREFIX)
            )
            and tag_name.endswith(TAG_SUFFIX)
            and not tag_name.startswith("/")
        )

    def _extract_bash_block(self, lines: list[str], start_idx: int) -> str | None:
        """Extract bash code block starting from the given index"""
        i = start_idx

        # Find ```bash
        while i < len(lines):
            line = lines[i].strip()
            if line == "```bash":
                break
            elif line and not line.startswith("#"):
                return None
            i += 1
        else:
            return None

        # Extract content until closing ```
        bash_lines = []
        i += 1
        while i < len(lines):
            line = lines[i]
            if line.strip() == "```":
                return "".join(bash_lines).strip()
            bash_lines.append(line)
            i += 1

        return None

    def _categorize_command(self, command: Command):
        """Categorize command and add to appropriate server"""
        tag_name = command.tag_name

        if tag_name.startswith(SETUP_TAG_PREFIX):
            # Extract server name: setup-{server-name}-endpoint-server
            server_name = tag_name[SETUP_TAG_PREFIX_LEN:-TAG_SUFFIX_LEN].rstrip(
                "-"
            )  # Remove prefix, suffix, and trailing dash

            if server_name not in self.servers:
                self.servers[server_name] = Server(
                    name=server_name,
                    setup_command=None,
                    health_check_command=None,
                    aiperf_commands=[],
                )

            if self.servers[server_name].setup_command is not None:
                logger.error(f"DUPLICATE SETUP COMMAND for server '{server_name}'")
                logger.error(
                    f"  First: {self.servers[server_name].setup_command.file_path}"
                )
                logger.error(f"  Second: {command.file_path}")
                sys.exit(1)

            self.servers[server_name].setup_command = command

        elif tag_name.startswith(HEALTH_CHECK_TAG_PREFIX):
            # Extract server name: health-check-{server-name}-endpoint-server
            server_name = tag_name[HEALTH_CHECK_TAG_PREFIX_LEN:-TAG_SUFFIX_LEN].rstrip(
                "-"
            )  # Remove prefix, suffix, and trailing dash

            if server_name not in self.servers:
                self.servers[server_name] = Server(
                    name=server_name,
                    setup_command=None,
                    health_check_command=None,
                    aiperf_commands=[],
                )

            if self.servers[server_name].health_check_command is not None:
                logger.error(
                    f"DUPLICATE HEALTH CHECK COMMAND for server '{server_name}'"
                )
                logger.error(
                    f"  First: {self.servers[server_name].health_check_command.file_path}"
                )
                logger.error(f"  Second: {command.file_path}")
                sys.exit(1)

            self.servers[server_name].health_check_command = command

        elif tag_name.startswith(AIPERF_RUN_TAG_PREFIX):
            # Extract server name: aiperf-run-{server-name}-endpoint-server
            server_name = tag_name[AIPERF_RUN_TAG_PREFIX_LEN:-TAG_SUFFIX_LEN].rstrip(
                "-"
            )  # Remove prefix, suffix, and trailing dash

            if server_name not in self.servers:
                self.servers[server_name] = Server(
                    name=server_name,
                    setup_command=None,
                    health_check_command=None,
                    aiperf_commands=[],
                )

            self.servers[server_name].aiperf_commands.append(command)
