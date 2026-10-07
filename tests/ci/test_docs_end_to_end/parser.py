# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    SETUP_FILE_TAG_PREFIX,
    SETUP_FILE_TAG_PREFIX_LEN,
    SETUP_TAG_PREFIX,
    SETUP_TAG_PREFIX_LEN,
    TAG_SUFFIX,
    TAG_SUFFIX_LEN,
)
from data_types import Command, FileFixture, Server

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
        # Tags inside a fenced block are documentation *about* the tag syntax,
        # not tags. Without this, docs/reference/docs-e2e-tagging.md's own
        # examples register as real commands against real server groups.
        fence: tuple[str, int] | None = None
        while i < len(lines):
            line = lines[i].strip()

            fence = self._next_fence_state(fence, lines[i])
            if fence is not None:
                i += 1
                continue

            # Look for HTML comment tags. Two forms:
            #   <!-- tag-name -->
            #   <!-- tag-name weight=<int> -->  (optional runtime hint, seconds)
            # The weight hint is only meaningful on opening aiperf-run tags;
            # it's silently ignored on closing tags (which the categorizer
            # filters out anyway).
            if line.startswith("<!--") and line.endswith("-->"):
                tag_match = re.match(r"<!--\s*(\S+)((?:\s+\w+=\S+)*)\s*-->", line)
                if tag_match:
                    tag_name = tag_match.group(1).strip()
                    attrs = dict(re.findall(r"(\w+)=(\S+)", tag_match.group(2) or ""))
                    weight_str = attrs.get("weight")

                    # setup-file- must be checked before setup-: it is a
                    # longer prefix of the same namespace, and the block it
                    # introduces is yaml/json/jsonl rather than bash.
                    if self._is_file_tag(tag_name):
                        path_attr = attrs.get("path")
                        if not path_attr:
                            logger.error(
                                f"{tag_name} at {file_path}:{i + 1} has no path= attribute; skipping"
                            )
                        else:
                            content = self._extract_fenced_block(lines, i + 1)
                            if content is None:
                                logger.warning(
                                    f"No fenced block found after tag {tag_name}"
                                )
                            else:
                                self._add_file_fixture(
                                    tag_name, path_attr, content, file_path, i + 1
                                )
                        i += 1
                        continue

                    # Check for setup or aiperf-run tags ending with endpoint-server
                    if self._is_target_tag(tag_name):
                        logger.info(f"Found target tag: {tag_name}")

                        # Extract the bash command
                        bash_content = self._extract_bash_block(lines, i + 1)

                        if bash_content:
                            command_kwargs = dict(
                                tag_name=tag_name,
                                command=bash_content,
                                file_path=file_path,
                                start_line=i + 1,
                                end_line=i + len(bash_content.split("\n")) + 2,
                            )
                            timeout_str = attrs.get("timeout")
                            weight = self._positive_int(
                                weight_str, "weight", tag_name, file_path, i + 1
                            )
                            timeout = self._positive_int(
                                timeout_str, "timeout", tag_name, file_path, i + 1
                            )
                            if (weight_str is not None and weight is None) or (
                                timeout_str is not None and timeout is None
                            ):
                                i += 1
                                continue
                            if weight is not None:
                                command_kwargs["weight"] = weight
                            if timeout is not None:
                                command_kwargs["timeout"] = timeout
                            command = Command(**command_kwargs)

                            self._categorize_command(command)
                        else:
                            logger.warning(f"No bash block found after tag {tag_name}")
            i += 1

    @staticmethod
    def _next_fence_state(
        fence: tuple[str, int] | None, raw_line: str
    ) -> tuple[str, int] | None:
        """Advance CommonMark fence state by one line.

        Returns the open fence as ``(char, length)``, or None outside one. A
        closer must use the same character and be at least as long as its
        opener, so a four-backtick block may quote a three-backtick block.

        Takes the *unstripped* line: CommonMark allows a fence marker at most
        three spaces of indentation, and at four it is an indented code block
        rather than a boundary. Stripping first would let a deeper-indented
        marker inside an example close the fence early, after which the
        following example tags parse as real commands.
        """
        match = re.match(r"( {0,3})(`{3,}|~{3,})(.*)$", raw_line)
        if fence is None:
            return (match.group(2)[0], len(match.group(2))) if match else None
        char, length = fence
        if (
            match
            and match.group(2)[0] == char
            and len(match.group(2)) >= length
            and not match.group(3).strip()
        ):
            return None
        return fence

    @staticmethod
    def _positive_int(
        raw: str | None,
        attr: str,
        tag_name: str,
        file_path: Path,
        line: int,
    ) -> int | None:
        """Parse a positive-integer tag attribute, or None if absent or invalid.

        An invalid value isolates to its own command rather than aborting
        discovery for every document: one typo in one guide must not silently
        drop the whole docs-e2e suite. Zero and negatives are rejected because
        both are quietly destructive -- ``timeout=0`` falls through to the
        global default and ``timeout=-1`` kills the command the moment it
        starts.
        """
        if raw is None:
            return None
        try:
            value = int(raw)
        except ValueError:
            logger.error(
                f"Ignoring command with non-integer {attr}={raw!r} in tag "
                f"{tag_name} ({file_path}:{line})"
            )
            return None
        if value <= 0:
            logger.error(
                f"Ignoring command with non-positive {attr}={value} in tag "
                f"{tag_name} ({file_path}:{line})"
            )
            return None
        return value

    def _is_file_tag(self, tag_name: str) -> bool:
        """Whether this tag declares a file to materialize before the run."""
        return (
            tag_name.startswith(SETUP_FILE_TAG_PREFIX)
            and tag_name.endswith(TAG_SUFFIX)
            and not tag_name.startswith("/")
        )

    def _extract_fenced_block(self, lines: list[str], start_idx: int) -> str | None:
        """Extract the next fenced block regardless of its language.

        ``_extract_bash_block`` deliberately only accepts ```bash; a config
        fixture is yaml/json/jsonl, so it needs its own reader.
        """
        i = start_idx
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("```"):
                break
            if line and not line.startswith("#"):
                return None
            i += 1
        else:
            return None

        body: list[str] = []
        i += 1
        while i < len(lines):
            if lines[i].strip() == "```":
                return "".join(body)
            body.append(lines[i])
            i += 1
        return None

    def _add_file_fixture(
        self, tag_name: str, path: str, content: str, file_path: str, line_no: int
    ) -> None:
        server_name = tag_name[SETUP_FILE_TAG_PREFIX_LEN:-TAG_SUFFIX_LEN].rstrip("-")
        server = self.servers.get(server_name)
        if server is None:
            server = Server(
                name=server_name,
                setup_command=None,
                health_check_command=None,
                aiperf_commands=[],
            )
            self.servers[server_name] = server
        existing = next((f for f in server.files if f.path == path), None)
        if existing is not None:
            if existing.content == content:
                # The same file documented twice: harmless, write it once.
                return
            logger.error(
                f"{file_path}:{line_no}: fixture '{path}' for server "
                f"'{server_name}' conflicts with {existing.file_path}:"
                f"{existing.start_line}. Both are written before any command "
                f"runs, so one guide would silently benchmark the other's file."
            )
            server.fixture_conflicts.append(path)
            return
        server.files.append(
            FileFixture(
                path=path, content=content, file_path=file_path, start_line=line_no
            )
        )
        logger.info(f"Registered file fixture {path} for server {server_name}")

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
