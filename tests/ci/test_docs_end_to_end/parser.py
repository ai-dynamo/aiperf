# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Markdown parser for extracting server setup and AIPerf run commands.
"""

import logging
import re
import subprocess
from pathlib import Path, PurePosixPath

from constants import (
    AIPERF_RUN_TAG_PREFIX,
    AIPERF_RUN_TAG_PREFIX_LEN,
    BASH_INFO_STRING,
    FILE_TAG_ATTRS,
    HEALTH_CHECK_TAG_PREFIX,
    HEALTH_CHECK_TAG_PREFIX_LEN,
    LIFECYCLE_TAG_ATTRS,
    RUN_TAG_ATTRS,
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

        for file_path in self._markdown_files(directory):
            logger.info(f"Parsing file: {file_path}")
            self._parse_file(str(file_path))

        return self.servers

    @staticmethod
    def _markdown_files(directory: str) -> list[Path]:
        """Markdown files the repository tracks, sorted for a stable order.

        A plain rglob also walks ignored trees -- a git worktree kept under the
        checkout, a local virtualenv, a sibling clone -- so a second copy of
        docs/ there reads as a duplicate setup tag and exits the process. Only
        tracked files can carry a tag the harness is meant to run.
        """
        root = Path(directory)
        try:
            result = subprocess.run(
                ["git", "-C", str(root), "ls-files", "-z", "--", "*.md"],
                capture_output=True,
                check=True,
            )
        except (OSError, subprocess.CalledProcessError):
            logger.warning(f"{root} is not a git checkout; falling back to rglob")
            return sorted(root.rglob("*.md"))

        names = result.stdout.decode("utf-8").split("\0")
        return sorted(root / name for name in names if name)

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
        # not tags. Without this, any page that spells a tag out to explain it
        # would have its own examples registered as real commands against real
        # server groups -- see this harness's README, which documents the tags
        # only in placeholder form for exactly that reason.
        fence: tuple[str, int] | None = None
        while i < len(lines):
            line = lines[i].strip()

            fence = self._next_fence_state(fence, lines[i])
            if fence is not None:
                i += 1
                continue

            # Look for HTML comment tags. Two forms:
            #   <!-- tag-name -->
            #   <!-- tag-name attr=<value> ... -->
            # Attributes parse generically but are then checked against the
            # set each tag kind reads: weight=<int> (runtime hint, seconds)
            # and timeout=<int> (per-command override) on aiperf-run, and
            # path=<relative path> on setup-file-. An unrecognised name is
            # rejected rather than ignored, because it is nearly always a
            # misspelling of one of these. weight and timeout are meaningful
            # only on opening tags; closing tags are filtered out by the
            # categorizer anyway.
            if line.startswith("<!--") and line.endswith("-->"):
                # Captured loosely and validated afterwards: a tag matched
                # strictly would fall through to prose when its attributes are
                # malformed, and the validation below cannot run on a tag that
                # was never matched.
                tag_match = re.match(r"<!--\s*(\S+)(.*?)-->", line)
                if tag_match:
                    tag_name = tag_match.group(1).strip()
                    attrs, malformed = self._parse_attrs(tag_match.group(2))
                    weight_str = attrs.get("weight")

                    # setup-file- must be checked before setup-: it is a
                    # longer prefix of the same namespace, and the block it
                    # introduces is yaml/json/jsonl rather than bash.
                    if self._is_file_tag(tag_name):
                        if self._reject_unknown_attrs(
                            tag_name, attrs, malformed, FILE_TAG_ATTRS, file_path, i + 1
                        ):
                            i += 1
                            continue
                        path_attr = attrs.get("path")
                        if not path_attr:
                            logger.error(
                                f"{tag_name} at {file_path}:{i + 1} has no path= attribute; skipping"
                            )
                            self._record_parse_error(
                                tag_name,
                                f"dropped the fixture at {file_path}:{i + 1} "
                                f"over a missing path= attribute",
                            )
                        else:
                            content = self._extract_fenced_block(lines, i + 1)
                            if content is None:
                                logger.warning(
                                    f"No fenced block found after tag {tag_name}"
                                )
                                self._record_parse_error(
                                    tag_name,
                                    f"dropped the fixture at {file_path}:{i + 1} "
                                    f"over a missing or unterminated fenced block",
                                )
                            else:
                                self._add_file_fixture(
                                    tag_name, path_attr, content, file_path, i + 1
                                )
                        i += 1
                        continue

                    # Check for setup or aiperf-run tags ending with endpoint-server
                    if self._is_target_tag(tag_name):
                        allowed = (
                            RUN_TAG_ATTRS
                            if tag_name.startswith(AIPERF_RUN_TAG_PREFIX)
                            else LIFECYCLE_TAG_ATTRS
                        )
                        if self._reject_unknown_attrs(
                            tag_name, attrs, malformed, allowed, file_path, i + 1
                        ):
                            i += 1
                            continue
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
                                self._record_parse_error(
                                    tag_name,
                                    f"dropped the command at {file_path}:{i + 1} "
                                    f"over an invalid tag attribute",
                                )
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
                            self._record_parse_error(
                                tag_name,
                                f"dropped the command at {file_path}:{i + 1} "
                                f"over a missing or unterminated bash block",
                            )
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
            if match is None:
                return None
            # CommonMark forbids a backtick inside a backtick fence's info
            # string, so a paragraph opening with inline code is text, not a
            # fence. Accepting it opens a block that never closes until the
            # next three backticks, and every tag in between stops being
            # collected -- a documented command silently leaves the suite
            # while the run still reports green.
            if match.group(2)[0] == "`" and "`" in match.group(3):
                return None
            return (match.group(2)[0], len(match.group(2)))
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
    def _parse_attrs(text: str | None) -> tuple[dict[str, str], list[str]]:
        """Split a tag's attribute region into readable pairs and junk.

        Returned rather than raised so the caller can decide: the same text
        after a tag name the harness does not own is an ordinary markdown
        comment, and policing those would reject every ``<!-- prettier-ignore
        -->`` in the docs.
        """
        attrs: dict[str, str] = {}
        malformed: list[str] = []
        for token in (text or "").split():
            pair = re.fullmatch(r"(\w+)=(\S+)", token)
            if pair is None:
                malformed.append(token)
            else:
                attrs[pair.group(1)] = pair.group(2)
        return attrs, malformed

    def _reject_unknown_attrs(
        self,
        tag_name: str,
        attrs: dict[str, str],
        malformed: list[str],
        allowed: frozenset[str],
        file_path: str,
        line: int,
    ) -> bool:
        """Whether this tag carries an attribute the harness does not read.

        An unreadable attribute is almost always a misspelling of one that is
        read, and ignoring it is the quiet kind of wrong: a misspelled
        ``timeout=`` looks like it raised the deadline, the command keeps the
        1200s default, and the SIGKILL twenty minutes later reads as the
        command being slow rather than the attribute never having applied.
        Treated like an invalid value -- drop the one command, record it, fail
        the run.

        ``allowed`` is empty for setup and health-check tags, where a
        correctly spelled ``timeout=`` is just as misleading: nothing reads it
        off a lifecycle command.
        """
        if malformed:
            logger.error(
                f"Ignoring command with malformed attribute(s) {malformed} in "
                f"tag {tag_name} ({file_path}:{line}); expected name=value "
                f"with no spaces"
            )
            self._record_parse_error(
                tag_name,
                f"dropped the block at {file_path}:{line} over malformed tag "
                f"attribute(s) {malformed}",
            )
            return True
        unknown = sorted(set(attrs) - allowed)
        if not unknown:
            return False
        expected = (
            f"expected one of {sorted(allowed)}"
            if allowed
            else "this tag kind reads no attributes"
        )
        logger.error(
            f"Ignoring command with unknown attribute(s) {unknown} in tag "
            f"{tag_name} ({file_path}:{line}); {expected}"
        )
        self._record_parse_error(
            tag_name,
            f"dropped the block at {file_path}:{line} over unknown tag "
            f"attribute(s) {unknown}",
        )
        return True

    @staticmethod
    def _positive_int(
        raw: str | None,
        attr: str,
        tag_name: str,
        file_path: str,
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

    @staticmethod
    def _server_name_for_tag(tag_name: str) -> str | None:
        """The server a tag belongs to, or None if it names no server.

        ``setup-file-`` is tried before ``setup-``: it is a longer prefix of
        the same namespace, so the shorter one would strip it to ``file-...``.
        """
        for prefix, prefix_len in (
            (SETUP_FILE_TAG_PREFIX, SETUP_FILE_TAG_PREFIX_LEN),
            (SETUP_TAG_PREFIX, SETUP_TAG_PREFIX_LEN),
            (HEALTH_CHECK_TAG_PREFIX, HEALTH_CHECK_TAG_PREFIX_LEN),
            (AIPERF_RUN_TAG_PREFIX, AIPERF_RUN_TAG_PREFIX_LEN),
        ):
            if tag_name.startswith(prefix):
                return tag_name[prefix_len:-TAG_SUFFIX_LEN].rstrip("-")
        return None

    def _ensure_server(self, server_name: str) -> Server:
        """The named server, created empty if this is its first tag."""
        server = self.servers.get(server_name)
        if server is None:
            server = Server(
                name=server_name,
                setup_command=None,
                health_check_command=None,
                aiperf_commands=[],
            )
            self.servers[server_name] = server
        return server

    def _record_parse_error(self, tag_name: str, problem: str) -> None:
        """Keep a rejected command visible in the run's final result.

        Dropping the command is right -- one typo must not abort discovery for
        every guide. Dropping it *silently* is not: a server whose remaining
        commands are intact still validates, runs them, and reports a pass
        while a documented command was never executed at all.
        """
        server_name = self._server_name_for_tag(tag_name)
        if server_name is None:
            return
        self._ensure_server(server_name).parse_errors.append(problem)

    def _is_file_tag(self, tag_name: str) -> bool:
        """Whether this tag declares a file to materialize before the run."""
        return tag_name.startswith(SETUP_FILE_TAG_PREFIX) and tag_name.endswith(
            TAG_SUFFIX
        )

    @staticmethod
    def _looks_like_tag(line: str) -> bool:
        """Whether a line is a harness tag rather than block content."""
        match = re.match(r"<!--\s*/?(\S+)", line)
        if match is None or not line.endswith("-->"):
            return False
        name = match.group(1)
        return name.endswith(TAG_SUFFIX) and name.startswith(
            (
                SETUP_FILE_TAG_PREFIX,
                SETUP_TAG_PREFIX,
                HEALTH_CHECK_TAG_PREFIX,
                AIPERF_RUN_TAG_PREFIX,
            )
        )

    def _extract_fenced_block(self, lines: list[str], start_idx: int) -> str | None:
        """Extract the next fenced block regardless of its language.

        ``_extract_bash_block`` deliberately only accepts ```bash; a config
        fixture is yaml/json/jsonl, so it needs its own reader.

        Fence matching goes through ``_next_fence_state`` so a four-backtick
        fixture may quote a three-backtick block and a tilde fence works at
        all, and so this reader cannot disagree with the tag scanner about
        where a block ends.
        """
        i = start_idx
        fence: tuple[str, int] | None = None
        while i < len(lines):
            fence = self._next_fence_state(None, lines[i])
            if fence is not None:
                break
            line = lines[i].strip()
            if line and not line.startswith("#"):
                return None
            i += 1
        if fence is None:
            return None

        body: list[str] = []
        i += 1
        while i < len(lines):
            if self._next_fence_state(fence, lines[i]) is None:
                return "".join(body)
            # A harness tag inside the body means the opener was never closed
            # and this reader has run past the end of the fixture. Hitting the
            # end of the file is the obvious shape of that mistake, but not
            # the common one: any block below closes the fixture on the
            # author's behalf, so the reader returns text that looks fine. It
            # is not -- the file written from it is markdown, and the command
            # whose tag it ate never registers. Refuse both.
            if self._looks_like_tag(lines[i].strip()):
                return None
            body.append(lines[i])
            i += 1
        return None

    def _add_file_fixture(
        self, tag_name: str, path: str, content: str, file_path: str, line_no: int
    ) -> None:
        server_name = self._server_name_for_tag(tag_name)
        server = self._ensure_server(server_name)
        # Compare destinations, not spellings: ``config.yaml`` and
        # ``./config.yaml`` are written to the same file by
        # ``_materialize_files``, so a raw string compare would miss the
        # collision and let the second guide silently overwrite the first.
        destination = PurePosixPath(path)
        existing = next(
            (f for f in server.files if PurePosixPath(f.path) == destination), None
        )
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
        """Extract the ```bash block a command tag introduces.

        Only ``bash`` is accepted: a tagged command is shell, and a guide that
        labels the block anything else is not describing something this
        harness runs. Fence matching goes through ``_next_fence_state`` so
        this reader cannot disagree with the tag scanner about where a block
        ends, and a harness tag inside the body means the opener was never
        closed -- the same refusal ``_extract_fenced_block`` makes. Returning
        None is a read failure the caller records as a parse error; a command
        never leaves the suite silently.
        """
        i = start_idx
        fence: tuple[str, int] | None = None
        while i < len(lines):
            fence = self._next_fence_state(None, lines[i])
            if fence is not None:
                break
            line = lines[i].strip()
            if line and not line.startswith("#"):
                return None
            i += 1
        if fence is None:
            return None
        if lines[i].strip().lstrip("`~").strip() != BASH_INFO_STRING:
            return None

        body: list[str] = []
        i += 1
        while i < len(lines):
            if self._next_fence_state(fence, lines[i]) is None:
                return "".join(body).strip()
            if self._looks_like_tag(lines[i].strip()):
                return None
            body.append(lines[i])
            i += 1
        return None

    def _categorize_command(self, command: Command):
        """Categorize command and add to appropriate server"""
        tag_name = command.tag_name

        if tag_name.startswith(SETUP_TAG_PREFIX):
            server_name = self._server_name_for_tag(tag_name)

            self._ensure_server(server_name)

            if self.servers[server_name].setup_command is not None:
                # Recorded rather than fatal: aborting the process here takes
                # every other documented server down with it, and one guide
                # disagreeing with another about this server says nothing
                # about the rest. _validate_servers skips this server and the
                # run still fails.
                logger.error(f"DUPLICATE SETUP COMMAND for server '{server_name}'")
                logger.error(
                    f"  First: {self.servers[server_name].setup_command.file_path}"
                )
                logger.error(f"  Second: {command.file_path}")
                self.servers[server_name].command_conflicts.append("setup")
                return

            self.servers[server_name].setup_command = command

        elif tag_name.startswith(HEALTH_CHECK_TAG_PREFIX):
            server_name = self._server_name_for_tag(tag_name)

            self._ensure_server(server_name)

            if self.servers[server_name].health_check_command is not None:
                logger.error(
                    f"DUPLICATE HEALTH CHECK COMMAND for server '{server_name}'"
                )
                logger.error(
                    f"  First: {self.servers[server_name].health_check_command.file_path}"
                )
                logger.error(f"  Second: {command.file_path}")
                self.servers[server_name].command_conflicts.append("health-check")
                return

            self.servers[server_name].health_check_command = command

        elif tag_name.startswith(AIPERF_RUN_TAG_PREFIX):
            server_name = self._server_name_for_tag(tag_name)

            self._ensure_server(server_name)

            self.servers[server_name].aiperf_commands.append(command)
