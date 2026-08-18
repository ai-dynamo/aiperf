# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``mp-context`` ergonomics check.

The check exists because of a shipped-and-reproduced bug: the global log queue
was built with a bare ``multiprocessing.Queue()`` (default start method, ``fork``
on Linux) while service processes were spawned from ``get_mp_context()``
(``forkserver`` on Linux). Handing the queue to a child then raised "A SemLock
created in a fork context is being shared with a process in a spawn context".

That failure is invisible to CI: the queue is only created when the dashboard UI
is active, and the dashboard downgrades itself on a non-TTY, so every pipe-captured
run passes. A static check is the only place it can be caught.
"""

import ast

import pytest

from tools.check_ergonomics import check_mp_context


def _violations(source: str, rel: str = "src/aiperf/common/example.py") -> list:
    return check_mp_context(ast.parse(source), rel)


class TestDefaultContextIsRejected:
    """The bare constructors that bind a primitive to the default start method."""

    def test_module_attribute_call_is_flagged(self) -> None:
        source = "import multiprocessing\nq = multiprocessing.Queue(maxsize=1)\n"
        violations = _violations(source)
        assert len(violations) == 1
        assert violations[0].check == "mp-context"
        assert "get_mp_context().Queue(...)" in violations[0].message

    def test_aliased_module_import_is_flagged(self) -> None:
        source = "import multiprocessing as mp\ne = mp.Event()\n"
        assert len(_violations(source)) == 1

    def test_from_import_is_flagged(self) -> None:
        source = "from multiprocessing import Queue\nq = Queue()\n"
        assert len(_violations(source)) == 1

    def test_aliased_from_import_is_flagged(self) -> None:
        source = "from multiprocessing import Queue as Q\nq = Q()\n"
        violations = _violations(source)
        assert len(violations) == 1
        # The identifier records the real primitive, not the local alias, so a
        # baseline entry cannot be dodged by renaming the import.
        assert violations[0].identifier.endswith("::Queue")

    @pytest.mark.parametrize(
        "primitive",
        ["Queue", "Process", "Event", "Lock", "Value", "SimpleQueue"],
    )  # fmt: skip
    def test_each_context_bound_primitive_is_flagged(self, primitive: str) -> None:
        source = f"import multiprocessing\nx = multiprocessing.{primitive}()\n"
        assert len(_violations(source)) == 1

    def test_violation_is_attributed_to_its_enclosing_function(self) -> None:
        source = (
            "import multiprocessing\n"
            "def build_queue():\n"
            "    return multiprocessing.Queue()\n"
        )
        assert _violations(source)[0].identifier == "build_queue::Queue"


class TestSanctionedFormsAreAccepted:
    """The context-object spellings the check must not fire on."""

    def test_get_mp_context_call_is_not_flagged(self) -> None:
        source = (
            "from aiperf.common.mp_context import get_mp_context\n"
            "q = get_mp_context().Queue(maxsize=1)\n"
        )
        assert _violations(source) == []

    def test_context_variable_is_not_flagged(self) -> None:
        source = "ctx = get_mp_context()\nq = ctx.Queue()\np = ctx.Process()\n"
        assert _violations(source) == []

    def test_unrelated_queue_class_is_not_flagged(self) -> None:
        source = "import asyncio\nq = asyncio.Queue()\n"
        assert _violations(source) == []

    def test_mp_context_module_itself_is_exempt(self) -> None:
        source = "import multiprocessing\nctx = multiprocessing.get_context('spawn')\n"
        assert _violations(source, rel="src/aiperf/common/mp_context.py") == []


class TestRegressionGuard:
    def test_the_original_log_queue_bug_is_caught(self) -> None:
        """The exact shape that broke every interactive Linux run."""
        source = (
            "import multiprocessing\n"
            "def get_global_log_queue():\n"
            "    global _global_log_queue\n"
            "    if _global_log_queue is None:\n"
            "        _global_log_queue = multiprocessing.Queue(maxsize=1000)\n"
            "    return _global_log_queue\n"
        )
        violations = _violations(source, rel="src/aiperf/common/logging.py")
        assert len(violations) == 1
        assert violations[0].identifier == "get_global_log_queue::Queue"
