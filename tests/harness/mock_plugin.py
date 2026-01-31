# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test utility for temporarily registering mock plugins."""

from __future__ import annotations

from typing import Any

from aiperf.plugin import plugins
from aiperf.plugin.types import PluginEntry  # Used in type hints


class mock_plugin:
    """Context manager for temporarily registering mock plugins in tests.

    Registers a plugin on entry and removes it on exit. Useful for testing
    plugin-dependent code without modifying the global registry permanently.

    Args:
        category: Plugin category to register under.
        name: Plugin name.
        cls: The class to register.
        priority: Optional priority (default: 100 to override built-ins).
        metadata: Optional metadata dict.

    Example:
        >>> from tests.harness import mock_plugin
        >>> class MockEndpoint:
        ...     pass
        ...
        >>> with mock_plugin("endpoint", "test-endpoint", MockEndpoint):
        ...     endpoint_cls = plugins.get_class("endpoint", "test-endpoint")
        ...     assert endpoint_cls is MockEndpoint
        >>> # Plugin is removed after context exits
    """

    def __init__(
        self,
        category: str,
        name: str,
        cls: type,
        *,
        priority: int = 100,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.category = str(category)
        self.name = name
        self.cls = cls
        self.priority = priority
        self.metadata = metadata or {}
        self._previous_entry: PluginEntry | None = None

    def __enter__(self) -> PluginEntry:
        # Save existing entry if present (for restoration)
        if plugins.has_entry(self.category, self.name):
            self._previous_entry = plugins.get_entry(self.category, self.name)

        # Register the mock plugin with optional metadata
        plugins.register(
            self.category,
            self.name,
            self.cls,
            priority=self.priority,
            metadata=self.metadata if self.metadata else None,
        )

        return plugins.get_entry(self.category, self.name)

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        # Use the public unregister API, optionally restoring the previous entry
        plugins.unregister(
            self.category,
            self.name,
            restore_entry=self._previous_entry,
        )
