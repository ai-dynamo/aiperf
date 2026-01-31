# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""URL sampling strategies for multi-URL load balancing.

Provides URL selection strategies for distributing requests across multiple
endpoint URLs when multiple `--url` values are provided.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class URLSelectionStrategyProtocol(Protocol):
    """Protocol for URL selection strategies.

    Used for load balancing across multiple endpoint URLs when multiple
    `--url` values are provided. Any class implementing this protocol must
    provide a `next_url_index` method that returns the next URL index.

    Thread Safety:
        Safe for asyncio single-threaded concurrency.
    """

    def __init__(self, urls: list[str], **kwargs) -> None: ...

    def next_url_index(self) -> int:
        """Return the index of the next URL to use.

        Returns:
            Index into the urls list (0 to len(urls)-1)
        """
        ...

    @property
    def urls(self) -> list[str]:
        """The list of URLs being sampled."""
        ...


class RoundRobinURLSampler:
    """Round-robin URL sampler for even distribution across endpoints.

    Distributes requests evenly across URLs in sequential order. Each call to
    `next_url_index()` returns the next URL index in the list, wrapping around
    when the end is reached.

    Thread Safety:
        Safe for asyncio single-threaded concurrency.
    """

    def __init__(self, urls: list[str], **kwargs) -> None:
        """Initialize with list of URLs.

        Args:
            urls: List of endpoint URLs to distribute across.
            **kwargs: Ignored additional arguments for factory compatibility.

        Raises:
            ValueError: If urls list is empty.
        """
        if not urls:
            raise ValueError("URLs list cannot be empty")
        self._urls = urls
        self._index = 0

    @property
    def urls(self) -> list[str]:
        """The list of URLs being sampled."""
        return self._urls

    def next_url_index(self) -> int:
        """Return the index of the next URL in round-robin order.

        Returns:
            Index into the urls list (0 to len(urls)-1).
        """
        index = self._index
        self._index = (self._index + 1) % len(self._urls)
        return index
