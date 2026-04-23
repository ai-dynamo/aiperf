# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers for downloading HTTP(S) image URLs and inlining them as base64 data URLs."""

from __future__ import annotations

import asyncio
from io import BytesIO
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import aiohttp
from PIL import Image as PILImage

from aiperf.common.enums import ImageFormat
from aiperf.common.environment import Environment
from aiperf.dataset.utils import encode_image
from aiperf.transports.aiohttp_client import create_tcp_connector
from aiperf.transports.http_defaults import AioHttpDefaults

if TYPE_CHECKING:
    from collections.abc import Iterable

    from aiperf.common.models import Conversation

URLLocations = dict[str, list[tuple[list[str], int]]]


def collect_http_image_urls(conversations: Iterable[Conversation]) -> URLLocations:
    """Walk conversations and return URL -> list of (contents_list, index) slots."""
    url_to_locations: URLLocations = {}
    for conversation in conversations:
        for turn in conversation.turns:
            for image in turn.images:
                for i, content in enumerate(image.contents):
                    parsed = urlparse(content)
                    if parsed.scheme in ("http", "https") and parsed.netloc:
                        url_to_locations.setdefault(content, []).append(
                            (image.contents, i)
                        )
    return url_to_locations


def _encode_as_data_url(data: bytes, url: str) -> str:
    img = PILImage.open(BytesIO(data))
    if img.format is None:
        raise RuntimeError(f"Failed to determine image format for URL '{url}'")
    if img.format.upper() not in list(ImageFormat):
        raise RuntimeError(
            f"'{img.format}' from URL '{url}' is not a supported "
            f"image format: {', '.join(ImageFormat)}"
        )
    return f"data:image/{img.format.lower()};base64,{encode_image(img, img.format)}"


async def _download_and_encode(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    timeout: aiohttp.ClientTimeout,
    url: str,
    *,
    url_to_data_url: dict[str, str],
) -> None:
    async with semaphore, session.get(url, timeout=timeout) as resp:
        if resp.status != 200:
            raise RuntimeError(
                f"Failed to download media URL '{url}': HTTP {resp.status}"
            )
        data = await resp.read()
    url_to_data_url[url] = _encode_as_data_url(data, url)


async def download_and_inline_urls(url_to_locations: URLLocations) -> None:
    """Download each URL once, then replace every occurrence with a data URL in-place."""
    dataset_env = Environment.DATASET
    timeout = aiohttp.ClientTimeout(total=dataset_env.MEDIA_DOWNLOAD_TIMEOUT)
    max_concurrency = dataset_env.MEDIA_DOWNLOAD_MAX_CONCURRENCY
    semaphore = asyncio.Semaphore(max_concurrency)
    url_to_data_url: dict[str, str] = {}

    connector = create_tcp_connector()
    async with aiohttp.ClientSession(
        connector=connector,
        trust_env=AioHttpDefaults.TRUST_ENV,
    ) as session:
        await asyncio.gather(
            *[
                _download_and_encode(
                    session,
                    semaphore,
                    timeout,
                    url,
                    url_to_data_url=url_to_data_url,
                )
                for url in url_to_locations
            ]
        )

    for url, locations in url_to_locations.items():
        data_url = url_to_data_url[url]
        for contents_list, index in locations:
            contents_list[index] = data_url
