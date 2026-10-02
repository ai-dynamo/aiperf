# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import logging

from aiperf.common.noisy_loggers import suppress_noisy_http_loggers


def test_suppress_raises_noisy_loggers_to_warning():
    logging.getLogger("aiohttp.access").setLevel(logging.DEBUG)
    suppress_noisy_http_loggers()
    assert logging.getLogger("aiohttp.access").level == logging.WARNING
    assert logging.getLogger("kubernetes_asyncio.client.rest").level == logging.WARNING
