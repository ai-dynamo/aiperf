# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Shared test configuration and fixtures for all test types.

ONLY ADD FIXTURES HERE THAT ARE USED IN ALL TEST TYPES.
DO NOT ADD FIXTURES THAT ARE ONLY USED IN A SPECIFIC TEST TYPE.
"""

# Suppress factory override warnings before any imports trigger registration
import logging

for _factory_logger in [
    "CommunicationFactory",
    "ServiceManagerFactory",
    "TransportFactory",
    "ZMQProxyFactory",
]:
    logging.getLogger(_factory_logger).setLevel(logging.ERROR)
