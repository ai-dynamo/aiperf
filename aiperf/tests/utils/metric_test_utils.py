#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0


# Minimal mock classes to simulate Record, Request, and Response
class MockRequest:
    def __init__(self, timestamp):
        self.timestamp = timestamp


class MockResponse:
    def __init__(self, timestamp):
        self.timestamp = timestamp


class MockRecord:
    def __init__(self, request, responses):
        self.request = request
        self.responses = responses
