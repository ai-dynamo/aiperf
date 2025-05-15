#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import sys
from typing import Any

from aiperf.common.config.service_config import WorkerConfig


class Worker:
    """Worker responsible for sending requests to the server."""

    def __init__(self, config: WorkerConfig):
        self.config = config

    async def start(self) -> None:
        """Start the worker."""
        # TODO: Implement worker start

    async def stop(self) -> None:
        """Stop the worker."""
        # TODO: Implement worker stop

    async def process_credit(self, credit_data: dict[str, Any]) -> None:
        """Process a credit by initiating a conversation."""
        # TODO: Implement worker process credit

    async def handle_conversation(self, conversation_data: dict[str, Any]) -> None:
        """Handle a conversation with the server."""
        # TODO: Implement worker handle conversation

    async def send_request(self, request_data: dict[str, Any]) -> dict[str, Any]:
        """Send a request to the server."""
        # TODO: Implement worker send request

    async def publish_result(self, result_data: dict[str, Any]) -> None:
        """Publish a result to the worker manager."""
        # TODO: Implement worker publish result


if __name__ == "__main__":
    import uvloop

    uvloop.install()

    # Load the service configuration
    from aiperf.common.config.loader import load_worker_config

    cfg = load_worker_config()
    worker = Worker(cfg)
    sys.exit(uvloop.run(worker.run()))
