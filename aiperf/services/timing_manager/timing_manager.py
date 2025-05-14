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

from aiperf.common.config.service_config import ServiceConfig
from aiperf.common.enums import Topic
from aiperf.common.models.messages import BaseMessage
from aiperf.common.service import BaseService


class TimingManager(BaseService):
    def __init__(self, config: ServiceConfig):
        super().__init__(service_type="timing_manager", config=config)

    async def _initialize(self) -> None:
        self.logger.debug("Initializing timing manager")
        # TODO: Implement timing manager initialization

    async def _on_start(self) -> None:
        self.logger.debug("Starting timing manager")
        # TODO: Implement timing manager start

    async def _on_stop(self) -> None:
        self.logger.debug("Stopping timing manager")
        # TODO: Implement timing manager stop

    async def _cleanup(self) -> None:
        self.logger.debug("Cleaning up timing manager")
        # TODO: Implement timing manager cleanup

    async def _process_message(self, topic: Topic, message: BaseMessage) -> None:
        self.logger.debug(f"Processing message in timing manager: {topic}, {message}")
        # TODO: Implement timing manager message processing


def main() -> None:
    from aiperf.common.bootstrap import bootstrap_and_run_service

    bootstrap_and_run_service(TimingManager)


if __name__ == "__main__":
    sys.exit(main())
