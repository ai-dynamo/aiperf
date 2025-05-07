import sys

from aiperf.common.config.service_config import ServiceConfig
from aiperf.common.enums import Topic
from aiperf.common.models.messages import BaseMessage
from aiperf.common.service import ServiceBase


class WorkerManager(ServiceBase):
    def __init__(self, config: ServiceConfig):
        super().__init__(service_type="worker_manager", config=config)

    async def _initialize(self) -> None:
        self.logger.debug("Initializing worker manager")
        # TODO: Implement worker manager initialization

    async def _on_start(self) -> None:
        self.logger.debug("Starting worker manager")
        # TODO: Implement worker manager start

    async def _on_stop(self) -> None:
        self.logger.debug("Stopping worker manager")
        # TODO: Implement worker manager stop

    async def _cleanup(self) -> None:
        self.logger.debug("Cleaning up worker manager")
        # TODO: Implement worker manager cleanup

    async def _process_message(self, topic: Topic, message: BaseMessage) -> None:
        self.logger.debug(f"Processing message in worker manager: {topic}, {message}")
        # TODO: Implement worker manager message processing


def main() -> None:
    from aiperf.common.bootstrap import bootstrap_and_run_service

    bootstrap_and_run_service(WorkerManager)


if __name__ == "__main__":
    sys.exit(main())
