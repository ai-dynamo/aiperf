import sys

from aiperf.common.config.service_config import ServiceConfig
from aiperf.common.enums import Topic
from aiperf.common.models.messages import BaseMessage
from aiperf.common.service import ServiceBase


class RecordsManager(ServiceBase):
    def __init__(self, config: ServiceConfig) -> None:
        super().__init__(service_type="records_manager", config=config)

    async def _initialize(self) -> None:
        self.logger.debug("Initializing records manager")
        # TODO: Implement records manager initialization

    async def _on_start(self) -> None:
        self.logger.debug("Starting records manager")
        # TODO: Implement records manager start

    async def _on_stop(self) -> None:
        self.logger.debug("Stopping records manager")
        # TODO: Implement records manager stop

    async def _cleanup(self) -> None:
        self.logger.debug("Cleaning up records manager")
        # TODO: Implement records manager cleanup

    async def _process_message(self, topic: Topic, message: BaseMessage) -> None:
        self.logger.debug(f"Processing message in records manager: {topic}, {message}")
        # TODO: Implement records manager message processing


def main() -> None:
    from aiperf.common.bootstrap import bootstrap_and_run_service

    bootstrap_and_run_service(RecordsManager)


if __name__ == "__main__":
    sys.exit(main())
