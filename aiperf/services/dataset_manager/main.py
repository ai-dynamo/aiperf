import sys

from aiperf.common.config.service_config import ServiceConfig
from aiperf.common.enums import Topic
from aiperf.common.models.messages import BaseMessage
from aiperf.common.service import ServiceBase


class DatasetManager(ServiceBase):
    def __init__(self, config: ServiceConfig):
        super().__init__(service_type="dataset_manager", config=config)

    async def _initialize(self) -> None:
        self.logger.debug("Initializing dataset manager")
        # TODO: Implement dataset manager initialization

    async def _on_start(self) -> None:
        self.logger.debug("Starting dataset manager")
        # TODO: Implement dataset manager start

    async def _on_stop(self) -> None:
        self.logger.debug("Stopping dataset manager")
        # TODO: Implement dataset manager stop

    async def _cleanup(self) -> None:
        self.logger.debug("Cleaning up dataset manager")
        # TODO: Implement dataset manager cleanup

    async def _process_message(self, topic: Topic, message: BaseMessage) -> None:
        self.logger.debug(f"Processing message in dataset manager: {topic}, {message}")
        # TODO: Implement dataset manager message processing


def main() -> None:
    from aiperf.common.bootstrap import bootstrap_and_run_service

    bootstrap_and_run_service(DatasetManager)


if __name__ == "__main__":
    sys.exit(main())
