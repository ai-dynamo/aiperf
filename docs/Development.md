<!--
#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
-->
# Developers Guide
Execute the following commands to set up your development environment for `aiperf`.
Make sure you are in the root directory of the `aiperf` repository.

## Development Environment
- Install uv
https://docs.astral.sh/uv/getting-started/installation/
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

- Create virtual env
```bash
uv venv
```

- Activate venv
```bash
source .venv/bin/activate
```

- Install `aiperf` package in editable development mode
```bash
uv pip install -e ".[dev]"
```

- Run `aiperf` in asyncio mode (current default)
```bash
aiperf --run-type async
```
Press `Ctrl-C` to stop the process

- Run `aiperf` in multiprocessing mode
```bash
aiperf --run-type process
```
Press `Ctrl-C` to stop the process

- Run `aiperf` with `--help` to see available commands
```bash
aiperf --help
```

## Code Overview

### Project Structure

```
aiperf/
├── aiperf/              # Main python package
│   ├── app/                 # Application components
│   ├── cli.py               # Command line interface
│   ├── common/              # Shared utilities and models
│   │   ├── config/             # Configuration management
│   │   ├── enums.py            # Enum definitions
│   │   ├── exceptions/         # Custom exceptions
│   │   ├── models/             # Pydantic data models
│   │   │   ├── messages.py         # Message definitions
│   │   │   └── service.py          # Service related model definitions
│   │   └── service.py          # Base service implementation
│   └── services/            # System services
│       ├── dataset_manager/        # Handles dataset operations
│       ├── post_processor_manager/ # Processes results
│       ├── records_manager/        # Manages test records
│       ├── system_controller/      # Controls system operation
│       ├── timing_manager/         # Handles timing and credits
│       ├── worker/                 # Executes benchmarks
│       └── worker_manager/         # Manages worker processes
├── docs/                # Documentation
├── tests/               # Test suite
├── pyproject.toml       # Project configuration
├── Makefile             # Build and automation scripts
└── README.md            # Project readme
```

### Core Components
> This comes from the AIPerf Design Document

AIPerf implements a distributed microservices architecture with the following key components:

- **System Controller**: Primary responsibility is to orchestrate the system. It will ensure all blocks are ready and healthy. It will also help orchestrating graceful shutdowns. This is the component that will contain the methods users can interact with.
- **Dataset Manager**: Primary responsibility is to manage the data: generation or acquisition. For  synthetic generation, it contains the code to generate the prompts or tokens. It will have an API for dataset acquisition of a dataset if available in a remote repository or database.

- **Worker Manager**: Primary responsibility is to pull data from the dataset manager after receiving the timing credit from the timing manager. It will then push the request data to the worker to issue to the request.
- **Worker**: Primarily responsible for converting the data into the appropriate format for the interface being used by the server. Also responsible for managing the conversation between turns.

- **Timing Manager**: Primary responsibility is to generate the schedule and issuing timing credits for requests.

- **Records Manager**: Primarily responsible for holding the results returned from the workers.

- **Post-Processor Manager**: Primarily responsible for iterating over the records to generate metrics and other conclusions from the records.


#### Communication System

Services communicate using a message-based system with the following components:

- **Topics**: Categorized channels for message distribution (commands, status, data, etc.)
- **Messages**: Strongly-typed data structures for inter-service communication
- **Service States**: Lifecycle states that services transition through (initializing, running, stopping, etc.)

#### Command Line Interface

The CLI (`aiperf/cli.py`) provides the entry point to the system


#### Message Processing

To handle messages:

1. Subscribe to relevant topics using `_subscribe_to_topic(Topic)`
2. Implement message processing logic in `_process_message(topic, message)`
3. Send messages using `_publish_message(topic, message)`


### Service Inheritance Model

AIPerf uses an inheritance-based architecture where all system services inherit from a common `ServiceBase` abstract class. This approach provides a consistent interface and shared functionality across services.

#### Base Service Responsibilities

The `ServiceBase` class (`aiperf/common/service.py`) provides automatically for all services:

- **Lifecycle Management**: Standard initialize/run/stop/cleanup methods
- **State Transitions**: Manages service state changes (INITIALIZING → RUNNING → STOPPING → STOPPED)
- **Communication**: Methods for publishing messages and subscribing to topics
- **Heartbeat**: Automatic heartbeat generation for service health monitoring

#### Service Implementation Requirements

When implementing a new service that inherits from `ServiceBase`, you must:

1. **Implement Abstract Methods**:
   - `_initialize()`: Set up service-specific resources
   - `_on_start()`: Main service logic
   - `_on_stop()`: Handle graceful shutdown
   - `_cleanup()`: Release resources
   - `_process_message()`: Handle incoming messages

2. **Configuration**:
   - Define service-specific configuration needs
   - Pass configuration to the base class constructor

#### Example Service Implementation

Here's a simplified example of a service implementation:

```python
from aiperf.common.service.base_service import BaseService
from aiperf.common.config.service_config import ServiceConfig
from aiperf.common.enums import Topic, ClientType
from aiperf.common.models import Message


class ExampleService(BaseService):
    def __init__(self, config: ServiceConfig) -> None:
        super().__init__(service_type="example_service", service_config=config)
        self.my_resource = None

    async def _initialize(self) -> None:
        """Initialize service-specific resources."""
        self.logger.debug("Initializing Example Service")
        # Subscribe to required topics
        # TODO: Fix this documentation
        await self._subscribe_to_topic(ClientType.CONTROLLER_SUB, Topic.COMMAND)
        await self._subscribe_to_topic(ClientType.INFERENCE_REQUEST_SUB, Topic.DATA)
        # Initialize resources
        self.my_resource = SomeResource()

    async def _on_start(self) -> None:
        """Main service logic."""
        self.logger.debug("Running Example Service")
        # Implement your service's main logic here
        # This method should typically set up ongoing tasks or loops

    async def _on_stop(self) -> None:
        """Handle graceful shutdown."""
        self.logger.debug("Stopping Example Service")
        # Cancel any ongoing tasks
        # Prepare for cleanup

    async def _cleanup(self) -> None:
        """Release resources."""
        self.logger.debug("Cleaning up Example Service")
        # Release any resources
        if self.my_resource:
            await self.my_resource.close()

    async def _process_message(self, topic: Topic, message: Message) -> None:
        """Handle incoming messages."""
        self.logger.debug(f"Processing response: {topic}, {message}")
        if topic == Topic.COMMAND:
            # Handle command messages
            await self._handle_command(message)
        elif topic == Topic.DATA:
            # Handle data messages
            await self._handle_data(message)

    async def _handle_command(self, message: Message) -> None:
        """Handle command messages."""
        # Implement command handling logic

    async def _handle_data(self, message: Message) -> None:
        """Handle data messages."""
        # Implement data handling logic
```

#### Using the Service

To instantiate and run a service:

```python
def main() -> None:
    from aiperf.common.bootstrap import bootstrap_and_run_service

    bootstrap_and_run_service(ExampleService)


if __name__ == "__main__":
    sys.exit(main())
```

The `bootstrap_and_run_service` function handles:
1. Setting up the event loop (using uvloop)
2. Creating an instance of your service
3. Running the service's lifecycle methods
4. Handling graceful shutdown

This inheritance model ensures consistent behavior across all services while allowing for service-specific customization.

