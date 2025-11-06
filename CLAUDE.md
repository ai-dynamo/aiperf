<!--
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# AIPerf Dev Guide

Python 3.10+ async AI Benchmarking Tool. 8 services via ZMQ. Update this guide only for major architectural shifts.

**Principles**: KISS + DRY. Extensibility + Usability + Accuracy + Scalability. One concern per PR. Review own diff first.

## Patterns

**Service** (stateless, separate process, bootstrap.py):
```python
from aiperf.common.base_component_service import BaseComponentService
from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.factories import ServiceFactory
from aiperf.common.enums import ServiceType

@ServiceFactory.register(ServiceType.MY_SERVICE)
class MyService(BaseComponentService):
    def __init__(self, service_config: ServiceConfig, user_config: UserConfig,
                 service_id: str | None = None, **kwargs) -> None:
        super().__init__(service_config, user_config, service_id, **kwargs)

    @on_message(MessageType.MY_MSG)
    async def _handle_my_msg(self, msg: MyMsg) -> None:
        await self.publish(ResponseMsg(data=msg.data))
```
ServiceConfig: infrastructure (ZMQ, logging). UserConfig: benchmark params (endpoints, loadgen).

**Models**: AIPerfBaseModel=data, BaseConfig=config, Field(description=...) always
```python
from pydantic import Field
from aiperf.common.models import AIPerfBaseModel
from aiperf.common.config import BaseConfig

class Record(AIPerfBaseModel):
    ts_ns: int = Field(description="Timestamp")
```

**Messages**: Set `message_type`, subscribe with `@on_message(MessageType.X)`. Framework auto-subscribes during `@on_init` by introspecting decorators.
```python
from aiperf.common.messages import Message
from aiperf.common.hooks import on_message

class MyMsg(Message):
    message_type: MessageType = MessageType.MY_MSG
    data: list[Record] = Field(description="Records")

@on_message(MessageType.MY_MSG)
async def _handle(self, msg: MyMsg) -> None:
    await self.publish(OtherMsg(data=msg.data))
```

**Protocol+Factory** (extensibility): Protocol `protocols.py` → Factory `factories.py` → Enum `plugin_enums.py` → `@Factory.register(Enum.TYPE)`
```python
# Registration
@EndpointFactory.register(EndpointType.OPENAI)
class OpenAIEndpoint(EndpointProtocol):
    def __init__(self, model_endpoint: ModelEndpoint) -> None: ...

# Usage
endpoint = EndpointFactory.create_instance(
    EndpointType.OPENAI,
    model_endpoint=model_endpoint,
)
```

**Error Handling**: Log with `self.error()`, publish `ErrorDetails` models in messages. Catch exceptions.
```python
try:
    await risky_operation()
except Exception as e:
    self.error(f"Operation failed: {e!r}")
    await self.publish(ResultMsg(error=ErrorDetails.from_exception(e)))
```

**Logging**: Lambda for expensive logs: `self.debug(lambda: f"{len(self._x())}")`. Direct string: `self.info("Starting")`.

**JSON**: Always orjson: `orjson.loads(s)`, `orjson.dumps(d)`

## Mixins & Base Classes

**Base Classes:**
- **BaseComponentService**: Services only. Includes lifecycle, message bus, commands, health. Use for all 8 services.
- **BaseService**: Abstract base for services. Use BaseComponentService instead unless creating new service type.

**Core Mixins** (usually via base class):
- **AIPerfLifecycleMixin**: State machine (`CREATED`→`INITIALIZING`→`INITIALIZED`→`STARTING`→`RUNNING`→`STOPPING`→`STOPPED`; `FAILED` is an additional terminal state). Provides `initialize()`, `start()`, `stop()`. Components may enter `FAILED` via `_fail()` when transitions error. Use for components with lifecycle.
- **HooksMixin**: Hook registration. Enables `@on_init`, `@on_start`, `@on_stop`, etc.
- **TaskManagerMixin**: Async task management. Provides `execute_async()`, `start_background_task()`. Auto-cancels tasks on stop.
- **AIPerfLoggerMixin**: Logging. Provides `self.debug()`, `self.info()`, `self.warning()`, `self.error()`.

**Communication Mixins:**
- **MessageBusClientMixin**: Pub/Sub. Provides `publish()`, `subscribe()`, `@on_message`. For async broadcast.
- **CommandHandlerMixin**: Command pattern. Provides `@on_command`, `send_command_and_wait_for_response()`. For control messages.
- **PullClientMixin**: Pull pattern. Provides `@on_pull_message`, creates pull client. For high-throughput streams.
- **ReplyClientMixin**: Request-reply. Provides `@on_request`. For synchronous responses.

**Specialized Mixins:**
- **ProcessHealthMixin**: System metrics (CPU, memory). Use for services needing health reporting.
- **BufferedJSONLWriterMixin**: Buffered JSONL file writer. Use for high-volume file output.
- **ProgressTrackerMixin**: UI progress tracking. Use for UI components subscribing to progress updates.

**Usage:** BaseComponentService includes everything. Other components compose mixins as needed.
```python
# Component with lifecycle + message bus
class MyComponent(MessageBusClientMixin):
    @on_init
    async def _setup(self) -> None:
        self._client = aiohttp.ClientSession()

    @on_stop
    async def _cleanup(self) -> None:
        await self._client.close()

    @on_message(MessageType.DATA)
    async def _handle(self, msg: DataMsg) -> None:
        await self.publish(ResponseMsg(data=msg.data))
```

## Anti-Patterns
One PR=one goal | Comments only for "why?" not "what" | No persistent mutable state | No blocking I/O

## Services
**SystemController**: orchestration, lifecycle management
**DatasetManager**: prompt/token generation
**TimingManager**: request scheduling, credit issuance
**WorkerManager**: worker lifecycle, health monitoring
**Worker** (N): LLM API calls, conversation state
**RecordProcessor** (N): metric computation, scales with load
**RecordsManager**: record aggregation
**TelemetryManager**: GPU telemetry from DCGM

Communication: ZMQ message bus via `await self.publish(msg)`. Services auto-subscribe based on `@on_message` decorators during `@on_init`.

## Testing
pytest. Auto-fixtures: time mocked, RNG=42, singletons reset. Use fixtures+helpers+parametrization: `@pytest.mark.parametrize("x,y", [(1,2)])`. put import statements at the top of the test file. Use `# fmt: skip` for long parameterize blocks.

## Python 3.10
`|` unions: `str | int | None`
`match/case`: `match x: case A: ...; case _: ...`
`@dataclass(slots=True)`

## Pre-Commit
1. Diff: all lines required? 2. `ruff format . && ruff check --fix .` 3. `pytest` 4. Type hints 5. `Field(description=...)` 6. `pytest -m integration` 7. `pre-commit run` 8. `git commit -s`

## Common Tasks
**Service**: BaseComponentService → enum `service_enums.py` → `@ServiceFactory.register()`
**Message**: Enum `message_enums.py` → class `messages/` → `@on_message()`
**Factory**: Protocol `protocols.py` → Factory `factories.py` → Enum `plugin_enums.py` → `@Factory.register()`

## Rules
1. BaseComponentService for services; AIPerfLifecycleMixin for components only
2. AIPerfBaseModel for data models; BaseConfig for configuration classes
3. Field(description="...") always for Pydantic fields
4. async/await for all I/O (no time.sleep, no blocking calls)
5. Message bus for communication (publish/push/pull, request-reply for sync operations)
6. Lambda for expensive logs: `self.debug(lambda: f"{self._x()}")`
7. No persistent mutable state in services (instance variables for config/deps are fine)
8. Type hints required on all functions (params and return)
9. Protocol + Factory + Enum for all extensible features
10. Tests use fixtures, helpers, and @pytest.mark.parametrize
11. KISS + DRY - optimize for reader

**Build systems that scale. Write code that lasts.**
