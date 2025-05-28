<!--
#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
-->
```mermaid
classDiagram
    direction LR

    %% Abstract classes
    class AbstractBaseService {
        <<abstract>>
    }

    %% Concrete base classes
    class BaseService {
        <<abstract>>
    }

    %% Specialized service types
    class BaseComponentService {
        <<abstract>>
    }

    class BaseControllerService {
        <<abstract>>
    }

    %% Concrete service implementations
    class SystemController {
    }

    class Worker {
    }

    class WorkerManager {
    }

    class DatasetManager {
    }

    class RecordsManager {
    }

    class TimingManager {
    }

    class PostProcessorManager {
    }

    %% Service management classes
    class BaseServiceManager {
        <<abstract>>
    }

    class MultiProcessServiceManager {
    }

    class KubernetesServiceManager {
    }

    %% Relationships
    AbstractBaseService <|-- BaseService
    BaseService <|-- BaseComponentService
    BaseService <|-- BaseControllerService
    BaseService <|-- Worker
    BaseControllerService <|-- SystemController
    BaseComponentService <|-- WorkerManager
    BaseComponentService <|-- DatasetManager
    BaseComponentService <|-- RecordsManager
    BaseComponentService <|-- TimingManager
    BaseComponentService <|-- PostProcessorManager


    SystemController ..> BaseServiceManager: uses
    BaseServiceManager <|-- MultiProcessServiceManager
    BaseServiceManager <|-- KubernetesServiceManager
    WorkerManager --|> Worker: spawns
```
