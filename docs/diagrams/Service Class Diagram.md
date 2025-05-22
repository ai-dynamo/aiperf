<!--
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
