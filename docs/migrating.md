<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Migrating from GenAI-Perf

AIPerf is designed to be a near drop in replacement for GenAI-Perf. There are only a few exceptions that will impact your installation and usage.
<br>

## Known Option Differences

- --max-threads: AIPerf will auto scale to deliver the workload requested. Setting a max-thread option is no longer necessary.
- "--": The passthrough args flag is no longer required. All options are now natively supported by AIPerf.

Removing the above options should be all that is required to have your previous GenAI-Perf commands work in AIPerf.

<br>

## Installation

Installation is now a single pip command:
```
pip install git+https://github.com/ai-dynamo/aiperf.git
```
<br>

---

With these simple updates to your previous scripts, AIPerf can replace your usage of GenAI-Perf.