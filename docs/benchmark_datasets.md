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

This document covers the various datasets AIPerf can leverage to generate stimulus. Additional support is in development so check back often.

## Dataset Options

<table style="width:100%; border-collapse: collapse;">
  <thead>
    <tr>
      <th style="width:15%; text-align: left;">Dataset</th>
      <th style="width:10%; text-align: center;">Support</th>
      <th style="width:65%; text-align: left;">Data Source</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Synthetic Text</strong></td>
      <td style="text-align: center;">✅</td>
      <td>Synthetically generated text prompts pulled from Shakespeare</td>
    </tr>
    <tr>
      <td><strong>Synthetic Audio</strong></td>
      <td style="text-align: center;">✅</td>
      <td>Synthetically generated audio samples</td>
    </tr>
    <tr>
      <td><strong>Synthetic Images</strong></td>
      <td style="text-align: center;">✅</td>
      <td>Synthetically generated image samples</td>
    </tr>
    <tr>
      <td><strong>Custom Data</strong></td>
      <td style="text-align: center;">✅</td>
      <td>--input-file your_file.jsonl --custom-dataset-type single-turn</td>
    </tr>
    <tr>
    <td><strong>Mooncake</strong></td>
    <td style="text-align: center;">✅</td>
    <td><a href="benchmark_modes/trace_replay.md"><code>--input-file your_trace_file.jsonl --custom-dataset-type mooncake_trace</code></a></td>
    </tr>
    <tr>
      <td><strong>ShareGPT</strong></td>
      <td style="text-align: center;">✅</td>
      <td><code>--public-dataset sharegpt</code></td>
    </tr>
    <tr>
      <td><strong>BurstGPT</strong></td>
      <td style="text-align: center;">Coming Soon</td>
      <td></td>
    </tr>
    <tr>
      <td><strong>HuggingFace-VisionArena</strong></td>
      <td style="text-align: center;">Coming Soon</td>
      <td></td>
    </tr>
    <tr>
      <td><strong>HuggingFace-AIMO</strong></td>
      <td style="text-align: center;">Coming Soon</td>
      <td></td>
    </tr>
  </tbody>
</table>

