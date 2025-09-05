<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

This document describes datasets that AIPerf can use to generate stimulus. Additional support is under development, so check back often.

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
    <td>Mooncake trace file <a href="benchmark_modes/trace_replay.md"><code>--input-file your_trace_file.jsonl --custom-dataset-type mooncake_trace</code></a></td>
    </tr>
    <tr>
      <td><strong>ShareGPT</strong></td>
      <td style="text-align: center;">✅</td>
      <td>Conversations from <a href="https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"><code>--public-dataset sharegpt</code></td>
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

