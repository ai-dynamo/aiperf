<!--
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->

## Command Line Options

### Endpoint Options

<table>
<thead>
<tr>
<th style='white-space: nowrap; width: 20%; min-width: 150px;'>Option</th>
<th style='width: 10%; min-width: 80px;'>Type</th>
<th style='width: 70%;'>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='model-names'></a><code>-m</code><br><code>--model-names</code><br><code>--model</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'><code>list</code></td>
<td style='width: 70%; vertical-align: top;'>Model name(s) to be benchmarked. Can be a comma-separated list or a single model name.</td>
</tr>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='model-selection-strategy'></a><code>--model-selection-strategy</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'><code>str</code></td>
<td style='width: 70%; vertical-align: top;'>When multiple models are specified, this is how a specific model should be assigned to a prompt. round_robin: nth prompt in the list gets assigned to n-mod len(models). random: assignment is uniformly random. <br><strong>Choices:</strong> <code>round_robin</code>, <code>random</code> <br><strong>Default:</strong> <code>round_robin</code></td>
</tr>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='custom-endpoint'></a><code>--custom-endpoint</code><br><code>--endpoint</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'><code>str</code></td>
<td style='width: 70%; vertical-align: top;'>Set a custom endpoint that differs from the OpenAI defaults.</td>
</tr>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='endpoint-type'></a><code>--endpoint-type</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'><code>str</code></td>
<td style='width: 70%; vertical-align: top;'>The endpoint type to send requests to on the server. <br><strong>Choices:</strong> <code>chat</code>, <code>completions</code>, <code>embeddings</code>, <code>rankings</code> <br><strong>Default:</strong> <code>chat</code></td>
</tr>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='streaming'></a><code>--streaming</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'></td>
<td style='width: 70%; vertical-align: top;'>An option to enable the use of the streaming API.</td>
</tr>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='url'></a><code>-u</code><br><code>--url</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'><code>str</code></td>
<td style='width: 70%; vertical-align: top;'>URL of the endpoint to target for benchmarking. <br><strong>Default:</strong> <code>localhost:8000</code></td>
</tr>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='request-timeout-seconds'></a><code>--request-timeout-seconds</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'><code>float</code></td>
<td style='width: 70%; vertical-align: top;'>The timeout in floating-point seconds for each request to the endpoint. <br><strong>Default:</strong> <code>600.0</code></td>
</tr>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='api-key'></a><code>--api-key</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'><code>str</code></td>
<td style='width: 70%; vertical-align: top;'>The API key to use for the endpoint. If provided, it will be sent with every request as a header: <code>Authorization: Bearer <api_key></code>.</td>
</tr>
</tbody>
</table>

### Input Options

<table>
<thead>
<tr>
<th style='white-space: nowrap; width: 20%; min-width: 150px;'>Option</th>
<th style='width: 10%; min-width: 80px;'>Type</th>
<th style='width: 70%;'>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='extra-inputs'></a><code>--extra-inputs</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'><code>list</code></td>
<td style='width: 70%; vertical-align: top;'>Provide additional inputs to include with every request. Inputs should be in an 'input_name:value' format. Alternatively, a string representing a json formatted dict can be provided. <br><strong>Default:</strong> <code>[]</code></td>
</tr>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='header'></a><code>-H</code><br><code>--header</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'><code>list</code></td>
<td style='width: 70%; vertical-align: top;'>Adds a custom header to the requests. Headers must be specified as 'Header:Value' pairs. Alternatively, a string representing a json formatted dict can be provided. <br><strong>Default:</strong> <code>[]</code></td>
</tr>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='input-file'></a><code>--input-file</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'><code>str</code></td>
<td style='width: 70%; vertical-align: top;'>The file or directory path that contains the dataset to use for profiling. This parameter is used in conjunction with the <code>custom_dataset_type</code> parameter to support different types of user provided datasets.</td>
</tr>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='fixed-schedule'></a><code>--fixed-schedule</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'></td>
<td style='width: 70%; vertical-align: top;'>Specifies to run a fixed schedule of requests. This is normally inferred from the --input-file parameter, but can be set manually here.</td>
</tr>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='fixed-schedule-auto-offset'></a><code>--fixed-schedule-auto-offset</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'></td>
<td style='width: 70%; vertical-align: top;'>Specifies to automatically offset the timestamps in the fixed schedule, such that the first timestamp is considered 0, and the rest are shifted accordingly. If disabled, the timestamps will be assumed to be relative to 0.</td>
</tr>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='fixed-schedule-start-offset'></a><code>--fixed-schedule-start-offset</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'><code>int</code></td>
<td style='width: 70%; vertical-align: top;'>Specifies the offset in milliseconds to start the fixed schedule at. By default, the schedule starts at 0, but this option can be used to start at a reference point further in the schedule. This option cannot be used in conjunction with the --fixed-schedule-auto-offset. The schedule will include any requests at the start offset.</td>
</tr>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='fixed-schedule-end-offset'></a><code>--fixed-schedule-end-offset</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'><code>int</code></td>
<td style='width: 70%; vertical-align: top;'>Specifies the offset in milliseconds to end the fixed schedule at. By default, the schedule ends at the last timestamp in the trace dataset, but this option can be used to only run a subset of the trace. The schedule will include any requests at the end offset.</td>
</tr>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='public-dataset'></a><code>--public-dataset</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'><code>str</code></td>
<td style='width: 70%; vertical-align: top;'>The public dataset to use for the requests. <br><strong>Choices:</strong> <code>sharegpt</code></td>
</tr>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='custom-dataset-type'></a><code>--custom-dataset-type</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'><code>str</code></td>
<td style='width: 70%; vertical-align: top;'>The type of custom dataset to use. This parameter is used in conjunction with the --input-file parameter. [choices: single_turn, multi_turn, random_pool, mooncake_trace].</td>
</tr>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='random-seed'></a><code>--random-seed</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'><code>int</code></td>
<td style='width: 70%; vertical-align: top;'>The seed used to generate random values. Set to some value to make the synthetic data generation deterministic. It will use system default if not provided.</td>
</tr>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='goodput'></a><code>--goodput</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'><code>str</code></td>
<td style='width: 70%; vertical-align: top;'>Specify service level objectives (SLOs) for goodput as space-separated 'KEY:VALUE' pairs, where KEY is a metric tag and VALUE is a number in the metric’s display unit (falls back to its base unit if no display unit is defined). Examples: 'request_latency:250' (ms), 'inter_token_latency:10' (ms), <code>output_token_throughput_per_user:600</code> (tokens/s). Only metrics applicable to the current endpoint/config are considered. For more context on the definition of goodput, refer to DistServe paper: https://arxiv.org/pdf/2401.09670 and the blog: https://hao-ai-lab.github.io/blogs/distserve.</td>
</tr>
</tbody>
</table>

### Audio Input Options

<table>
<thead>
<tr>
<th style='white-space: nowrap; width: 20%; min-width: 150px;'>Option</th>
<th style='width: 10%; min-width: 80px;'>Type</th>
<th style='width: 70%;'>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='audio-batch-size'></a><code>--audio-batch-size</code><br><code>--batch-size-audio</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'><code>int</code></td>
<td style='width: 70%; vertical-align: top;'>The batch size of audio requests AIPerf should send. This is currently supported with the OpenAI <code>chat</code> endpoint type. <br><strong>Default:</strong> <code>1</code></td>
</tr>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='audio-length-mean'></a><code>--audio-length-mean</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'><code>float</code></td>
<td style='width: 70%; vertical-align: top;'>The mean length of the audio in seconds. <br><strong>Default:</strong> <code>0.0</code></td>
</tr>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='audio-length-stddev'></a><code>--audio-length-stddev</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'><code>float</code></td>
<td style='width: 70%; vertical-align: top;'>The standard deviation of the length of the audio in seconds. <br><strong>Default:</strong> <code>0.0</code></td>
</tr>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='audio-format'></a><code>--audio-format</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'><code>str</code></td>
<td style='width: 70%; vertical-align: top;'>The format of the audio files (wav or mp3). <br><strong>Choices:</strong> <code>wav</code>, <code>mp3</code> <br><strong>Default:</strong> <code>wav</code></td>
</tr>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='audio-depths'></a><code>--audio-depths</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'><code>list</code></td>
<td style='width: 70%; vertical-align: top;'>A list of audio bit depths to randomly select from in bits. <br><strong>Default:</strong> <code>[16]</code></td>
</tr>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='audio-sample-rates'></a><code>--audio-sample-rates</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'><code>list</code></td>
<td style='width: 70%; vertical-align: top;'>A list of audio sample rates to randomly select from in kHz. Common sample rates are 16, 44.1, 48, 96, etc. <br><strong>Default:</strong> <code>[16.0]</code></td>
</tr>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='audio-num-channels'></a><code>--audio-num-channels</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'><code>int</code></td>
<td style='width: 70%; vertical-align: top;'>The number of audio channels to use for the audio data generation. <br><strong>Default:</strong> <code>1</code></td>
</tr>
</tbody>
</table>

### Image Input Options

<table>
<thead>
<tr>
<th style='white-space: nowrap; width: 20%; min-width: 150px;'>Option</th>
<th style='width: 10%; min-width: 80px;'>Type</th>
<th style='width: 70%;'>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='image-width-mean'></a><code>--image-width-mean</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'><code>float</code></td>
<td style='width: 70%; vertical-align: top;'>The mean width of images when generating synthetic image data. <br><strong>Default:</strong> <code>0.0</code></td>
</tr>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='image-width-stddev'></a><code>--image-width-stddev</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'><code>float</code></td>
<td style='width: 70%; vertical-align: top;'>The standard deviation of width of images when generating synthetic image data. <br><strong>Default:</strong> <code>0.0</code></td>
</tr>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='image-height-mean'></a><code>--image-height-mean</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'><code>float</code></td>
<td style='width: 70%; vertical-align: top;'>The mean height of images when generating synthetic image data. <br><strong>Default:</strong> <code>0.0</code></td>
</tr>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='image-height-stddev'></a><code>--image-height-stddev</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'><code>float</code></td>
<td style='width: 70%; vertical-align: top;'>The standard deviation of height of images when generating synthetic image data. <br><strong>Default:</strong> <code>0.0</code></td>
</tr>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='image-batch-size'></a><code>--image-batch-size</code><br><code>--batch-size-image</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'><code>int</code></td>
<td style='width: 70%; vertical-align: top;'>The image batch size of the requests AIPerf should send. This is currently supported with the image retrieval endpoint type. <br><strong>Default:</strong> <code>1</code></td>
</tr>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='image-format'></a><code>--image-format</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'><code>str</code></td>
<td style='width: 70%; vertical-align: top;'>The compression format of the images. <br><strong>Choices:</strong> <code>png</code>, <code>jpeg</code>, <code>random</code> <br><strong>Default:</strong> <code>png</code></td>
</tr>
</tbody>
</table>

### Prompt Options

<table>
<thead>
<tr>
<th style='white-space: nowrap; width: 20%; min-width: 150px;'>Option</th>
<th style='width: 10%; min-width: 80px;'>Type</th>
<th style='width: 70%;'>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='prompt-batch-size'></a><code>-b</code><br><code>--prompt-batch-size</code><br><code>--batch-size-text</code><br><code>--batch-size</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'><code>int</code></td>
<td style='width: 70%; vertical-align: top;'>The batch size of text requests AIPerf should send. This is currently supported with the embeddings and rankings endpoint types. <br><strong>Default:</strong> <code>1</code></td>
</tr>
</tbody>
</table>

### Input Sequence Length (ISL) Options

<table>
<thead>
<tr>
<th style='white-space: nowrap; width: 20%; min-width: 150px;'>Option</th>
<th style='width: 10%; min-width: 80px;'>Type</th>
<th style='width: 70%;'>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='prompt-input-tokens-mean'></a><code>--prompt-input-tokens-mean</code><br><code>--synthetic-input-tokens-mean</code><br><code>--isl</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'><code>int</code></td>
<td style='width: 70%; vertical-align: top;'>The mean of number of tokens in the generated prompts when using synthetic data. <br><strong>Default:</strong> <code>550</code></td>
</tr>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='prompt-input-tokens-stddev'></a><code>--prompt-input-tokens-stddev</code><br><code>--synthetic-input-tokens-stddev</code><br><code>--isl-stddev</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'><code>float</code></td>
<td style='width: 70%; vertical-align: top;'>The standard deviation of number of tokens in the generated prompts when using synthetic data. <br><strong>Default:</strong> <code>0.0</code></td>
</tr>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='prompt-input-tokens-block-size'></a><code>--prompt-input-tokens-block-size</code><br><code>--synthetic-input-tokens-block-size</code><br><code>--isl-block-size</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'><code>int</code></td>
<td style='width: 70%; vertical-align: top;'>The block size of the prompt. <br><strong>Default:</strong> <code>512</code></td>
</tr>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='seq-dist'></a><code>--seq-dist</code><br><code>--sequence-distribution</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'><code>str</code></td>
<td style='width: 70%; vertical-align: top;'>Sequence length distribution specification for varying ISL/OSL pairs.</td>
</tr>
</tbody>
</table>

### Output Sequence Length (OSL) Options

<table>
<thead>
<tr>
<th style='white-space: nowrap; width: 20%; min-width: 150px;'>Option</th>
<th style='width: 10%; min-width: 80px;'>Type</th>
<th style='width: 70%;'>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='prompt-output-tokens-mean'></a><code>--prompt-output-tokens-mean</code><br><code>--output-tokens-mean</code><br><code>--osl</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'><code>int</code></td>
<td style='width: 70%; vertical-align: top;'>The mean number of tokens in each output.</td>
</tr>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='prompt-output-tokens-stddev'></a><code>--prompt-output-tokens-stddev</code><br><code>--output-tokens-stddev</code><br><code>--osl-stddev</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'><code>float</code></td>
<td style='width: 70%; vertical-align: top;'>The standard deviation of the number of tokens in each output. <br><strong>Default:</strong> <code>0</code></td>
</tr>
</tbody>
</table>

### Prefix Prompt Options

<table>
<thead>
<tr>
<th style='white-space: nowrap; width: 20%; min-width: 150px;'>Option</th>
<th style='width: 10%; min-width: 80px;'>Type</th>
<th style='width: 70%;'>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='prompt-prefix-pool-size'></a><code>--prompt-prefix-pool-size</code><br><code>--prefix-prompt-pool-size</code><br><code>--num-prefix-prompts</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'><code>int</code></td>
<td style='width: 70%; vertical-align: top;'>The total size of the prefix prompt pool to select prefixes from. If this value is not zero, these are prompts that are prepended to input prompts. This is useful for benchmarking models that use a K-V cache. <br><strong>Default:</strong> <code>0</code></td>
</tr>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='prompt-prefix-length'></a><code>--prompt-prefix-length</code><br><code>--prefix-prompt-length</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'><code>int</code></td>
<td style='width: 70%; vertical-align: top;'>The number of tokens in each prefix prompt. This is only used if "num" is greater than zero. Note that due to the prefix and user prompts being concatenated, the number of tokens in the final prompt may be off by one. <br><strong>Default:</strong> <code>0</code></td>
</tr>
</tbody>
</table>

### Conversation Input Options

<table>
<thead>
<tr>
<th style='white-space: nowrap; width: 20%; min-width: 150px;'>Option</th>
<th style='width: 10%; min-width: 80px;'>Type</th>
<th style='width: 70%;'>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='conversation-num'></a><code>--conversation-num</code><br><code>--num-conversations</code><br><code>--num-sessions</code><br><code>--num-dataset-entries</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'><code>int</code></td>
<td style='width: 70%; vertical-align: top;'>The total number of unique conversations to generate. Each conversation represents a single request session between client and server. Supported on synthetic mode and the custom random_pool dataset. The number of conversations  will be used to determine the number of entries in both the custom random_pool and synthetic  datasets and will be reused until benchmarking is complete. <br><strong>Default:</strong> <code>100</code></td>
</tr>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='conversation-turn-mean'></a><code>--conversation-turn-mean</code><br><code>--session-turns-mean</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'><code>int</code></td>
<td style='width: 70%; vertical-align: top;'>The mean number of turns within a conversation. <br><strong>Default:</strong> <code>1</code></td>
</tr>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='conversation-turn-stddev'></a><code>--conversation-turn-stddev</code><br><code>--session-turns-stddev</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'><code>int</code></td>
<td style='width: 70%; vertical-align: top;'>The standard deviation of the number of turns within a conversation. <br><strong>Default:</strong> <code>0</code></td>
</tr>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='conversation-turn-delay-mean'></a><code>--conversation-turn-delay-mean</code><br><code>--session-turn-delay-mean</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'><code>float</code></td>
<td style='width: 70%; vertical-align: top;'>The mean delay between turns within a conversation in milliseconds. <br><strong>Default:</strong> <code>0.0</code></td>
</tr>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='conversation-turn-delay-stddev'></a><code>--conversation-turn-delay-stddev</code><br><code>--session-turn-delay-stddev</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'><code>float</code></td>
<td style='width: 70%; vertical-align: top;'>The standard deviation of the delay between turns  within a conversation in milliseconds. <br><strong>Default:</strong> <code>0.0</code></td>
</tr>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='conversation-turn-delay-ratio'></a><code>--conversation-turn-delay-ratio</code><br><code>--session-delay-ratio</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'><code>float</code></td>
<td style='width: 70%; vertical-align: top;'>A ratio to scale multi-turn delays. <br><strong>Default:</strong> <code>1.0</code></td>
</tr>
</tbody>
</table>

### Output Options

<table>
<thead>
<tr>
<th style='white-space: nowrap; width: 20%; min-width: 150px;'>Option</th>
<th style='width: 10%; min-width: 80px;'>Type</th>
<th style='width: 70%;'>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='output-artifact-dir'></a><code>--output-artifact-dir</code><br><code>--artifact-dir</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'><code>str</code></td>
<td style='width: 70%; vertical-align: top;'>The directory to store all the (output) artifacts generated by AIPerf. <br><strong>Default:</strong> <code>artifacts</code></td>
</tr>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='profile-export-file'></a><code>--profile-export-file</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'><code>str</code></td>
<td style='width: 70%; vertical-align: top;'>The file to store the profile export in JSONL format. <br><strong>Default:</strong> <code>profile_export.jsonl</code></td>
</tr>
</tbody>
</table>

### Tokenizer Options

<table>
<thead>
<tr>
<th style='white-space: nowrap; width: 20%; min-width: 150px;'>Option</th>
<th style='width: 10%; min-width: 80px;'>Type</th>
<th style='width: 70%;'>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='tokenizer'></a><code>--tokenizer</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'><code>str</code></td>
<td style='width: 70%; vertical-align: top;'>The HuggingFace tokenizer to use to interpret token metrics from prompts and responses. The value can be the name of a tokenizer or the filepath of the tokenizer. The default value is the model name.</td>
</tr>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='tokenizer-revision'></a><code>--tokenizer-revision</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'><code>str</code></td>
<td style='width: 70%; vertical-align: top;'>The specific model version to use. It can be a branch name, tag name, or commit ID. <br><strong>Default:</strong> <code>main</code></td>
</tr>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='tokenizer-trust-remote-code'></a><code>--tokenizer-trust-remote-code</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'></td>
<td style='width: 70%; vertical-align: top;'>Allows custom tokenizer to be downloaded and executed. This carries security risks and should only be used for repositories you trust. This is only necessary for custom tokenizers stored in HuggingFace Hub.</td>
</tr>
</tbody>
</table>

### Load Generator Options

<table>
<thead>
<tr>
<th style='white-space: nowrap; width: 20%; min-width: 150px;'>Option</th>
<th style='width: 10%; min-width: 80px;'>Type</th>
<th style='width: 70%;'>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='benchmark-duration'></a><code>--benchmark-duration</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'><code>float</code></td>
<td style='width: 70%; vertical-align: top;'>The duration in seconds for benchmarking.</td>
</tr>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='benchmark-grace-period'></a><code>--benchmark-grace-period</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'><code>float</code></td>
<td style='width: 70%; vertical-align: top;'>The grace period in seconds to wait for responses after benchmark duration ends. Only applies when --benchmark-duration is set. Responses received within this period are included in metrics. <br><strong>Default:</strong> <code>30.0</code></td>
</tr>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='concurrency'></a><code>--concurrency</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'><code>int</code></td>
<td style='width: 70%; vertical-align: top;'>The concurrency value to benchmark.</td>
</tr>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='request-rate'></a><code>--request-rate</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'><code>float</code></td>
<td style='width: 70%; vertical-align: top;'>Sets the request rate for the load generated by AIPerf. Unit: requests/second.</td>
</tr>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='request-rate-mode'></a><code>--request-rate-mode</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'><code>str</code></td>
<td style='width: 70%; vertical-align: top;'>Sets the request rate mode for the load generated by AIPerf. Valid values: constant, poisson. constant: Generate requests at a fixed rate. poisson: Generate requests using a poisson distribution. <br><strong>Default:</strong> <code>poisson</code></td>
</tr>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='request-count'></a><code>--request-count</code><br><code>--num-requests</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'><code>int</code></td>
<td style='width: 70%; vertical-align: top;'>The number of requests to use for measurement. <br><strong>Default:</strong> <code>10</code></td>
</tr>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='warmup-request-count'></a><code>--warmup-request-count</code><br><code>--num-warmup-requests</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'><code>int</code></td>
<td style='width: 70%; vertical-align: top;'>The number of warmup requests to send before benchmarking. <br><strong>Default:</strong> <code>0</code></td>
</tr>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='request-cancellation-rate'></a><code>--request-cancellation-rate</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'><code>float</code></td>
<td style='width: 70%; vertical-align: top;'>The percentage of requests to cancel. <br><strong>Default:</strong> <code>0.0</code></td>
</tr>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='request-cancellation-delay'></a><code>--request-cancellation-delay</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'><code>float</code></td>
<td style='width: 70%; vertical-align: top;'>The delay in seconds before cancelling requests. This is used when --request-cancellation-rate is greater than 0. <br><strong>Default:</strong> <code>0.0</code></td>
</tr>
</tbody>
</table>

### ZMQ Communication Options

<table>
<thead>
<tr>
<th style='white-space: nowrap; width: 20%; min-width: 150px;'>Option</th>
<th style='width: 10%; min-width: 80px;'>Type</th>
<th style='width: 70%;'>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='zmq-host'></a><code>--zmq-host</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'><code>str</code></td>
<td style='width: 70%; vertical-align: top;'>Host address for TCP connections. <br><strong>Default:</strong> <code>127.0.0.1</code></td>
</tr>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='zmq-ipc-path'></a><code>--zmq-ipc-path</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'><code>str</code></td>
<td style='width: 70%; vertical-align: top;'>Path for IPC sockets.</td>
</tr>
</tbody>
</table>

### Workers Options

<table>
<thead>
<tr>
<th style='white-space: nowrap; width: 20%; min-width: 150px;'>Option</th>
<th style='width: 10%; min-width: 80px;'>Type</th>
<th style='width: 70%;'>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='workers-max'></a><code>--workers-max</code><br><code>--max-workers</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'><code>int</code></td>
<td style='width: 70%; vertical-align: top;'>Maximum number of workers to create. If not specified, the number of workers will be determined by the formula <code>min(concurrency, (num CPUs * 0.75) - 1)</code>,  with a default max cap of <code>32</code>. Any value provided will still be capped by the concurrency value (if specified), but not by the max cap.</td>
</tr>
</tbody>
</table>

### Service Options

<table>
<thead>
<tr>
<th style='white-space: nowrap; width: 20%; min-width: 150px;'>Option</th>
<th style='width: 10%; min-width: 80px;'>Type</th>
<th style='width: 70%;'>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='log-level'></a><code>--log-level</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'><code>str</code></td>
<td style='width: 70%; vertical-align: top;'>Logging level. <br><strong>Choices:</strong> <code>TRACE</code>, <code>DEBUG</code>, <code>INFO</code>, <code>NOTICE</code>, <code>WARNING</code>, <code>SUCCESS</code>, <code>ERROR</code>, <code>CRITICAL</code> <br><strong>Default:</strong> <code>INFO</code></td>
</tr>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='verbose'></a><code>-v</code><br><code>--verbose</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'></td>
<td style='width: 70%; vertical-align: top;'>Equivalent to --log-level DEBUG. Enables more verbose logging output, but lacks some raw message logging.</td>
</tr>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='extra-verbose'></a><code>-vv</code><br><code>--extra-verbose</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'></td>
<td style='width: 70%; vertical-align: top;'>Equivalent to --log-level TRACE. Enables the most verbose logging output possible.</td>
</tr>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='record-processor-service-count'></a><code>--record-processor-service-count</code><br><code>--record-processors</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'><code>int</code></td>
<td style='width: 70%; vertical-align: top;'>Number of services to spawn for processing records. The higher the request rate, the more services should be spawned in order to keep up with the incoming records. If not specified, the number of services will be automatically determined based on the worker count.</td>
</tr>
<tr>
<td style='white-space: nowrap; width: 20%; min-width: 150px; vertical-align: top;'><a id='ui-type'></a><code>--ui-type</code><br><code>--ui</code></td>
<td style='width: 10%; min-width: 80px; vertical-align: top;'><code>str</code></td>
<td style='width: 70%; vertical-align: top;'>Type of UI to use. <br><strong>Choices:</strong> <code>dashboard</code>, <code>simple</code>, <code>none</code> <br><strong>Default:</strong> <code>dashboard</code></td>
</tr>
</tbody>
</table>
