<!--
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->

## Command Line Options

### Endpoint Options

<table>
<thead>
<tr>
<th style='white-space: nowrap;'>Option</th>
<th>Type</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td style='white-space: nowrap;'><a id='model-names'></a>`-m`<br>`--model-names`<br>`--model`</td>
<td>`list`</td>
<td>Model name(s) to be benchmarked. Can be a comma-separated list or a single model name.</td>
</tr>
<tr>
<td style='white-space: nowrap;'><a id='model-selection-strategy'></a>`--model-selection-strategy`</td>
<td>`str`</td>
<td>When multiple models are specified, this is how a specific model should be assigned to a prompt. round_robin: nth prompt in the list gets assigned to n-mod len(models). random: assignment is uniformly random. <br>**Choices:** `round_robin`, `random` <br>**Default:** `round_robin`</td>
</tr>
<tr>
<td style='white-space: nowrap;'><a id='custom-endpoint'></a>`--custom-endpoint`<br>`--endpoint`</td>
<td>`str`</td>
<td>Set a custom endpoint that differs from the OpenAI defaults.</td>
</tr>
<tr>
<td style='white-space: nowrap;'><a id='endpoint-type'></a>`--endpoint-type`</td>
<td>`str`</td>
<td>The endpoint type to send requests to on the server. <br>**Choices:** `chat`, `completions`, `embeddings`, `rankings` <br>**Default:** `chat`</td>
</tr>
<tr>
<td style='white-space: nowrap;'><a id='streaming'></a>`--streaming`</td>
<td></td>
<td>An option to enable the use of the streaming API.</td>
</tr>
<tr>
<td style='white-space: nowrap;'><a id='url'></a>`-u`<br>`--url`</td>
<td>`str`</td>
<td>URL of the endpoint to target for benchmarking. <br>**Default:** `localhost:8000`</td>
</tr>
<tr>
<td style='white-space: nowrap;'><a id='request-timeout-seconds'></a>`--request-timeout-seconds`</td>
<td>`float`</td>
<td>The timeout in floating-point seconds for each request to the endpoint. <br>**Default:** `600.0`</td>
</tr>
<tr>
<td style='white-space: nowrap;'><a id='api-key'></a>`--api-key`</td>
<td>`str`</td>
<td>The API key to use for the endpoint. If provided, it will be sent with every request as a header: `Authorization: Bearer <api_key>`.</td>
</tr>
</tbody>
</table>

### Input Options

<table>
<thead>
<tr>
<th style='white-space: nowrap;'>Option</th>
<th>Type</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td style='white-space: nowrap;'><a id='extra-inputs'></a>`--extra-inputs`</td>
<td>`list`</td>
<td>Provide additional inputs to include with every request. Inputs should be in an 'input_name:value' format. Alternatively, a string representing a json formatted dict can be provided. <br>**Default:** `[]`</td>
</tr>
<tr>
<td style='white-space: nowrap;'><a id='header'></a>`-H`<br>`--header`</td>
<td>`list`</td>
<td>Adds a custom header to the requests. Headers must be specified as 'Header:Value' pairs. Alternatively, a string representing a json formatted dict can be provided. <br>**Default:** `[]`</td>
</tr>
<tr>
<td style='white-space: nowrap;'><a id='input-file'></a>`--input-file`</td>
<td>`str`</td>
<td>The file or directory path that contains the dataset to use for profiling. This parameter is used in conjunction with the `custom_dataset_type` parameter to support different types of user provided datasets.</td>
</tr>
<tr>
<td style='white-space: nowrap;'><a id='fixed-schedule'></a>`--fixed-schedule`</td>
<td></td>
<td>Specifies to run a fixed schedule of requests. This is normally inferred from the --input-file parameter, but can be set manually here.</td>
</tr>
<tr>
<td style='white-space: nowrap;'><a id='fixed-schedule-auto-offset'></a>`--fixed-schedule-auto-offset`</td>
<td></td>
<td>Specifies to automatically offset the timestamps in the fixed schedule, such that the first timestamp is considered 0, and the rest are shifted accordingly. If disabled, the timestamps will be assumed to be relative to 0.</td>
</tr>
<tr>
<td style='white-space: nowrap;'><a id='fixed-schedule-start-offset'></a>`--fixed-schedule-start-offset`</td>
<td>`int`</td>
<td>Specifies the offset in milliseconds to start the fixed schedule at. By default, the schedule starts at 0, but this option can be used to start at a reference point further in the schedule. This option cannot be used in conjunction with the --fixed-schedule-auto-offset. The schedule will include any requests at the start offset.</td>
</tr>
<tr>
<td style='white-space: nowrap;'><a id='fixed-schedule-end-offset'></a>`--fixed-schedule-end-offset`</td>
<td>`int`</td>
<td>Specifies the offset in milliseconds to end the fixed schedule at. By default, the schedule ends at the last timestamp in the trace dataset, but this option can be used to only run a subset of the trace. The schedule will include any requests at the end offset.</td>
</tr>
<tr>
<td style='white-space: nowrap;'><a id='public-dataset'></a>`--public-dataset`</td>
<td>`str`</td>
<td>The public dataset to use for the requests. <br>**Choices:** `sharegpt`</td>
</tr>
<tr>
<td style='white-space: nowrap;'><a id='custom-dataset-type'></a>`--custom-dataset-type`</td>
<td>`str`</td>
<td>The type of custom dataset to use. This parameter is used in conjunction with the --input-file parameter. [choices: single_turn, multi_turn, random_pool, mooncake_trace].</td>
</tr>
<tr>
<td style='white-space: nowrap;'><a id='random-seed'></a>`--random-seed`</td>
<td>`int`</td>
<td>The seed used to generate random values. Set to some value to make the synthetic data generation deterministic. It will use system default if not provided.</td>
</tr>
<tr>
<td style='white-space: nowrap;'><a id='goodput'></a>`--goodput`</td>
<td>`str`</td>
<td>Specify service level objectives (SLOs) for goodput as space-separated 'KEY:VALUE' pairs, where KEY is a metric tag and VALUE is a number in the metric’s display unit (falls back to its base unit if no display unit is defined). Examples: 'request_latency:250' (ms), 'inter_token_latency:10' (ms), `output_token_throughput_per_user:600` (tokens/s). Only metrics applicable to the current endpoint/config are considered. For more context on the definition of goodput, refer to DistServe paper: https://arxiv.org/pdf/2401.09670 and the blog: https://hao-ai-lab.github.io/blogs/distserve.</td>
</tr>
</tbody>
</table>

### Audio Input Options

<table>
<thead>
<tr>
<th style='white-space: nowrap;'>Option</th>
<th>Type</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td style='white-space: nowrap;'><a id='audio-batch-size'></a>`--audio-batch-size`<br>`--batch-size-audio`</td>
<td>`int`</td>
<td>The batch size of audio requests AIPerf should send. This is currently supported with the OpenAI `chat` endpoint type. <br>**Default:** `1`</td>
</tr>
<tr>
<td style='white-space: nowrap;'><a id='audio-length-mean'></a>`--audio-length-mean`</td>
<td>`float`</td>
<td>The mean length of the audio in seconds. <br>**Default:** `0.0`</td>
</tr>
<tr>
<td style='white-space: nowrap;'><a id='audio-length-stddev'></a>`--audio-length-stddev`</td>
<td>`float`</td>
<td>The standard deviation of the length of the audio in seconds. <br>**Default:** `0.0`</td>
</tr>
<tr>
<td style='white-space: nowrap;'><a id='audio-format'></a>`--audio-format`</td>
<td>`str`</td>
<td>The format of the audio files (wav or mp3). <br>**Choices:** `wav`, `mp3` <br>**Default:** `wav`</td>
</tr>
<tr>
<td style='white-space: nowrap;'><a id='audio-depths'></a>`--audio-depths`</td>
<td>`list`</td>
<td>A list of audio bit depths to randomly select from in bits. <br>**Default:** `[16]`</td>
</tr>
<tr>
<td style='white-space: nowrap;'><a id='audio-sample-rates'></a>`--audio-sample-rates`</td>
<td>`list`</td>
<td>A list of audio sample rates to randomly select from in kHz. Common sample rates are 16, 44.1, 48, 96, etc. <br>**Default:** `[16.0]`</td>
</tr>
<tr>
<td style='white-space: nowrap;'><a id='audio-num-channels'></a>`--audio-num-channels`</td>
<td>`int`</td>
<td>The number of audio channels to use for the audio data generation. <br>**Default:** `1`</td>
</tr>
</tbody>
</table>

### Image Input Options

<table>
<thead>
<tr>
<th style='white-space: nowrap;'>Option</th>
<th>Type</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td style='white-space: nowrap;'><a id='image-width-mean'></a>`--image-width-mean`</td>
<td>`float`</td>
<td>The mean width of images when generating synthetic image data. <br>**Default:** `0.0`</td>
</tr>
<tr>
<td style='white-space: nowrap;'><a id='image-width-stddev'></a>`--image-width-stddev`</td>
<td>`float`</td>
<td>The standard deviation of width of images when generating synthetic image data. <br>**Default:** `0.0`</td>
</tr>
<tr>
<td style='white-space: nowrap;'><a id='image-height-mean'></a>`--image-height-mean`</td>
<td>`float`</td>
<td>The mean height of images when generating synthetic image data. <br>**Default:** `0.0`</td>
</tr>
<tr>
<td style='white-space: nowrap;'><a id='image-height-stddev'></a>`--image-height-stddev`</td>
<td>`float`</td>
<td>The standard deviation of height of images when generating synthetic image data. <br>**Default:** `0.0`</td>
</tr>
<tr>
<td style='white-space: nowrap;'><a id='image-batch-size'></a>`--image-batch-size`<br>`--batch-size-image`</td>
<td>`int`</td>
<td>The image batch size of the requests AIPerf should send. This is currently supported with the image retrieval endpoint type. <br>**Default:** `1`</td>
</tr>
<tr>
<td style='white-space: nowrap;'><a id='image-format'></a>`--image-format`</td>
<td>`str`</td>
<td>The compression format of the images. <br>**Choices:** `png`, `jpeg`, `random` <br>**Default:** `png`</td>
</tr>
</tbody>
</table>

### Prompt Options

<table>
<thead>
<tr>
<th style='white-space: nowrap;'>Option</th>
<th>Type</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td style='white-space: nowrap;'><a id='prompt-batch-size'></a>`-b`<br>`--prompt-batch-size`<br>`--batch-size-text`<br>`--batch-size`</td>
<td>`int`</td>
<td>The batch size of text requests AIPerf should send. This is currently supported with the embeddings and rankings endpoint types. <br>**Default:** `1`</td>
</tr>
</tbody>
</table>

### Input Sequence Length (ISL) Options

<table>
<thead>
<tr>
<th style='white-space: nowrap;'>Option</th>
<th>Type</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td style='white-space: nowrap;'><a id='prompt-input-tokens-mean'></a>`--prompt-input-tokens-mean`<br>`--synthetic-input-tokens-mean`<br>`--isl`</td>
<td>`int`</td>
<td>The mean of number of tokens in the generated prompts when using synthetic data. <br>**Default:** `550`</td>
</tr>
<tr>
<td style='white-space: nowrap;'><a id='prompt-input-tokens-stddev'></a>`--prompt-input-tokens-stddev`<br>`--synthetic-input-tokens-stddev`<br>`--isl-stddev`</td>
<td>`float`</td>
<td>The standard deviation of number of tokens in the generated prompts when using synthetic data. <br>**Default:** `0.0`</td>
</tr>
<tr>
<td style='white-space: nowrap;'><a id='prompt-input-tokens-block-size'></a>`--prompt-input-tokens-block-size`<br>`--synthetic-input-tokens-block-size`<br>`--isl-block-size`</td>
<td>`int`</td>
<td>The block size of the prompt. <br>**Default:** `512`</td>
</tr>
<tr>
<td style='white-space: nowrap;'><a id='seq-dist'></a>`--seq-dist`<br>`--sequence-distribution`</td>
<td>`str`</td>
<td>Sequence length distribution specification for varying ISL/OSL pairs.</td>
</tr>
</tbody>
</table>

### Output Sequence Length (OSL) Options

<table>
<thead>
<tr>
<th style='white-space: nowrap;'>Option</th>
<th>Type</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td style='white-space: nowrap;'><a id='prompt-output-tokens-mean'></a>`--prompt-output-tokens-mean`<br>`--output-tokens-mean`<br>`--osl`</td>
<td>`int`</td>
<td>The mean number of tokens in each output.</td>
</tr>
<tr>
<td style='white-space: nowrap;'><a id='prompt-output-tokens-stddev'></a>`--prompt-output-tokens-stddev`<br>`--output-tokens-stddev`<br>`--osl-stddev`</td>
<td>`float`</td>
<td>The standard deviation of the number of tokens in each output. <br>**Default:** `0`</td>
</tr>
</tbody>
</table>

### Prefix Prompt Options

<table>
<thead>
<tr>
<th style='white-space: nowrap;'>Option</th>
<th>Type</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td style='white-space: nowrap;'><a id='prompt-prefix-pool-size'></a>`--prompt-prefix-pool-size`<br>`--prefix-prompt-pool-size`<br>`--num-prefix-prompts`</td>
<td>`int`</td>
<td>The total size of the prefix prompt pool to select prefixes from. If this value is not zero, these are prompts that are prepended to input prompts. This is useful for benchmarking models that use a K-V cache. <br>**Default:** `0`</td>
</tr>
<tr>
<td style='white-space: nowrap;'><a id='prompt-prefix-length'></a>`--prompt-prefix-length`<br>`--prefix-prompt-length`</td>
<td>`int`</td>
<td>The number of tokens in each prefix prompt. This is only used if "num" is greater than zero. Note that due to the prefix and user prompts being concatenated, the number of tokens in the final prompt may be off by one. <br>**Default:** `0`</td>
</tr>
</tbody>
</table>

### Conversation Input Options

<table>
<thead>
<tr>
<th style='white-space: nowrap;'>Option</th>
<th>Type</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td style='white-space: nowrap;'><a id='conversation-num'></a>`--conversation-num`<br>`--num-conversations`<br>`--num-sessions`<br>`--num-dataset-entries`</td>
<td>`int`</td>
<td>The total number of unique conversations to generate. Each conversation represents a single request session between client and server. Supported on synthetic mode and the custom random_pool dataset. The number of conversations  will be used to determine the number of entries in both the custom random_pool and synthetic  datasets and will be reused until benchmarking is complete. <br>**Default:** `100`</td>
</tr>
<tr>
<td style='white-space: nowrap;'><a id='conversation-turn-mean'></a>`--conversation-turn-mean`<br>`--session-turns-mean`</td>
<td>`int`</td>
<td>The mean number of turns within a conversation. <br>**Default:** `1`</td>
</tr>
<tr>
<td style='white-space: nowrap;'><a id='conversation-turn-stddev'></a>`--conversation-turn-stddev`<br>`--session-turns-stddev`</td>
<td>`int`</td>
<td>The standard deviation of the number of turns within a conversation. <br>**Default:** `0`</td>
</tr>
<tr>
<td style='white-space: nowrap;'><a id='conversation-turn-delay-mean'></a>`--conversation-turn-delay-mean`<br>`--session-turn-delay-mean`</td>
<td>`float`</td>
<td>The mean delay between turns within a conversation in milliseconds. <br>**Default:** `0.0`</td>
</tr>
<tr>
<td style='white-space: nowrap;'><a id='conversation-turn-delay-stddev'></a>`--conversation-turn-delay-stddev`<br>`--session-turn-delay-stddev`</td>
<td>`float`</td>
<td>The standard deviation of the delay between turns  within a conversation in milliseconds. <br>**Default:** `0.0`</td>
</tr>
<tr>
<td style='white-space: nowrap;'><a id='conversation-turn-delay-ratio'></a>`--conversation-turn-delay-ratio`<br>`--session-delay-ratio`</td>
<td>`float`</td>
<td>A ratio to scale multi-turn delays. <br>**Default:** `1.0`</td>
</tr>
</tbody>
</table>

### Output Options

<table>
<thead>
<tr>
<th style='white-space: nowrap;'>Option</th>
<th>Type</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td style='white-space: nowrap;'><a id='output-artifact-dir'></a>`--output-artifact-dir`<br>`--artifact-dir`</td>
<td>`str`</td>
<td>The directory to store all the (output) artifacts generated by AIPerf. <br>**Default:** `artifacts`</td>
</tr>
<tr>
<td style='white-space: nowrap;'><a id='profile-export-file'></a>`--profile-export-file`</td>
<td>`str`</td>
<td>The file to store the profile export in JSONL format. <br>**Default:** `profile_export.jsonl`</td>
</tr>
</tbody>
</table>

### Tokenizer Options

<table>
<thead>
<tr>
<th style='white-space: nowrap;'>Option</th>
<th>Type</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td style='white-space: nowrap;'><a id='tokenizer'></a>`--tokenizer`</td>
<td>`str`</td>
<td>The HuggingFace tokenizer to use to interpret token metrics from prompts and responses. The value can be the name of a tokenizer or the filepath of the tokenizer. The default value is the model name.</td>
</tr>
<tr>
<td style='white-space: nowrap;'><a id='tokenizer-revision'></a>`--tokenizer-revision`</td>
<td>`str`</td>
<td>The specific model version to use. It can be a branch name, tag name, or commit ID. <br>**Default:** `main`</td>
</tr>
<tr>
<td style='white-space: nowrap;'><a id='tokenizer-trust-remote-code'></a>`--tokenizer-trust-remote-code`</td>
<td></td>
<td>Allows custom tokenizer to be downloaded and executed. This carries security risks and should only be used for repositories you trust. This is only necessary for custom tokenizers stored in HuggingFace Hub.</td>
</tr>
</tbody>
</table>

### Load Generator Options

<table>
<thead>
<tr>
<th style='white-space: nowrap;'>Option</th>
<th>Type</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td style='white-space: nowrap;'><a id='benchmark-duration'></a>`--benchmark-duration`</td>
<td>`float`</td>
<td>The duration in seconds for benchmarking.</td>
</tr>
<tr>
<td style='white-space: nowrap;'><a id='benchmark-grace-period'></a>`--benchmark-grace-period`</td>
<td>`float`</td>
<td>The grace period in seconds to wait for responses after benchmark duration ends. Only applies when --benchmark-duration is set. Responses received within this period are included in metrics. <br>**Default:** `30.0`</td>
</tr>
<tr>
<td style='white-space: nowrap;'><a id='concurrency'></a>`--concurrency`</td>
<td>`int`</td>
<td>The concurrency value to benchmark.</td>
</tr>
<tr>
<td style='white-space: nowrap;'><a id='request-rate'></a>`--request-rate`</td>
<td>`float`</td>
<td>Sets the request rate for the load generated by AIPerf. Unit: requests/second.</td>
</tr>
<tr>
<td style='white-space: nowrap;'><a id='request-rate-mode'></a>`--request-rate-mode`</td>
<td>`str`</td>
<td>Sets the request rate mode for the load generated by AIPerf. Valid values: constant, poisson. constant: Generate requests at a fixed rate. poisson: Generate requests using a poisson distribution. <br>**Default:** `poisson`</td>
</tr>
<tr>
<td style='white-space: nowrap;'><a id='request-count'></a>`--request-count`<br>`--num-requests`</td>
<td>`int`</td>
<td>The number of requests to use for measurement. <br>**Default:** `10`</td>
</tr>
<tr>
<td style='white-space: nowrap;'><a id='warmup-request-count'></a>`--warmup-request-count`<br>`--num-warmup-requests`</td>
<td>`int`</td>
<td>The number of warmup requests to send before benchmarking. <br>**Default:** `0`</td>
</tr>
<tr>
<td style='white-space: nowrap;'><a id='request-cancellation-rate'></a>`--request-cancellation-rate`</td>
<td>`float`</td>
<td>The percentage of requests to cancel. <br>**Default:** `0.0`</td>
</tr>
<tr>
<td style='white-space: nowrap;'><a id='request-cancellation-delay'></a>`--request-cancellation-delay`</td>
<td>`float`</td>
<td>The delay in seconds before cancelling requests. This is used when --request-cancellation-rate is greater than 0. <br>**Default:** `0.0`</td>
</tr>
</tbody>
</table>

### ZMQ Communication Options

<table>
<thead>
<tr>
<th style='white-space: nowrap;'>Option</th>
<th>Type</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td style='white-space: nowrap;'><a id='zmq-host'></a>`--zmq-host`</td>
<td>`str`</td>
<td>Host address for TCP connections. <br>**Default:** `127.0.0.1`</td>
</tr>
<tr>
<td style='white-space: nowrap;'><a id='zmq-ipc-path'></a>`--zmq-ipc-path`</td>
<td>`str`</td>
<td>Path for IPC sockets.</td>
</tr>
</tbody>
</table>

### Workers Options

<table>
<thead>
<tr>
<th style='white-space: nowrap;'>Option</th>
<th>Type</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td style='white-space: nowrap;'><a id='workers-max'></a>`--workers-max`<br>`--max-workers`</td>
<td>`int`</td>
<td>Maximum number of workers to create. If not specified, the number of workers will be determined by the formula `min(concurrency, (num CPUs * 0.75) - 1)`,  with a default max cap of `32`. Any value provided will still be capped by the concurrency value (if specified), but not by the max cap.</td>
</tr>
</tbody>
</table>

### Service Options

<table>
<thead>
<tr>
<th style='white-space: nowrap;'>Option</th>
<th>Type</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td style='white-space: nowrap;'><a id='log-level'></a>`--log-level`</td>
<td>`str`</td>
<td>Logging level. <br>**Choices:** `TRACE`, `DEBUG`, `INFO`, `NOTICE`, `WARNING`, `SUCCESS`, `ERROR`, `CRITICAL` <br>**Default:** `INFO`</td>
</tr>
<tr>
<td style='white-space: nowrap;'><a id='verbose'></a>`-v`<br>`--verbose`</td>
<td></td>
<td>Equivalent to --log-level DEBUG. Enables more verbose logging output, but lacks some raw message logging.</td>
</tr>
<tr>
<td style='white-space: nowrap;'><a id='extra-verbose'></a>`-vv`<br>`--extra-verbose`</td>
<td></td>
<td>Equivalent to --log-level TRACE. Enables the most verbose logging output possible.</td>
</tr>
<tr>
<td style='white-space: nowrap;'><a id='record-processor-service-count'></a>`--record-processor-service-count`<br>`--record-processors`</td>
<td>`int`</td>
<td>Number of services to spawn for processing records. The higher the request rate, the more services should be spawned in order to keep up with the incoming records. If not specified, the number of services will be automatically determined based on the worker count.</td>
</tr>
<tr>
<td style='white-space: nowrap;'><a id='ui-type'></a>`--ui-type`<br>`--ui`</td>
<td>`str`</td>
<td>Type of UI to use. <br>**Choices:** `dashboard`, `simple`, `none` <br>**Default:** `dashboard`</td>
</tr>
</tbody>
</table>
