<!--
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->

## Command Line Options

### Endpoint Options

| Option | Description |
|--------|-------------|
| <a id='model-names'></a><nobr>`-m <list>`</nobr><br><nobr>`--model-names <list>`</nobr><br><nobr>`--model <list>`</nobr> | Model name(s) to be benchmarked. Can be a comma-separated list or a single model name. |
| <a id='model-selection-strategy'></a><nobr>`--model-selection-strategy <str>`</nobr> | When multiple models are specified, this is how a specific model should be assigned to a prompt. round_robin: nth prompt in the list gets assigned to n-mod len(models). random: assignment is uniformly random. <br>**Choices:** `round_robin`, `random` <br>**Default:** `round_robin` |
| <a id='custom-endpoint'></a><nobr>`--custom-endpoint <str>`</nobr><br><nobr>`--endpoint <str>`</nobr> | Set a custom endpoint that differs from the OpenAI defaults. |
| <a id='endpoint-type'></a><nobr>`--endpoint-type <str>`</nobr> | The endpoint type to send requests to on the server. <br>**Choices:** `chat`, `completions`, `embeddings`, `rankings` <br>**Default:** `chat` |
| <a id='streaming'></a><nobr>`--streaming`</nobr> | An option to enable the use of the streaming API. |
| <a id='url'></a><nobr>`-u <str>`</nobr><br><nobr>`--url <str>`</nobr> | URL of the endpoint to target for benchmarking. <br>**Default:** `localhost:8000` |
| <a id='request-timeout-seconds'></a><nobr>`--request-timeout-seconds <float>`</nobr> | The timeout in floating-point seconds for each request to the endpoint. <br>**Default:** `600.0` |
| <a id='api-key'></a><nobr>`--api-key <str>`</nobr> | The API key to use for the endpoint. If provided, it will be sent with every request as a header: `Authorization: Bearer <api_key>`. |

### Input Options

| Option | Description |
|--------|-------------|
| <a id='extra-inputs'></a><nobr>`--extra-inputs <list>`</nobr> | Provide additional inputs to include with every request. Inputs should be in an 'input_name:value' format. Alternatively, a string representing a json formatted dict can be provided. <br>**Default:** `[]` |
| <a id='header'></a><nobr>`-H <list>`</nobr><br><nobr>`--header <list>`</nobr> | Adds a custom header to the requests. Headers must be specified as 'Header:Value' pairs. Alternatively, a string representing a json formatted dict can be provided. <br>**Default:** `[]` |
| <a id='input-file'></a><nobr>`--input-file <str>`</nobr> | The file or directory path that contains the dataset to use for profiling. This parameter is used in conjunction with the `custom_dataset_type` parameter to support different types of user provided datasets. |
| <a id='fixed-schedule'></a><nobr>`--fixed-schedule`</nobr> | Specifies to run a fixed schedule of requests. This is normally inferred from the --input-file parameter, but can be set manually here. |
| <a id='fixed-schedule-auto-offset'></a><nobr>`--fixed-schedule-auto-offset`</nobr> | Specifies to automatically offset the timestamps in the fixed schedule, such that the first timestamp is considered 0, and the rest are shifted accordingly. If disabled, the timestamps will be assumed to be relative to 0. |
| <a id='fixed-schedule-start-offset'></a><nobr>`--fixed-schedule-start-offset <int>`</nobr> | Specifies the offset in milliseconds to start the fixed schedule at. By default, the schedule starts at 0, but this option can be used to start at a reference point further in the schedule. This option cannot be used in conjunction with the --fixed-schedule-auto-offset. The schedule will include any requests at the start offset. |
| <a id='fixed-schedule-end-offset'></a><nobr>`--fixed-schedule-end-offset <int>`</nobr> | Specifies the offset in milliseconds to end the fixed schedule at. By default, the schedule ends at the last timestamp in the trace dataset, but this option can be used to only run a subset of the trace. The schedule will include any requests at the end offset. |
| <a id='public-dataset'></a><nobr>`--public-dataset <str>`</nobr> | The public dataset to use for the requests. <br>**Choices:** `sharegpt` |
| <a id='custom-dataset-type'></a><nobr>`--custom-dataset-type <str>`</nobr> | The type of custom dataset to use. This parameter is used in conjunction with the --input-file parameter. [choices: single_turn, multi_turn, random_pool, mooncake_trace]. |
| <a id='random-seed'></a><nobr>`--random-seed <int>`</nobr> | The seed used to generate random values. Set to some value to make the synthetic data generation deterministic. It will use system default if not provided. |
| <a id='goodput'></a><nobr>`--goodput <str>`</nobr> | Specify service level objectives (SLOs) for goodput as space-separated 'KEY:VALUE' pairs, where KEY is a metric tag and VALUE is a number in the metric’s display unit (falls back to its base unit if no display unit is defined). Examples: 'request_latency:250' (ms), 'inter_token_latency:10' (ms), `output_token_throughput_per_user:600` (tokens/s). Only metrics applicable to the current endpoint/config are considered. For more context on the definition of goodput, refer to DistServe paper: https://arxiv.org/pdf/2401.09670 and the blog: https://hao-ai-lab.github.io/blogs/distserve. |

### Audio Input Options

| Option | Description |
|--------|-------------|
| <a id='audio-batch-size'></a><nobr>`--audio-batch-size <int>`</nobr><br><nobr>`--batch-size-audio <int>`</nobr> | The batch size of audio requests AIPerf should send. This is currently supported with the OpenAI `chat` endpoint type. <br>**Default:** `1` |
| <a id='audio-length-mean'></a><nobr>`--audio-length-mean <float>`</nobr> | The mean length of the audio in seconds. <br>**Default:** `0.0` |
| <a id='audio-length-stddev'></a><nobr>`--audio-length-stddev <float>`</nobr> | The standard deviation of the length of the audio in seconds. <br>**Default:** `0.0` |
| <a id='audio-format'></a><nobr>`--audio-format <str>`</nobr> | The format of the audio files (wav or mp3). <br>**Choices:** `wav`, `mp3` <br>**Default:** `wav` |
| <a id='audio-depths'></a><nobr>`--audio-depths <list>`</nobr> | A list of audio bit depths to randomly select from in bits. <br>**Default:** `[16]` |
| <a id='audio-sample-rates'></a><nobr>`--audio-sample-rates <list>`</nobr> | A list of audio sample rates to randomly select from in kHz. Common sample rates are 16, 44.1, 48, 96, etc. <br>**Default:** `[16.0]` |
| <a id='audio-num-channels'></a><nobr>`--audio-num-channels <int>`</nobr> | The number of audio channels to use for the audio data generation. <br>**Default:** `1` |

### Image Input Options

| Option | Description |
|--------|-------------|
| <a id='image-width-mean'></a><nobr>`--image-width-mean <float>`</nobr> | The mean width of images when generating synthetic image data. <br>**Default:** `0.0` |
| <a id='image-width-stddev'></a><nobr>`--image-width-stddev <float>`</nobr> | The standard deviation of width of images when generating synthetic image data. <br>**Default:** `0.0` |
| <a id='image-height-mean'></a><nobr>`--image-height-mean <float>`</nobr> | The mean height of images when generating synthetic image data. <br>**Default:** `0.0` |
| <a id='image-height-stddev'></a><nobr>`--image-height-stddev <float>`</nobr> | The standard deviation of height of images when generating synthetic image data. <br>**Default:** `0.0` |
| <a id='image-batch-size'></a><nobr>`--image-batch-size <int>`</nobr><br><nobr>`--batch-size-image <int>`</nobr> | The image batch size of the requests AIPerf should send. This is currently supported with the image retrieval endpoint type. <br>**Default:** `1` |
| <a id='image-format'></a><nobr>`--image-format <str>`</nobr> | The compression format of the images. <br>**Choices:** `png`, `jpeg`, `random` <br>**Default:** `png` |

### Prompt Options

| Option | Description |
|--------|-------------|
| <a id='prompt-batch-size'></a><nobr>`-b <int>`</nobr><br><nobr>`--prompt-batch-size <int>`</nobr><br><nobr>`--batch-size-text <int>`</nobr><br><nobr>`--batch-size <int>`</nobr> | The batch size of text requests AIPerf should send. This is currently supported with the embeddings and rankings endpoint types. <br>**Default:** `1` |

### Input Sequence Length (ISL) Options

| Option | Description |
|--------|-------------|
| <a id='prompt-input-tokens-mean'></a><nobr>`--prompt-input-tokens-mean <int>`</nobr><br><nobr>`--synthetic-input-tokens-mean <int>`</nobr><br><nobr>`--isl <int>`</nobr> | The mean of number of tokens in the generated prompts when using synthetic data. <br>**Default:** `550` |
| <a id='prompt-input-tokens-stddev'></a><nobr>`--prompt-input-tokens-stddev <float>`</nobr><br><nobr>`--synthetic-input-tokens-stddev <float>`</nobr><br><nobr>`--isl-stddev <float>`</nobr> | The standard deviation of number of tokens in the generated prompts when using synthetic data. <br>**Default:** `0.0` |
| <a id='prompt-input-tokens-block-size'></a><nobr>`--prompt-input-tokens-block-size <int>`</nobr><br><nobr>`--synthetic-input-tokens-block-size <int>`</nobr><br><nobr>`--isl-block-size <int>`</nobr> | The block size of the prompt. <br>**Default:** `512` |
| <a id='seq-dist'></a><nobr>`--seq-dist <str>`</nobr><br><nobr>`--sequence-distribution <str>`</nobr> | Sequence length distribution specification for varying ISL/OSL pairs. |

### Output Sequence Length (OSL) Options

| Option | Description |
|--------|-------------|
| <a id='prompt-output-tokens-mean'></a><nobr>`--prompt-output-tokens-mean <int>`</nobr><br><nobr>`--output-tokens-mean <int>`</nobr><br><nobr>`--osl <int>`</nobr> | The mean number of tokens in each output. |
| <a id='prompt-output-tokens-stddev'></a><nobr>`--prompt-output-tokens-stddev <float>`</nobr><br><nobr>`--output-tokens-stddev <float>`</nobr><br><nobr>`--osl-stddev <float>`</nobr> | The standard deviation of the number of tokens in each output. <br>**Default:** `0` |

### Prefix Prompt Options

| Option | Description |
|--------|-------------|
| <a id='prompt-prefix-pool-size'></a><nobr>`--prompt-prefix-pool-size <int>`</nobr><br><nobr>`--prefix-prompt-pool-size <int>`</nobr><br><nobr>`--num-prefix-prompts <int>`</nobr> | The total size of the prefix prompt pool to select prefixes from. If this value is not zero, these are prompts that are prepended to input prompts. This is useful for benchmarking models that use a K-V cache. <br>**Default:** `0` |
| <a id='prompt-prefix-length'></a><nobr>`--prompt-prefix-length <int>`</nobr><br><nobr>`--prefix-prompt-length <int>`</nobr> | The number of tokens in each prefix prompt. This is only used if "num" is greater than zero. Note that due to the prefix and user prompts being concatenated, the number of tokens in the final prompt may be off by one. <br>**Default:** `0` |

### Conversation Input Options

| Option | Description |
|--------|-------------|
| <a id='conversation-num'></a><nobr>`--conversation-num <int>`</nobr><br><nobr>`--num-conversations <int>`</nobr><br><nobr>`--num-sessions <int>`</nobr><br><nobr>`--num-dataset-entries <int>`</nobr> | The total number of unique conversations to generate. Each conversation represents a single request session between client and server. Supported on synthetic mode and the custom random_pool dataset. The number of conversations  will be used to determine the number of entries in both the custom random_pool and synthetic  datasets and will be reused until benchmarking is complete. <br>**Default:** `100` |
| <a id='conversation-turn-mean'></a><nobr>`--conversation-turn-mean <int>`</nobr><br><nobr>`--session-turns-mean <int>`</nobr> | The mean number of turns within a conversation. <br>**Default:** `1` |
| <a id='conversation-turn-stddev'></a><nobr>`--conversation-turn-stddev <int>`</nobr><br><nobr>`--session-turns-stddev <int>`</nobr> | The standard deviation of the number of turns within a conversation. <br>**Default:** `0` |
| <a id='conversation-turn-delay-mean'></a><nobr>`--conversation-turn-delay-mean <float>`</nobr><br><nobr>`--session-turn-delay-mean <float>`</nobr> | The mean delay between turns within a conversation in milliseconds. <br>**Default:** `0.0` |
| <a id='conversation-turn-delay-stddev'></a><nobr>`--conversation-turn-delay-stddev <float>`</nobr><br><nobr>`--session-turn-delay-stddev <float>`</nobr> | The standard deviation of the delay between turns  within a conversation in milliseconds. <br>**Default:** `0.0` |
| <a id='conversation-turn-delay-ratio'></a><nobr>`--conversation-turn-delay-ratio <float>`</nobr><br><nobr>`--session-delay-ratio <float>`</nobr> | A ratio to scale multi-turn delays. <br>**Default:** `1.0` |

### Output Options

| Option | Description |
|--------|-------------|
| <a id='output-artifact-dir'></a><nobr>`--output-artifact-dir <str>`</nobr><br><nobr>`--artifact-dir <str>`</nobr> | The directory to store all the (output) artifacts generated by AIPerf. <br>**Default:** `artifacts` |
| <a id='profile-export-file'></a><nobr>`--profile-export-file <str>`</nobr> | The file to store the profile export in JSONL format. <br>**Default:** `profile_export.jsonl` |

### Tokenizer Options

| Option | Description |
|--------|-------------|
| <a id='tokenizer'></a><nobr>`--tokenizer <str>`</nobr> | The HuggingFace tokenizer to use to interpret token metrics from prompts and responses. The value can be the name of a tokenizer or the filepath of the tokenizer. The default value is the model name. |
| <a id='tokenizer-revision'></a><nobr>`--tokenizer-revision <str>`</nobr> | The specific model version to use. It can be a branch name, tag name, or commit ID. <br>**Default:** `main` |
| <a id='tokenizer-trust-remote-code'></a><nobr>`--tokenizer-trust-remote-code`</nobr> | Allows custom tokenizer to be downloaded and executed. This carries security risks and should only be used for repositories you trust. This is only necessary for custom tokenizers stored in HuggingFace Hub. |

### Load Generator Options

| Option | Description |
|--------|-------------|
| <a id='benchmark-duration'></a><nobr>`--benchmark-duration <float>`</nobr> | The duration in seconds for benchmarking. |
| <a id='benchmark-grace-period'></a><nobr>`--benchmark-grace-period <float>`</nobr> | The grace period in seconds to wait for responses after benchmark duration ends. Only applies when --benchmark-duration is set. Responses received within this period are included in metrics. <br>**Default:** `30.0` |
| <a id='concurrency'></a><nobr>`--concurrency <int>`</nobr> | The concurrency value to benchmark. |
| <a id='request-rate'></a><nobr>`--request-rate <float>`</nobr> | Sets the request rate for the load generated by AIPerf. Unit: requests/second. |
| <a id='request-rate-mode'></a><nobr>`--request-rate-mode <str>`</nobr> | Sets the request rate mode for the load generated by AIPerf. Valid values: constant, poisson. constant: Generate requests at a fixed rate. poisson: Generate requests using a poisson distribution. <br>**Default:** `poisson` |
| <a id='request-count'></a><nobr>`--request-count <int>`</nobr><br><nobr>`--num-requests <int>`</nobr> | The number of requests to use for measurement. <br>**Default:** `10` |
| <a id='warmup-request-count'></a><nobr>`--warmup-request-count <int>`</nobr><br><nobr>`--num-warmup-requests <int>`</nobr> | The number of warmup requests to send before benchmarking. <br>**Default:** `0` |
| <a id='request-cancellation-rate'></a><nobr>`--request-cancellation-rate <float>`</nobr> | The percentage of requests to cancel. <br>**Default:** `0.0` |
| <a id='request-cancellation-delay'></a><nobr>`--request-cancellation-delay <float>`</nobr> | The delay in seconds before cancelling requests. This is used when --request-cancellation-rate is greater than 0. <br>**Default:** `0.0` |

### ZMQ Communication Options

| Option | Description |
|--------|-------------|
| <a id='zmq-host'></a><nobr>`--zmq-host <str>`</nobr> | Host address for TCP connections. <br>**Default:** `127.0.0.1` |
| <a id='zmq-ipc-path'></a><nobr>`--zmq-ipc-path <str>`</nobr> | Path for IPC sockets. |

### Workers Options

| Option | Description |
|--------|-------------|
| <a id='workers-max'></a><nobr>`--workers-max <int>`</nobr><br><nobr>`--max-workers <int>`</nobr> | Maximum number of workers to create. If not specified, the number of workers will be determined by the formula `min(concurrency, (num CPUs * 0.75) - 1)`,  with a default max cap of `32`. Any value provided will still be capped by the concurrency value (if specified), but not by the max cap. |

### Service Options

| Option | Description |
|--------|-------------|
| <a id='log-level'></a><nobr>`--log-level <str>`</nobr> | Logging level. <br>**Choices:** `TRACE`, `DEBUG`, `INFO`, `NOTICE`, `WARNING`, `SUCCESS`, `ERROR`, `CRITICAL` <br>**Default:** `INFO` |
| <a id='verbose'></a><nobr>`-v`</nobr><br><nobr>`--verbose`</nobr> | Equivalent to --log-level DEBUG. Enables more verbose logging output, but lacks some raw message logging. |
| <a id='extra-verbose'></a><nobr>`-vv`</nobr><br><nobr>`--extra-verbose`</nobr> | Equivalent to --log-level TRACE. Enables the most verbose logging output possible. |
| <a id='record-processor-service-count'></a><nobr>`--record-processor-service-count <int>`</nobr><br><nobr>`--record-processors <int>`</nobr> | Number of services to spawn for processing records. The higher the request rate, the more services should be spawned in order to keep up with the incoming records. If not specified, the number of services will be automatically determined based on the worker count. |
| <a id='ui-type'></a><nobr>`--ui-type <str>`</nobr><br><nobr>`--ui <str>`</nobr> | Type of UI to use. <br>**Choices:** `dashboard`, `simple`, `none` <br>**Default:** `dashboard` |
