<!--
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->

## Command Line Options

### Endpoint Options

| Option | Description |
|--------|-------------|
| <a id='model-names'></a>`-m <list>`<br>`--model-names <list>`<br>`--model <list>` | Model name(s) to be benchmarked. Can be a comma-separated list or a single model name. |
| <a id='model-selection-strategy'></a>`--model-selection-strategy <str>` | When multiple models are specified, this is how a specific model should be assigned to a prompt. round_robin: nth prompt in the list gets assigned to n-mod len(models). random: assignment is uniformly random. <br>**Choices:** `round_robin`, `random` <br>**Default:** `round_robin` |
| <a id='custom-endpoint'></a>`--custom-endpoint <str>`<br>`--endpoint <str>` | Set a custom endpoint that differs from the OpenAI defaults. |
| <a id='endpoint-type'></a>`--endpoint-type <str>` | The endpoint type to send requests to on the server. <br>**Choices:** `chat`, `completions`, `embeddings`, `rankings` <br>**Default:** `chat` |
| <a id='streaming'></a>`--streaming` | An option to enable the use of the streaming API. |
| <a id='url'></a>`-u <str>`<br>`--url <str>` | URL of the endpoint to target for benchmarking. <br>**Default:** `localhost:8000` |
| <a id='request-timeout-seconds'></a>`--request-timeout-seconds <float>` | The timeout in floating-point seconds for each request to the endpoint. <br>**Default:** `600.0` |
| <a id='api-key'></a>`--api-key <str>` | The API key to use for the endpoint. If provided, it will be sent with every request as a header: `Authorization: Bearer <api_key>`. |

### Input Options

| Option | Description |
|--------|-------------|
| <a id='extra-inputs'></a>`--extra-inputs <list>` | Provide additional inputs to include with every request. Inputs should be in an 'input_name:value' format. Alternatively, a string representing a json formatted dict can be provided. <br>**Default:** `[]` |
| <a id='header'></a>`-H <list>`<br>`--header <list>` | Adds a custom header to the requests. Headers must be specified as 'Header:Value' pairs. Alternatively, a string representing a json formatted dict can be provided. <br>**Default:** `[]` |
| <a id='input-file'></a>`--input-file <str>` | The file or directory path that contains the dataset to use for profiling. This parameter is used in conjunction with the `custom_dataset_type` parameter to support different types of user provided datasets. |
| <a id='fixed-schedule'></a>`--fixed-schedule` | Specifies to run a fixed schedule of requests. This is normally inferred from the --input-file parameter, but can be set manually here. |
| <a id='fixed-schedule-auto-offset'></a>`--fixed-schedule-auto-offset` | Specifies to automatically offset the timestamps in the fixed schedule, such that the first timestamp is considered 0, and the rest are shifted accordingly. If disabled, the timestamps will be assumed to be relative to 0. |
| <a id='fixed-schedule-start-offset'></a>`--fixed-schedule-start-offset <int>` | Specifies the offset in milliseconds to start the fixed schedule at. By default, the schedule starts at 0, but this option can be used to start at a reference point further in the schedule. This option cannot be used in conjunction with the --fixed-schedule-auto-offset. The schedule will include any requests at the start offset. |
| <a id='fixed-schedule-end-offset'></a>`--fixed-schedule-end-offset <int>` | Specifies the offset in milliseconds to end the fixed schedule at. By default, the schedule ends at the last timestamp in the trace dataset, but this option can be used to only run a subset of the trace. The schedule will include any requests at the end offset. |
| <a id='public-dataset'></a>`--public-dataset <str>` | The public dataset to use for the requests. <br>**Choices:** `sharegpt` |
| <a id='custom-dataset-type'></a>`--custom-dataset-type <str>` | The type of custom dataset to use. This parameter is used in conjunction with the --input-file parameter. [choices: single_turn, multi_turn, random_pool, mooncake_trace]. |
| <a id='random-seed'></a>`--random-seed <int>` | The seed used to generate random values. Set to some value to make the synthetic data generation deterministic. It will use system default if not provided. |
| <a id='goodput'></a>`--goodput <str>` | Specify service level objectives (SLOs) for goodput as space-separated 'KEY:VALUE' pairs, where KEY is a metric tag and VALUE is a number in the metric’s display unit (falls back to its base unit if no display unit is defined). Examples: 'request_latency:250' (ms), 'inter_token_latency:10' (ms), `output_token_throughput_per_user:600` (tokens/s). Only metrics applicable to the current endpoint/config are considered. For more context on the definition of goodput, refer to DistServe paper: https://arxiv.org/pdf/2401.09670 and the blog: https://hao-ai-lab.github.io/blogs/distserve. |

### Audio Input Options

| Option | Description |
|--------|-------------|
| <a id='audio-batch-size'></a>`--audio-batch-size <int>`<br>`--batch-size-audio <int>` | The batch size of audio requests AIPerf should send. This is currently supported with the OpenAI `chat` endpoint type. <br>**Default:** `1` |
| <a id='audio-length-mean'></a>`--audio-length-mean <float>` | The mean length of the audio in seconds. <br>**Default:** `0.0` |
| <a id='audio-length-stddev'></a>`--audio-length-stddev <float>` | The standard deviation of the length of the audio in seconds. <br>**Default:** `0.0` |
| <a id='audio-format'></a>`--audio-format <str>` | The format of the audio files (wav or mp3). <br>**Choices:** `wav`, `mp3` <br>**Default:** `wav` |
| <a id='audio-depths'></a>`--audio-depths <list>` | A list of audio bit depths to randomly select from in bits. <br>**Default:** `[16]` |
| <a id='audio-sample-rates'></a>`--audio-sample-rates <list>` | A list of audio sample rates to randomly select from in kHz. Common sample rates are 16, 44.1, 48, 96, etc. <br>**Default:** `[16.0]` |
| <a id='audio-num-channels'></a>`--audio-num-channels <int>` | The number of audio channels to use for the audio data generation. <br>**Default:** `1` |

### Image Input Options

| Option | Description |
|--------|-------------|
| <a id='image-width-mean'></a>`--image-width-mean <float>` | The mean width of images when generating synthetic image data. <br>**Default:** `0.0` |
| <a id='image-width-stddev'></a>`--image-width-stddev <float>` | The standard deviation of width of images when generating synthetic image data. <br>**Default:** `0.0` |
| <a id='image-height-mean'></a>`--image-height-mean <float>` | The mean height of images when generating synthetic image data. <br>**Default:** `0.0` |
| <a id='image-height-stddev'></a>`--image-height-stddev <float>` | The standard deviation of height of images when generating synthetic image data. <br>**Default:** `0.0` |
| <a id='image-batch-size'></a>`--image-batch-size <int>`<br>`--batch-size-image <int>` | The image batch size of the requests AIPerf should send. This is currently supported with the image retrieval endpoint type. <br>**Default:** `1` |
| <a id='image-format'></a>`--image-format <str>` | The compression format of the images. <br>**Choices:** `png`, `jpeg`, `random` <br>**Default:** `png` |

### Prompt Options

| Option | Description |
|--------|-------------|
| <a id='prompt-batch-size'></a>`-b <int>`<br>`--prompt-batch-size <int>`<br>`--batch-size-text <int>`<br>`--batch-size <int>` | The batch size of text requests AIPerf should send. This is currently supported with the embeddings and rankings endpoint types. <br>**Default:** `1` |

### Input Sequence Length (ISL) Options

| Option | Description |
|--------|-------------|
| <a id='prompt-input-tokens-mean'></a>`--prompt-input-tokens-mean <int>`<br>`--synthetic-input-tokens-mean <int>`<br>`--isl <int>` | The mean of number of tokens in the generated prompts when using synthetic data. <br>**Default:** `550` |
| <a id='prompt-input-tokens-stddev'></a>`--prompt-input-tokens-stddev <float>`<br>`--synthetic-input-tokens-stddev <float>`<br>`--isl-stddev <float>` | The standard deviation of number of tokens in the generated prompts when using synthetic data. <br>**Default:** `0.0` |
| <a id='prompt-input-tokens-block-size'></a>`--prompt-input-tokens-block-size <int>`<br>`--synthetic-input-tokens-block-size <int>`<br>`--isl-block-size <int>` | The block size of the prompt. <br>**Default:** `512` |
| <a id='seq-dist'></a>`--seq-dist <str>`<br>`--sequence-distribution <str>` | Sequence length distribution specification for varying ISL/OSL pairs. |

### Output Sequence Length (OSL) Options

| Option | Description |
|--------|-------------|
| <a id='prompt-output-tokens-mean'></a>`--prompt-output-tokens-mean <int>`<br>`--output-tokens-mean <int>`<br>`--osl <int>` | The mean number of tokens in each output. |
| <a id='prompt-output-tokens-stddev'></a>`--prompt-output-tokens-stddev <float>`<br>`--output-tokens-stddev <float>`<br>`--osl-stddev <float>` | The standard deviation of the number of tokens in each output. <br>**Default:** `0` |

### Prefix Prompt Options

| Option | Description |
|--------|-------------|
| <a id='prompt-prefix-pool-size'></a>`--prompt-prefix-pool-size <int>`<br>`--prefix-prompt-pool-size <int>`<br>`--num-prefix-prompts <int>` | The total size of the prefix prompt pool to select prefixes from. If this value is not zero, these are prompts that are prepended to input prompts. This is useful for benchmarking models that use a K-V cache. <br>**Default:** `0` |
| <a id='prompt-prefix-length'></a>`--prompt-prefix-length <int>`<br>`--prefix-prompt-length <int>` | The number of tokens in each prefix prompt. This is only used if "num" is greater than zero. Note that due to the prefix and user prompts being concatenated, the number of tokens in the final prompt may be off by one. <br>**Default:** `0` |

### Conversation Input Options

| Option | Description |
|--------|-------------|
| <a id='conversation-num'></a>`--conversation-num <int>`<br>`--num-conversations <int>`<br>`--num-sessions <int>`<br>`--num-dataset-entries <int>` | The total number of unique conversations to generate. Each conversation represents a single request session between client and server. Supported on synthetic mode and the custom random_pool dataset. The number of conversations  will be used to determine the number of entries in both the custom random_pool and synthetic  datasets and will be reused until benchmarking is complete. <br>**Default:** `100` |
| <a id='conversation-turn-mean'></a>`--conversation-turn-mean <int>`<br>`--session-turns-mean <int>` | The mean number of turns within a conversation. <br>**Default:** `1` |
| <a id='conversation-turn-stddev'></a>`--conversation-turn-stddev <int>`<br>`--session-turns-stddev <int>` | The standard deviation of the number of turns within a conversation. <br>**Default:** `0` |
| <a id='conversation-turn-delay-mean'></a>`--conversation-turn-delay-mean <float>`<br>`--session-turn-delay-mean <float>` | The mean delay between turns within a conversation in milliseconds. <br>**Default:** `0.0` |
| <a id='conversation-turn-delay-stddev'></a>`--conversation-turn-delay-stddev <float>`<br>`--session-turn-delay-stddev <float>` | The standard deviation of the delay between turns  within a conversation in milliseconds. <br>**Default:** `0.0` |
| <a id='conversation-turn-delay-ratio'></a>`--conversation-turn-delay-ratio <float>`<br>`--session-delay-ratio <float>` | A ratio to scale multi-turn delays. <br>**Default:** `1.0` |

### Output Options

| Option | Description |
|--------|-------------|
| <a id='output-artifact-dir'></a>`--output-artifact-dir <str>`<br>`--artifact-dir <str>` | The directory to store all the (output) artifacts generated by AIPerf. <br>**Default:** `artifacts` |
| <a id='profile-export-file'></a>`--profile-export-file <str>` | The file to store the profile export in JSONL format. <br>**Default:** `profile_export.jsonl` |

### Tokenizer Options

| Option | Description |
|--------|-------------|
| <a id='tokenizer'></a>`--tokenizer <str>` | The HuggingFace tokenizer to use to interpret token metrics from prompts and responses. The value can be the name of a tokenizer or the filepath of the tokenizer. The default value is the model name. |
| <a id='tokenizer-revision'></a>`--tokenizer-revision <str>` | The specific model version to use. It can be a branch name, tag name, or commit ID. <br>**Default:** `main` |
| <a id='tokenizer-trust-remote-code'></a>`--tokenizer-trust-remote-code` | Allows custom tokenizer to be downloaded and executed. This carries security risks and should only be used for repositories you trust. This is only necessary for custom tokenizers stored in HuggingFace Hub. |

### Load Generator Options

| Option | Description |
|--------|-------------|
| <a id='benchmark-duration'></a>`--benchmark-duration <float>` | The duration in seconds for benchmarking. |
| <a id='benchmark-grace-period'></a>`--benchmark-grace-period <float>` | The grace period in seconds to wait for responses after benchmark duration ends. Only applies when --benchmark-duration is set. Responses received within this period are included in metrics. <br>**Default:** `30.0` |
| <a id='concurrency'></a>`--concurrency <int>` | The concurrency value to benchmark. |
| <a id='request-rate'></a>`--request-rate <float>` | Sets the request rate for the load generated by AIPerf. Unit: requests/second. |
| <a id='request-rate-mode'></a>`--request-rate-mode <str>` | Sets the request rate mode for the load generated by AIPerf. Valid values: constant, poisson. constant: Generate requests at a fixed rate. poisson: Generate requests using a poisson distribution. <br>**Default:** `poisson` |
| <a id='request-count'></a>`--request-count <int>`<br>`--num-requests <int>` | The number of requests to use for measurement. <br>**Default:** `10` |
| <a id='warmup-request-count'></a>`--warmup-request-count <int>`<br>`--num-warmup-requests <int>` | The number of warmup requests to send before benchmarking. <br>**Default:** `0` |
| <a id='request-cancellation-rate'></a>`--request-cancellation-rate <float>` | The percentage of requests to cancel. <br>**Default:** `0.0` |
| <a id='request-cancellation-delay'></a>`--request-cancellation-delay <float>` | The delay in seconds before cancelling requests. This is used when --request-cancellation-rate is greater than 0. <br>**Default:** `0.0` |

### ZMQ Communication Options

| Option | Description |
|--------|-------------|
| <a id='zmq-host'></a>`--zmq-host <str>` | Host address for TCP connections. <br>**Default:** `127.0.0.1` |
| <a id='zmq-ipc-path'></a>`--zmq-ipc-path <str>` | Path for IPC sockets. |

### Workers Options

| Option | Description |
|--------|-------------|
| <a id='workers-max'></a>`--workers-max <int>`<br>`--max-workers <int>` | Maximum number of workers to create. If not specified, the number of workers will be determined by the formula `min(concurrency, (num CPUs * 0.75) - 1)`,  with a default max cap of `32`. Any value provided will still be capped by the concurrency value (if specified), but not by the max cap. |

### Service Options

| Option | Description |
|--------|-------------|
| <a id='log-level'></a>`--log-level <str>` | Logging level. <br>**Choices:** `TRACE`, `DEBUG`, `INFO`, `NOTICE`, `WARNING`, `SUCCESS`, `ERROR`, `CRITICAL` <br>**Default:** `INFO` |
| <a id='verbose'></a>`-v`<br>`--verbose` | Equivalent to --log-level DEBUG. Enables more verbose logging output, but lacks some raw message logging. |
| <a id='extra-verbose'></a>`-vv`<br>`--extra-verbose` | Equivalent to --log-level TRACE. Enables the most verbose logging output possible. |
| <a id='record-processor-service-count'></a>`--record-processor-service-count <int>`<br>`--record-processors <int>` | Number of services to spawn for processing records. The higher the request rate, the more services should be spawned in order to keep up with the incoming records. If not specified, the number of services will be automatically determined based on the worker count. |
| <a id='ui-type'></a>`--ui-type <str>`<br>`--ui <str>` | Type of UI to use. <br>**Choices:** `dashboard`, `simple`, `none` <br>**Default:** `dashboard` |
