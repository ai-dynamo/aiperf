<!--
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->

## Command Line Options

### Endpoint Options

#### Model Names _(Required)_

`-m <list>` | `--model-names <list>` | `--model <list>`

Model name(s) to be benchmarked. Can be a comma-separated list or a single model name.

#### Model Selection Strategy

`--model-selection-strategy <str>`

When multiple models are specified, this is how a specific model should be assigned to a prompt.
round_robin: nth prompt in the list gets assigned to n-mod len(models).
random: assignment is uniformly random. (Choices: `round_robin`, `random`). (Default: `round_robin`).

#### Custom Endpoint

`--custom-endpoint <str>` | `--endpoint <str>`

Set a custom endpoint that differs from the OpenAI defaults.

#### Endpoint Type

`--endpoint-type <str>`

The endpoint type to send requests to on the server. (Choices: `chat`, `completions`, `embeddings`, `rankings`, `responses`). (Default: `chat`).

#### Streaming

`--streaming`

An option to enable the use of the streaming API.

#### Url

`-u <str>` | `--url <str>`

URL of the endpoint to target for benchmarking. (Default: `localhost:8000`).

#### Request Timeout Seconds

`--request-timeout-seconds <float>`

The timeout in floating-point seconds for each request to the endpoint. (Default: `600.0`).

#### Api Key

`--api-key <str>`

The API key to use for the endpoint. If provided, it will be sent with every request as a header: `Authorization: Bearer <api_key>`.

### Input Options

#### Extra Inputs

`--extra-inputs <list>`

Provide additional inputs to include with every request.
Inputs should be in an 'input_name:value' format.
Alternatively, a string representing a json formatted dict can be provided. (Default: `[]`).

#### Header

`-H <list>` | `--header <list>`

Adds a custom header to the requests.
Headers must be specified as 'Header:Value' pairs.
Alternatively, a string representing a json formatted dict can be provided. (Default: `[]`).

#### Input File

`--input-file <str>`

The file or directory path that contains the dataset to use for profiling.
This parameter is used in conjunction with the `custom_dataset_type` parameter
to support different types of user provided datasets.

#### Fixed Schedule

`--fixed-schedule`

Specifies to run a fixed schedule of requests. This is normally inferred from the --input-file parameter, but can be set manually here.

#### Fixed Schedule Auto Offset

`--fixed-schedule-auto-offset`

Specifies to automatically offset the timestamps in the fixed schedule, such that the first timestamp is considered 0, and the rest are shifted accordingly. If disabled, the timestamps will be assumed to be relative to 0.

#### Fixed Schedule Start Offset

`--fixed-schedule-start-offset <int>`

Specifies the offset in milliseconds to start the fixed schedule at. By default, the schedule starts at 0, but this option can be used to start at a reference point further in the schedule. This option cannot be used in conjunction with the --fixed-schedule-auto-offset. The schedule will include any requests at the start offset.

#### Fixed Schedule End Offset

`--fixed-schedule-end-offset <int>`

Specifies the offset in milliseconds to end the fixed schedule at. By default, the schedule ends at the last timestamp in the trace dataset, but this option can be used to only run a subset of the trace. The schedule will include any requests at the end offset.

#### Public Dataset

`--public-dataset <str>`

The public dataset to use for the requests. (Choices: `sharegpt`).

#### Custom Dataset Type

`--custom-dataset-type <str>`

The type of custom dataset to use.
This parameter is used in conjunction with the --input-file parameter.
[choices: single_turn, multi_turn, random_pool, mooncake_trace].

#### Random Seed

`--random-seed <int>`

The seed used to generate random values.
Set to some value to make the synthetic data generation deterministic.
It will use system default if not provided.

### Audio Input Options

#### Audio Batch Size

`--audio-batch-size <int>` | `--batch-size-audio <int>`

The batch size of audio requests AIPerf should send.
This is currently supported with the OpenAI `chat` endpoint type. (Default: `1`).

#### Audio Length Mean

`--audio-length-mean <float>`

The mean length of the audio in seconds. (Default: `0.0`).

#### Audio Length Stddev

`--audio-length-stddev <float>`

The standard deviation of the length of the audio in seconds. (Default: `0.0`).

#### Audio Format

`--audio-format <str>`

The format of the audio files (wav or mp3). (Choices: `wav`, `mp3`). (Default: `wav`).

#### Audio Depths

`--audio-depths <list>`

A list of audio bit depths to randomly select from in bits. (Default: `[16]`).

#### Audio Sample Rates

`--audio-sample-rates <list>`

A list of audio sample rates to randomly select from in kHz.
Common sample rates are 16, 44.1, 48, 96, etc. (Default: `[16.0]`).

#### Audio Num Channels

`--audio-num-channels <int>`

The number of audio channels to use for the audio data generation. (Default: `1`).

### Image Input Options

#### Image Width Mean

`--image-width-mean <float>`

The mean width of images when generating synthetic image data. (Default: `0.0`).

#### Image Width Stddev

`--image-width-stddev <float>`

The standard deviation of width of images when generating synthetic image data. (Default: `0.0`).

#### Image Height Mean

`--image-height-mean <float>`

The mean height of images when generating synthetic image data. (Default: `0.0`).

#### Image Height Stddev

`--image-height-stddev <float>`

The standard deviation of height of images when generating synthetic image data. (Default: `0.0`).

#### Image Batch Size

`--image-batch-size <int>` | `--batch-size-image <int>`

The image batch size of the requests AIPerf should send.
This is currently supported with the image retrieval endpoint type. (Default: `1`).

#### Image Format

`--image-format <str>`

The compression format of the images. (Choices: `png`, `jpeg`, `random`). (Default: `png`).

### Prompt Options

#### Prompt Batch Size

`-b <int>` | `--prompt-batch-size <int>` | `--batch-size-text <int>` | `--batch-size <int>`

The batch size of text requests AIPerf should send.
This is currently supported with the embeddings and rankings endpoint types. (Default: `1`).

### Input Sequence Length (ISL) Options

#### Prompt Input Tokens Mean

`--prompt-input-tokens-mean <int>` | `--synthetic-input-tokens-mean <int>` | `--isl <int>`

The mean of number of tokens in the generated prompts when using synthetic data. (Default: `550`).

#### Prompt Input Tokens Stddev

`--prompt-input-tokens-stddev <float>` | `--synthetic-input-tokens-stddev <float>` | `--isl-stddev <float>`

The standard deviation of number of tokens in the generated prompts when using synthetic data. (Default: `0.0`).

#### Prompt Input Tokens Block Size

`--prompt-input-tokens-block-size <int>` | `--synthetic-input-tokens-block-size <int>` | `--isl-block-size <int>`

The block size of the prompt. (Default: `512`).

### Output Sequence Length (OSL) Options

#### Prompt Output Tokens Mean

`--prompt-output-tokens-mean <int>` | `--output-tokens-mean <int>` | `--osl <int>`

The mean number of tokens in each output.

#### Prompt Output Tokens Stddev

`--prompt-output-tokens-stddev <float>` | `--output-tokens-stddev <float>` | `--osl-stddev <float>`

The standard deviation of the number of tokens in each output. (Default: `0`).

### Prefix Prompt Options

#### Prompt Prefix Pool Size

`--prompt-prefix-pool-size <int>` | `--prefix-prompt-pool-size <int>` | `--num-prefix-prompts <int>`

The total size of the prefix prompt pool to select prefixes from.
If this value is not zero, these are prompts that are prepended to input prompts.
This is useful for benchmarking models that use a K-V cache. (Default: `0`).

#### Prompt Prefix Length

`--prompt-prefix-length <int>` | `--prefix-prompt-length <int>`

The number of tokens in each prefix prompt.
This is only used if "num" is greater than zero.
Note that due to the prefix and user prompts being concatenated,
the number of tokens in the final prompt may be off by one. (Default: `0`).

### Conversation Input Options

#### Conversation Num

`--conversation-num <int>` | `--num-conversations <int>` | `--num-sessions <int>` | `--num-dataset-entries <int>`

The total number of unique conversations to generate.
Each conversation represents a single request session between client and server.
Supported on synthetic mode and the custom random_pool dataset. The number of conversations
will be used to determine the number of entries in both the custom random_pool and synthetic
datasets and will be reused until benchmarking is complete. (Default: `100`).

#### Conversation Turn Mean

`--conversation-turn-mean <int>` | `--session-turns-mean <int>`

The mean number of turns within a conversation. (Default: `1`).

#### Conversation Turn Stddev

`--conversation-turn-stddev <int>` | `--session-turns-stddev <int>`

The standard deviation of the number of turns within a conversation. (Default: `0`).

#### Conversation Turn Delay Mean

`--conversation-turn-delay-mean <float>` | `--session-turn-delay-mean <float>`

The mean delay between turns within a conversation in milliseconds. (Default: `0.0`).

#### Conversation Turn Delay Stddev

`--conversation-turn-delay-stddev <float>` | `--session-turn-delay-stddev <float>`

The standard deviation of the delay between turns
within a conversation in milliseconds. (Default: `0.0`).

#### Conversation Turn Delay Ratio

`--conversation-turn-delay-ratio <float>` | `--session-delay-ratio <float>`

A ratio to scale multi-turn delays. (Default: `1.0`).

### Output Options

#### Output Artifact Dir

`--output-artifact-dir <str>` | `--artifact-dir <str>`

The directory to store all the (output) artifacts generated by AIPerf. (Default: `artifacts`).

### Tokenizer Options

#### Tokenizer

`--tokenizer <str>`

The HuggingFace tokenizer to use to interpret token metrics from prompts and responses.
The value can be the name of a tokenizer or the filepath of the tokenizer.
The default value is the model name.

#### Tokenizer Revision

`--tokenizer-revision <str>`

The specific model version to use.
It can be a branch name, tag name, or commit ID. (Default: `main`).

#### Tokenizer Trust Remote Code

`--tokenizer-trust-remote-code`

Allows custom tokenizer to be downloaded and executed.
This carries security risks and should only be used for repositories you trust.
This is only necessary for custom tokenizers stored in HuggingFace Hub.

### Load Generator Options

#### Benchmark Duration

`--benchmark-duration <float>`

The duration in seconds for benchmarking.

#### Benchmark Grace Period

`--benchmark-grace-period <float>`

The grace period in seconds to wait for responses after benchmark duration ends. Only applies when --benchmark-duration is set. Responses received within this period are included in metrics. (Default: `30.0`).

#### Concurrency

`--concurrency <int>`

The concurrency value to benchmark.

#### Request Rate

`--request-rate <float>`

Sets the request rate for the load generated by AIPerf. Unit: requests/second.

#### Request Rate Mode

`--request-rate-mode <str>`

Sets the request rate mode for the load generated by AIPerf. Valid values: constant, poisson.
constant: Generate requests at a fixed rate.
poisson: Generate requests using a poisson distribution. (Default: `poisson`).

#### Request Count

`--request-count <int>` | `--num-requests <int>`

The number of requests to use for measurement. (Default: `10`).

#### Warmup Request Count

`--warmup-request-count <int>` | `--num-warmup-requests <int>`

The number of warmup requests to send before benchmarking. (Default: `0`).

#### Request Cancellation Rate

`--request-cancellation-rate <float>`

The percentage of requests to cancel. (Default: `0.0`).

#### Request Cancellation Delay

`--request-cancellation-delay <float>`

The delay in seconds before cancelling requests. This is used when --request-cancellation-rate is greater than 0. (Default: `0.0`).

### ZMQ Communication Options

#### Zmq Host

`--zmq-host <str>`

Host address for TCP connections. (Default: `127.0.0.1`).

#### Zmq Ipc Path

`--zmq-ipc-path <str>`

Path for IPC sockets.

### Workers Options

#### Workers Max

`--workers-max <int>` | `--max-workers <int>`

Maximum number of workers to create. If not specified, the number of workers will be determined by the formula `min(concurrency, (num CPUs * 0.75) - 1)`,  with a default max cap of `32`. Any value provided will still be capped by the concurrency value (if specified), but not by the max cap.

### Service Options

#### Log Level

`--log-level <str>`

Logging level. (Choices: `TRACE`, `DEBUG`, `INFO`, `NOTICE`, `WARNING`, `SUCCESS`, `ERROR`, `CRITICAL`). (Default: `INFO`).

#### Verbose

`-v` | `--verbose`

Equivalent to --log-level DEBUG. Enables more verbose logging output, but lacks some raw message logging.

#### Extra Verbose

`-vv` | `--extra-verbose`

Equivalent to --log-level TRACE. Enables the most verbose logging output possible.

#### Record Processor Service Count

`--record-processor-service-count <int>` | `--record-processors <int>`

Number of services to spawn for processing records. The higher the request rate, the more services should be spawned in order to keep up with the incoming records. If not specified, the number of services will be automatically determined based on the worker count.

#### Ui Type

`--ui-type <str>` | `--ui <str>`

Type of UI to use. (Choices: `dashboard`, `simple`, `none`). (Default: `dashboard`).
