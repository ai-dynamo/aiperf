<!--
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
## Command Line Options

### Endpoint Options

##### MODEL-NAMES

`-m <list>` | `--model-names <list>` | `--model <list>`

Model name(s) to be benchmarked. Can be a comma-separated list or a single model name.

##### MODEL-SELECTION-STRATEGY

`--model-selection-strategy <str>`

When multiple models are specified, this is how a specific model should be assigned to a prompt.
round_robin: nth prompt in the list gets assigned to n-mod len(models).
random: assignment is uniformly random

##### CUSTOM-ENDPOINT

`--custom-endpoint <str>` | `--endpoint <str>`

Set a custom endpoint that differs from the OpenAI defaults.

##### ENDPOINT-TYPE

`--endpoint-type <str>`

The endpoint type to send requests to on the server.

##### STREAMING

`--streaming`

An option to enable the use of the streaming API.

##### URL

`-u <str>` | `--url <str>`

URL of the endpoint to target for benchmarking.

##### REQUEST-TIMEOUT-SECONDS

`--request-timeout-seconds <float>`

The timeout in floating-point seconds for each request to the endpoint.

##### API-KEY

`--api-key <str>`

The API key to use for the endpoint. If provided, it will be sent with every request as a header: `Authorization: Bearer <api_key>`.

### Input Options

##### EXTRA-INPUTS

`--extra-inputs <list>`

Provide additional inputs to include with every request.
Inputs should be in an 'input_name:value' format.
Alternatively, a string representing a json formatted dict can be provided.

##### HEADER

`-H <list>` | `--header <list>`

Adds a custom header to the requests.
Headers must be specified as 'Header:Value' pairs.
Alternatively, a string representing a json formatted dict can be provided.

##### INPUT-FILE

`--input-file <str>`

The file or directory path that contains the dataset to use for profiling.
This parameter is used in conjunction with the `custom_dataset_type` parameter
to support different types of user provided datasets.

##### FIXED-SCHEDULE

`--fixed-schedule`

Specifies to run a fixed schedule of requests. This is normally inferred from the --input-file parameter, but can be set manually here.

##### FIXED-SCHEDULE-AUTO-OFFSET

`--fixed-schedule-auto-offset`

Specifies to automatically offset the timestamps in the fixed schedule, such that the first timestamp is considered 0, and the rest are shifted accordingly. If disabled, the timestamps will be assumed to be relative to 0.

##### FIXED-SCHEDULE-START-OFFSET

`--fixed-schedule-start-offset <int>`

Specifies the offset in milliseconds to start the fixed schedule at. By default, the schedule starts at 0, but this option can be used to start at a reference point further in the schedule. This option cannot be used in conjunction with the --fixed-schedule-auto-offset. The schedule will include any requests at the start offset.

##### FIXED-SCHEDULE-END-OFFSET

`--fixed-schedule-end-offset <int>`

Specifies the offset in milliseconds to end the fixed schedule at. By default, the schedule ends at the last timestamp in the trace dataset, but this option can be used to only run a subset of the trace. The schedule will include any requests at the end offset.

##### PUBLIC-DATASET

`--public-dataset <str>`

The public dataset to use for the requests.

##### CUSTOM-DATASET-TYPE

`--custom-dataset-type <str>`

The type of custom dataset to use.
This parameter is used in conjunction with the --input-file parameter.
[choices: single_turn, multi_turn, random_pool, mooncake_trace]

##### RANDOM-SEED

`--random-seed <int>`

The seed used to generate random values.
Set to some value to make the synthetic data generation deterministic.
It will use system default if not provided.

### Audio Input Options

##### AUDIO-BATCH-SIZE

`--audio-batch-size <int>` | `--batch-size-audio <int>`

The batch size of audio requests AIPerf should send.
This is currently supported with the OpenAI `chat` endpoint type

##### AUDIO-LENGTH-MEAN

`--audio-length-mean <float>`

The mean length of the audio in seconds.

##### AUDIO-LENGTH-STDDEV

`--audio-length-stddev <float>`

The standard deviation of the length of the audio in seconds.

##### AUDIO-FORMAT

`--audio-format <str>`

The format of the audio files (wav or mp3).

##### AUDIO-DEPTHS

`--audio-depths <list>`

A list of audio bit depths to randomly select from in bits.

##### AUDIO-SAMPLE-RATES

`--audio-sample-rates <list>`

A list of audio sample rates to randomly select from in kHz.
Common sample rates are 16, 44.1, 48, 96, etc.

##### AUDIO-NUM-CHANNELS

`--audio-num-channels <int>`

The number of audio channels to use for the audio data generation.

### Image Input Options

##### IMAGE-WIDTH-MEAN

`--image-width-mean <float>`

The mean width of images when generating synthetic image data.

##### IMAGE-WIDTH-STDDEV

`--image-width-stddev <float>`

The standard deviation of width of images when generating synthetic image data.

##### IMAGE-HEIGHT-MEAN

`--image-height-mean <float>`

The mean height of images when generating synthetic image data.

##### IMAGE-HEIGHT-STDDEV

`--image-height-stddev <float>`

The standard deviation of height of images when generating synthetic image data.

##### IMAGE-BATCH-SIZE

`--image-batch-size <int>` | `--batch-size-image <int>`

The image batch size of the requests AIPerf should send.
This is currently supported with the image retrieval endpoint type.

##### IMAGE-FORMAT

`--image-format <str>`

The compression format of the images.

### Prompt Options

##### PROMPT-BATCH-SIZE

`-b <int>` | `--prompt-batch-size <int>` | `--batch-size-text <int>` | `--batch-size <int>`

The batch size of text requests AIPerf should send.
This is currently supported with the embeddings and rankings endpoint types

### Input Sequence Length (ISL) Options

##### PROMPT-INPUT-TOKENS-MEAN

`--prompt-input-tokens-mean <int>` | `--synthetic-input-tokens-mean <int>` | `--isl <int>`

The mean of number of tokens in the generated prompts when using synthetic data.

##### PROMPT-INPUT-TOKENS-STDDEV

`--prompt-input-tokens-stddev <float>` | `--synthetic-input-tokens-stddev <float>` | `--isl-stddev <float>`

The standard deviation of number of tokens in the generated prompts when using synthetic data.

##### PROMPT-INPUT-TOKENS-BLOCK-SIZE

`--prompt-input-tokens-block-size <int>` | `--synthetic-input-tokens-block-size <int>` | `--isl-block-size <int>`

The block size of the prompt.

### Output Sequence Length (OSL) Options

##### PROMPT-OUTPUT-TOKENS-MEAN

`--prompt-output-tokens-mean <int>` | `--output-tokens-mean <int>` | `--osl <int>`

The mean number of tokens in each output.

##### PROMPT-OUTPUT-TOKENS-STDDEV

`--prompt-output-tokens-stddev <float>` | `--output-tokens-stddev <float>` | `--osl-stddev <float>`

The standard deviation of the number of tokens in each output.

### Prefix Prompt Options

##### PROMPT-PREFIX-POOL-SIZE

`--prompt-prefix-pool-size <int>` | `--prefix-prompt-pool-size <int>` | `--num-prefix-prompts <int>`

The total size of the prefix prompt pool to select prefixes from.
If this value is not zero, these are prompts that are prepended to input prompts.
This is useful for benchmarking models that use a K-V cache.

##### PROMPT-PREFIX-LENGTH

`--prompt-prefix-length <int>` | `--prefix-prompt-length <int>`

The number of tokens in each prefix prompt.
This is only used if "num" is greater than zero.
Note that due to the prefix and user prompts being concatenated,
the number of tokens in the final prompt may be off by one.

### Conversation Input Options

##### CONVERSATION-NUM

`--conversation-num <int>` | `--num-conversations <int>` | `--num-sessions <int>` | `--num-dataset-entries <int>`

The total number of unique conversations to generate.
Each conversation represents a single request session between client and server.
Supported on synthetic mode and the custom random_pool dataset. The number of conversations
will be used to determine the number of entries in both the custom random_pool and synthetic
datasets and will be reused until benchmarking is complete.

##### CONVERSATION-TURN-MEAN

`--conversation-turn-mean <int>` | `--session-turns-mean <int>`

The mean number of turns within a conversation.

##### CONVERSATION-TURN-STDDEV

`--conversation-turn-stddev <int>` | `--session-turns-stddev <int>`

The standard deviation of the number of turns within a conversation.

##### CONVERSATION-TURN-DELAY-MEAN

`--conversation-turn-delay-mean <float>` | `--session-turn-delay-mean <float>`

The mean delay between turns within a conversation in milliseconds.

##### CONVERSATION-TURN-DELAY-STDDEV

`--conversation-turn-delay-stddev <float>` | `--session-turn-delay-stddev <float>`

The standard deviation of the delay between turns
within a conversation in milliseconds.

##### CONVERSATION-TURN-DELAY-RATIO

`--conversation-turn-delay-ratio <float>` | `--session-delay-ratio <float>`

A ratio to scale multi-turn delays.

### Output Options

##### OUTPUT-ARTIFACT-DIR

`--output-artifact-dir <str>` | `--artifact-dir <str>`

The directory to store all the (output) artifacts generated by AIPerf.

### Tokenizer Options

##### TOKENIZER

`--tokenizer <str>`

The HuggingFace tokenizer to use to interpret token metrics from prompts and responses.
The value can be the name of a tokenizer or the filepath of the tokenizer.
The default value is the model name.

##### TOKENIZER-REVISION

`--tokenizer-revision <str>`

The specific model version to use.
It can be a branch name, tag name, or commit ID.

##### TOKENIZER-TRUST-REMOTE-CODE

`--tokenizer-trust-remote-code`

Allows custom tokenizer to be downloaded and executed.
This carries security risks and should only be used for repositories you trust.
This is only necessary for custom tokenizers stored in HuggingFace Hub.

### Load Generator Options

##### BENCHMARK-DURATION

`--benchmark-duration <float>`

The duration in seconds for benchmarking.

##### BENCHMARK-GRACE-PERIOD

`--benchmark-grace-period <float>`

The grace period in seconds to wait for responses after benchmark duration ends. Only applies when --benchmark-duration is set. Responses received within this period are included in metrics.

##### CONCURRENCY

`--concurrency <int>`

The concurrency value to benchmark.

##### REQUEST-RATE

`--request-rate <float>`

Sets the request rate for the load generated by AIPerf. Unit: requests/second

##### REQUEST-RATE-MODE

`--request-rate-mode <str>`

Sets the request rate mode for the load generated by AIPerf. Valid values: constant, poisson.
constant: Generate requests at a fixed rate.
poisson: Generate requests using a poisson distribution.

##### REQUEST-COUNT

`--request-count <int>` | `--num-requests <int>`

The number of requests to use for measurement.

##### WARMUP-REQUEST-COUNT

`--warmup-request-count <int>` | `--num-warmup-requests <int>`

The number of warmup requests to send before benchmarking.

##### REQUEST-CANCELLATION-RATE

`--request-cancellation-rate <float>`

The percentage of requests to cancel.

##### REQUEST-CANCELLATION-DELAY

`--request-cancellation-delay <float>`

The delay in seconds before cancelling requests. This is used when --request-cancellation-rate is greater than 0.

### ZMQ Communication Options

##### ZMQ-HOST

`--zmq-host <str>`

Host address for TCP connections

##### ZMQ-IPC-PATH

`--zmq-ipc-path <str>`

Path for IPC sockets

### Workers Options

##### WORKERS-MAX

`--workers-max <int>` | `--max-workers <int>`

Maximum number of workers to create. If not specified, the number of workers will be determined by the formula `min(concurrency, (num CPUs * 0.75) - 1)`,  with a default max cap of `32`. Any value provided will still be capped by the concurrency value (if specified), but not by the max cap.

### Service Options

##### LOG-LEVEL

`--log-level <str>`

Logging level

##### VERBOSE

`-v` | `--verbose`

Equivalent to --log-level DEBUG. Enables more verbose logging output, but lacks some raw message logging.

##### EXTRA-VERBOSE

`-vv` | `--extra-verbose`

Equivalent to --log-level TRACE. Enables the most verbose logging output possible.

##### RECORD-PROCESSOR-SERVICE-COUNT

`--record-processor-service-count <int>` | `--record-processors <int>`

Number of services to spawn for processing records. The higher the request rate, the more services should be spawned in order to keep up with the incoming records. If not specified, the number of services will be automatically determined based on the worker count.

##### UI-TYPE

`--ui-type <str>` | `--ui <str>`

Type of UI to use
