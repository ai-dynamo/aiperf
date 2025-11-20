<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Command Line Options

## `aiperf` Commands

- [`profile`](#aiperf-profile) - Run the Profile subcommand.

## `aiperf profile`

## Endpoint Options

#### `-m`, `--model-names`, `--model` `<list>` _(Required)_

Model name(s) to be benchmarked. Can be a comma-separated list or a single model name.

#### `--model-selection-strategy` `<str>`

When multiple models are specified, this is how a specific model should be assigned to a prompt. round_robin: nth prompt in the list gets assigned to n-mod len(models). random: assignment is uniformly random.
<br>_Choices: [`round_robin`, `random`]_
<br>_Default: `round_robin`_

#### `--custom-endpoint`, `--endpoint` `<str>`

Set a custom endpoint that differs from the OpenAI defaults.

#### `--endpoint-type` `<str>`

The endpoint type to send requests to on the server.
<br>_Choices: [`chat`, `completions`, `cohere_rankings`, `embeddings`, `hf_tei_rankings`, `huggingface_generate`, `image_generation`, `nim_rankings`, `solido_rag`, `template`]_
<br>_Default: `chat`_

#### `--streaming`

An option to enable the use of the streaming API.

#### `-u`, `--url` `<str>`

URL of the endpoint to target for benchmarking.
<br>_Default: `localhost:8000`_

#### `--request-timeout-seconds` `<float>`

The timeout in floating-point seconds for each request to the endpoint.
<br>_Default: `600.0`_

#### `--api-key` `<str>`

The API key to use for the endpoint. If provided, it will be sent with every request as a header: `Authorization: Bearer <api_key>`.

#### `--transport`, `--transport-type` `<str>`

The transport to use for the endpoint. If not provided, it will be auto-detected from the URL.This can also be used to force an alternative transport or implementation.
<br>_Choices: [`http`]_

#### `--use-legacy-max-tokens`

Use the legacy 'max_tokens' field instead of 'max_completion_tokens' in request payloads. The OpenAI API now prefers 'max_completion_tokens', but some older APIs or implementations may require 'max_tokens'.

## Input Options

#### `--extra-inputs` `<list>`

Provide additional inputs to include with every request. Inputs should be in an 'input_name:value' format. Alternatively, a string representing a json formatted dict can be provided.
<br>_Default: `[]`_

#### `-H`, `--header` `<list>`

Adds a custom header to the requests. Headers must be specified as 'Header:Value' pairs. Alternatively, a string representing a json formatted dict can be provided.
<br>_Default: `[]`_

#### `--input-file` `<str>`

The file or directory path that contains the dataset to use for profiling. This parameter is used in conjunction with the `custom_dataset_type` parameter to support different types of user provided datasets.

#### `--fixed-schedule`

Specifies to run a fixed schedule of requests. This is normally inferred from the --input-file parameter, but can be set manually here.

#### `--fixed-schedule-auto-offset`

Specifies to automatically offset the timestamps in the fixed schedule, such that the first timestamp is considered 0, and the rest are shifted accordingly. If disabled, the timestamps will be assumed to be relative to 0.

#### `--fixed-schedule-start-offset` `<int>`

Specifies the offset in milliseconds to start the fixed schedule at. By default, the schedule starts at 0, but this option can be used to start at a reference point further in the schedule. This option cannot be used in conjunction with the --fixed-schedule-auto-offset. The schedule will include any requests at the start offset.

#### `--fixed-schedule-end-offset` `<int>`

Specifies the offset in milliseconds to end the fixed schedule at. By default, the schedule ends at the last timestamp in the trace dataset, but this option can be used to only run a subset of the trace. The schedule will include any requests at the end offset.

#### `--public-dataset` `<str>`

The public dataset to use for the requests.
<br>_Choices: [`sharegpt`]_

#### `--custom-dataset-type` `<str>`

The type of custom dataset to use. This parameter is used in conjunction with the --input-file parameter. [choices: single_turn, multi_turn, random_pool, mooncake_trace].

#### `--dataset-sampling-strategy` `<str>`

The strategy to use for sampling the dataset. `sequential`: Iterate through the dataset sequentially, then wrap around to the beginning. `random`: Randomly select a conversation from the dataset. Will randomly sample with replacement. `shuffle`: Shuffle the dataset and iterate through it. Will randomly sample without replacement. Once the end of the dataset is reached, shuffle the dataset again and start over.
<br>_Choices: [`sequential`, `random`, `shuffle`]_

#### `--random-seed` `<int>`

The seed used to generate random values. Set to some value to make the synthetic data generation deterministic. It will use system default if not provided.

#### `--goodput` `<str>`

Specify service level objectives (SLOs) for goodput as space-separated 'KEY:VALUE' pairs, where KEY is a metric tag and VALUE is a number in the metric’s display unit (falls back to its base unit if no display unit is defined). Examples: 'request_latency:250' (ms), 'inter_token_latency:10' (ms), `output_token_throughput_per_user:600` (tokens/s). Only metrics applicable to the current endpoint/config are considered. For more context on the definition of goodput, refer to DistServe paper: https://arxiv.org/pdf/2401.09670 and the blog: https://hao-ai-lab.github.io/blogs/distserve.

#### `--rankings-passages-mean` `<int>`

Mean number of passages per rankings entry (per query)(default 1).
<br>_Default: `1`_

#### `--rankings-passages-stddev` `<int>`

Stddev for passages per rankings entry (default 0).
<br>_Default: `0`_

## Audio Input Options

#### `--audio-batch-size`, `--batch-size-audio` `<int>`

The batch size of audio requests AIPerf should send. This is currently supported with the OpenAI `chat` endpoint type.
<br>_Default: `1`_

#### `--audio-length-mean` `<float>`

The mean length of the audio in seconds.
<br>_Default: `0.0`_

#### `--audio-length-stddev` `<float>`

The standard deviation of the length of the audio in seconds.
<br>_Default: `0.0`_

#### `--audio-format` `<str>`

The format of the audio files (wav or mp3).
<br>_Choices: [`wav`, `mp3`]_
<br>_Default: `wav`_

#### `--audio-depths` `<list>`

A list of audio bit depths to randomly select from in bits.
<br>_Default: `[16]`_

#### `--audio-sample-rates` `<list>`

A list of audio sample rates to randomly select from in kHz. Common sample rates are 16, 44.1, 48, 96, etc.
<br>_Default: `[16.0]`_

#### `--audio-num-channels` `<int>`

The number of audio channels to use for the audio data generation.
<br>_Default: `1`_

## Image Input Options

#### `--image-width-mean` `<float>`

The mean width of images when generating synthetic image data.
<br>_Default: `0.0`_

#### `--image-width-stddev` `<float>`

The standard deviation of width of images when generating synthetic image data.
<br>_Default: `0.0`_

#### `--image-height-mean` `<float>`

The mean height of images when generating synthetic image data.
<br>_Default: `0.0`_

#### `--image-height-stddev` `<float>`

The standard deviation of height of images when generating synthetic image data.
<br>_Default: `0.0`_

#### `--image-batch-size`, `--batch-size-image` `<int>`

The image batch size of the requests AIPerf should send.
<br>_Default: `1`_

#### `--image-format` `<str>`

The compression format of the images.
<br>_Choices: [`png`, `jpeg`, `random`]_
<br>_Default: `png`_

## Video Input Options

#### `--video-batch-size`, `--batch-size-video` `<int>`

The video batch size of the requests AIPerf should send.
<br>_Default: `1`_

#### `--video-duration` `<float>`

Seconds per clip (default: 5.0).
<br>_Default: `5.0`_

#### `--video-fps` `<int>`

Frames per second (default/recommended for Cosmos: 4).
<br>_Default: `4`_

#### `--video-width` `<int>`

Video width in pixels.

#### `--video-height` `<int>`

Video height in pixels.

#### `--video-synth-type` `<str>`

Synthetic generator type.
<br>_Choices: [`moving_shapes`, `grid_clock`]_
<br>_Default: `moving_shapes`_

#### `--video-format` `<str>`

The video format of the generated files.
<br>_Choices: [`mp4`, `webm`]_
<br>_Default: `webm`_

#### `--video-codec` `<str>`

The video codec to use for encoding. Common options: libvpx-vp9 (CPU, BSD-licensed, default for WebM), libx264 (CPU, GPL-licensed, widely compatible), libx265 (CPU, GPL-licensed, smaller files), h264_nvenc (NVIDIA GPU), hevc_nvenc (NVIDIA GPU, smaller files). Any FFmpeg-supported codec can be used.
<br>_Default: `libvpx-vp9`_

## Prompt Options

#### `-b`, `--prompt-batch-size`, `--batch-size-text`, `--batch-size` `<int>`

The batch size of text requests AIPerf should send. This is currently supported with the embeddings and rankings endpoint types.
<br>_Default: `1`_

## Input Sequence Length (ISL) Options

#### `--prompt-input-tokens-mean`, `--synthetic-input-tokens-mean`, `--isl` `<int>`

The mean of number of tokens in the generated prompts when using synthetic data.
<br>_Default: `550`_

#### `--prompt-input-tokens-stddev`, `--synthetic-input-tokens-stddev`, `--isl-stddev` `<float>`

The standard deviation of number of tokens in the generated prompts when using synthetic data.
<br>_Default: `0.0`_

#### `--prompt-input-tokens-block-size`, `--synthetic-input-tokens-block-size`, `--isl-block-size` `<int>`

The block size of the prompt.
<br>_Default: `512`_

#### `--seq-dist`, `--sequence-distribution` `<str>`

Sequence length distribution specification for varying ISL/OSL pairs.

## Output Sequence Length (OSL) Options

#### `--prompt-output-tokens-mean`, `--output-tokens-mean`, `--osl` `<int>`

The mean number of tokens in each output.

#### `--prompt-output-tokens-stddev`, `--output-tokens-stddev`, `--osl-stddev` `<float>`

The standard deviation of the number of tokens in each output.
<br>_Default: `0`_

## Prefix Prompt Options

#### `--prompt-prefix-pool-size`, `--prefix-prompt-pool-size`, `--num-prefix-prompts` `<int>`

The total size of the prefix prompt pool to select prefixes from. If this value is not zero, these are prompts that are prepended to input prompts. This is useful for benchmarking models that use a K-V cache.
<br>_Default: `0`_

#### `--prompt-prefix-length`, `--prefix-prompt-length` `<int>`

The number of tokens in each prefix prompt. This is only used if "num" is greater than zero. Note that due to the prefix and user prompts being concatenated, the number of tokens in the final prompt may be off by one.
<br>_Default: `0`_

## Conversation Input Options

#### `--conversation-num`, `--num-conversations`, `--num-sessions` `<int>`

The total number of unique conversations to generate. Each conversation represents a single request session between client and server. Supported on synthetic mode and the custom random_pool dataset. The number of conversations will be used to determine the number of entries in both the custom random_pool and synthetic datasets and will be reused until benchmarking is complete.

#### `--num-dataset-entries`, `--num-prompts` `<int>`

The total number of unique dataset entries to generate for the dataset. Each entry represents a single turn used in a request.
<br>_Default: `100`_

#### `--conversation-turn-mean`, `--session-turns-mean` `<int>`

The mean number of turns within a conversation.
<br>_Default: `1`_

#### `--conversation-turn-stddev`, `--session-turns-stddev` `<int>`

The standard deviation of the number of turns within a conversation.
<br>_Default: `0`_

#### `--conversation-turn-delay-mean`, `--session-turn-delay-mean` `<float>`

The mean delay between turns within a conversation in milliseconds.
<br>_Default: `0.0`_

#### `--conversation-turn-delay-stddev`, `--session-turn-delay-stddev` `<float>`

The standard deviation of the delay between turns within a conversation in milliseconds.
<br>_Default: `0.0`_

#### `--conversation-turn-delay-ratio`, `--session-delay-ratio` `<float>`

A ratio to scale multi-turn delays.
<br>_Default: `1.0`_

## Output Options

#### `--output-artifact-dir`, `--artifact-dir` `<str>`

The directory to store all the (output) artifacts generated by AIPerf.
<br>_Default: `artifacts`_

#### `--profile-export-prefix`, `--profile-export-file` `<str>`

The prefix for the profile export file names. Will be suffixed with .csv, .json, .jsonl, and _raw.jsonl.If not provided, the default profile export file names will be used: profile_export_aiperf.csv, profile_export_aiperf.json, profile_export.jsonl, and profile_export_raw.jsonl.

#### `--export-level`, `--profile-export-level` `<str>`

The level of profile export files to create.
<br>_Choices: [`summary`, `records`, `raw`]_
<br>_Default: `records`_

#### `--slice-duration` `<float>`

The duration (in seconds) of an individual time slice to be used post-benchmark in time-slicing mode.

## Tokenizer Options

#### `--tokenizer` `<str>`

The HuggingFace tokenizer to use to interpret token metrics from prompts and responses. The value can be the name of a tokenizer or the filepath of the tokenizer. The default value is the model name.

#### `--tokenizer-revision` `<str>`

The specific model version to use. It can be a branch name, tag name, or commit ID.
<br>_Default: `main`_

#### `--tokenizer-trust-remote-code`

Allows custom tokenizer to be downloaded and executed. This carries security risks and should only be used for repositories you trust. This is only necessary for custom tokenizers stored in HuggingFace Hub.

## Load Generator Options

#### `--benchmark-duration` `<float>`

The duration in seconds for benchmarking.

#### `--benchmark-grace-period` `<float>`

The grace period in seconds to wait for responses after benchmark duration ends. Only applies when --benchmark-duration is set. Responses received within this period are included in metrics.
<br>_Default: `30.0`_

#### `--concurrency` `<int>`

The concurrency value to benchmark.

#### `--request-rate` `<float>`

Sets the request rate for the load generated by AIPerf. Unit: requests/second.

#### `--request-rate-mode` `<str>`

Sets the request rate mode for the load generated by AIPerf. Valid values: constant, poisson. constant: Generate requests at a fixed rate. poisson: Generate requests using a poisson distribution.
<br>_Default: `poisson`_

#### `--request-count`, `--num-requests` `<int>`

The number of requests to use for measurement.
<br>_Default: `10`_

#### `--warmup-request-count`, `--num-warmup-requests` `<int>`

The number of warmup requests to send before benchmarking.
<br>_Default: `0`_

#### `--request-cancellation-rate` `<float>`

The percentage of requests to cancel.
<br>_Default: `0.0`_

#### `--request-cancellation-delay` `<float>`

The delay in seconds before cancelling requests. This is used when --request-cancellation-rate is greater than 0.
<br>_Default: `0.0`_

## Telemetry Options

#### `--gpu-telemetry` `<list>`

Enable GPU telemetry console display and optionally specify: (1) 'dashboard' for realtime dashboard mode, (2) custom DCGM exporter URLs (e.g., http://node1:9401/metrics), (3) custom metrics CSV file (e.g., custom_gpu_metrics.csv). Default endpoints localhost:9400 and localhost:9401 are always attempted. Example: --gpu-telemetry dashboard node1:9400 custom.csv.

## ZMQ Communication Options

#### `--zmq-host` `<str>`

Host address for TCP connections.
<br>_Default: `127.0.0.1`_

#### `--zmq-ipc-path` `<str>`

Path for IPC sockets.

## Workers Options

#### `--workers-max`, `--max-workers` `<int>`

Maximum number of workers to create. If not specified, the number of workers will be determined by the formula `min(concurrency, (num CPUs * 0.75) - 1)`, with a default max cap of `32`. Any value provided will still be capped by the concurrency value (if specified), but not by the max cap.

## Service Options

#### `--log-level` `<str>`

Logging level.
<br>_Choices: [`TRACE`, `DEBUG`, `INFO`, `NOTICE`, `WARNING`, `SUCCESS`, `ERROR`, `CRITICAL`]_
<br>_Default: `INFO`_

#### `-v`, `--verbose`

Equivalent to --log-level DEBUG. Enables more verbose logging output, but lacks some raw message logging.

#### `-vv`, `--extra-verbose`

Equivalent to --log-level TRACE. Enables the most verbose logging output possible.

#### `--record-processor-service-count`, `--record-processors` `<int>`

Number of services to spawn for processing records. The higher the request rate, the more services should be spawned in order to keep up with the incoming records. If not specified, the number of services will be automatically determined based on the worker count.

#### `--ui-type`, `--ui` `<str>`

Type of UI to use.
<br>_Choices: [`none`, `simple`, `dashboard`]_
<br>_Default: `dashboard`_
