<!--
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->

# Command Line Options


## Endpoint Options

<dl>

<dt><code>-m</code>, <code>--model-names</code>, <code>--model</code> &lt;list&gt; <em>(Required)</em></dt>

<dd>

Model name(s) to be benchmarked. Can be a comma-separated list or a single model name.

</dd>


<dt><code>--model-selection-strategy</code> &lt;str&gt;</dt>

<dd>

When multiple models are specified, this is how a specific model should be assigned to a prompt.
round_robin: nth prompt in the list gets assigned to n-mod len(models).
random: assignment is uniformly random.


**Choices:** `round_robin`, `random`<br>
**Default:** `round_robin`<br>
</dd>


<dt><code>--custom-endpoint</code>, <code>--endpoint</code> &lt;str&gt;</dt>

<dd>

Set a custom endpoint that differs from the OpenAI defaults.

</dd>


<dt><code>--endpoint-type</code> &lt;str&gt;</dt>

<dd>

The endpoint type to send requests to on the server.


**Choices:** `chat`, `completions`, `cohere_rankings`, `embeddings`, `hf_tei_rankings`, `huggingface_generate`, `image_generation`, `nim_rankings`, `solido_rag`, `template`<br>
**Default:** `chat`<br>
</dd>


<dt><code>--streaming</code></dt>

<dd>

An option to enable the use of the streaming API.

</dd>


<dt><code>-u</code>, <code>--url</code> &lt;str&gt;</dt>

<dd>

URL of the endpoint to target for benchmarking.

**Default:** `localhost:8000`<br>
</dd>


<dt><code>--request-timeout-seconds</code> &lt;float&gt;</dt>

<dd>

The timeout in floating-point seconds for each request to the endpoint.

**Default:** `600.0`<br>
</dd>


<dt><code>--api-key</code> &lt;str&gt;</dt>

<dd>

The API key to use for the endpoint. If provided, it will be sent with every request as a header: `Authorization: Bearer <api_key>`.

</dd>


<dt><code>--transport</code>, <code>--transport-type</code> &lt;str&gt;</dt>

<dd>

The transport to use for the endpoint. If not provided, it will be auto-detected from the URL.This can also be used to force an alternative transport or implementation.


**Choices:** `http`<br>
</dd>

</dl>

## Input Options

<dl>

<dt><code>--extra-inputs</code> &lt;list&gt;</dt>

<dd>

Provide additional inputs to include with every request.
Inputs should be in an 'input_name:value' format.
Alternatively, a string representing a json formatted dict can be provided.

**Default:** `[]`<br>
</dd>


<dt><code>-H</code>, <code>--header</code> &lt;list&gt;</dt>

<dd>

Adds a custom header to the requests.
Headers must be specified as 'Header:Value' pairs.
Alternatively, a string representing a json formatted dict can be provided.

**Default:** `[]`<br>
</dd>


<dt><code>--input-file</code> &lt;str&gt;</dt>

<dd>

The file or directory path that contains the dataset to use for profiling.
This parameter is used in conjunction with the `custom_dataset_type` parameter
to support different types of user provided datasets.

</dd>


<dt><code>--fixed-schedule</code></dt>

<dd>

Specifies to run a fixed schedule of requests. This is normally inferred from the --input-file parameter, but can be set manually here.

</dd>


<dt><code>--fixed-schedule-auto-offset</code></dt>

<dd>

Specifies to automatically offset the timestamps in the fixed schedule, such that the first timestamp is considered 0, and the rest are shifted accordingly. If disabled, the timestamps will be assumed to be relative to 0.

</dd>


<dt><code>--fixed-schedule-start-offset</code> &lt;int&gt;</dt>

<dd>

Specifies the offset in milliseconds to start the fixed schedule at. By default, the schedule starts at 0, but this option can be used to start at a reference point further in the schedule. This option cannot be used in conjunction with the --fixed-schedule-auto-offset. The schedule will include any requests at the start offset.

</dd>


<dt><code>--fixed-schedule-end-offset</code> &lt;int&gt;</dt>

<dd>

Specifies the offset in milliseconds to end the fixed schedule at. By default, the schedule ends at the last timestamp in the trace dataset, but this option can be used to only run a subset of the trace. The schedule will include any requests at the end offset.

</dd>


<dt><code>--public-dataset</code> &lt;str&gt;</dt>

<dd>

The public dataset to use for the requests.


**Choices:** `sharegpt`<br>
</dd>


<dt><code>--custom-dataset-type</code> &lt;str&gt;</dt>

<dd>

The type of custom dataset to use.
This parameter is used in conjunction with the --input-file parameter.
[choices: single_turn, multi_turn, random_pool, mooncake_trace].

</dd>


<dt><code>--dataset-sampling-strategy</code> &lt;str&gt;</dt>

<dd>

The strategy to use for sampling the dataset.
`sequential`: Iterate through the dataset sequentially, then wrap around to the beginning.
`random`: Randomly select a conversation from the dataset. Will randomly sample with replacement.
`shuffle`: Shuffle the dataset and iterate through it. Will randomly sample without replacement.
Once the end of the dataset is reached, shuffle the dataset again and start over.


**Choices:** `sequential`, `random`, `shuffle`<br>
</dd>


<dt><code>--random-seed</code> &lt;int&gt;</dt>

<dd>

The seed used to generate random values.
Set to some value to make the synthetic data generation deterministic.
It will use system default if not provided.

</dd>


<dt><code>--goodput</code> &lt;str&gt;</dt>

<dd>

Specify service level objectives (SLOs) for goodput as space-separated 'KEY:VALUE' pairs, where KEY is a metric tag and VALUE is a number in the metric’s display unit (falls back to its base unit if no display unit is defined). Examples: 'request_latency:250' (ms), 'inter_token_latency:10' (ms), `output_token_throughput_per_user:600` (tokens/s).
Only metrics applicable to the current endpoint/config are considered. For more context on the definition of goodput, refer to DistServe paper: https://arxiv.org/pdf/2401.09670 and the blog: https://hao-ai-lab.github.io/blogs/distserve.

</dd>


<dt><code>--rankings-passages-mean</code> &lt;int&gt;</dt>

<dd>

Mean number of passages per rankings entry (per query)(default 1).

**Default:** `1`<br>
</dd>


<dt><code>--rankings-passages-stddev</code> &lt;int&gt;</dt>

<dd>

Stddev for passages per rankings entry (default 0).

**Default:** `0`<br>
</dd>

</dl>

## Audio Input Options

<dl>

<dt><code>--audio-batch-size</code>, <code>--batch-size-audio</code> &lt;int&gt;</dt>

<dd>

The batch size of audio requests AIPerf should send.
This is currently supported with the OpenAI `chat` endpoint type.

**Default:** `1`<br>
</dd>


<dt><code>--audio-length-mean</code> &lt;float&gt;</dt>

<dd>

The mean length of the audio in seconds.

**Default:** `0.0`<br>
</dd>


<dt><code>--audio-length-stddev</code> &lt;float&gt;</dt>

<dd>

The standard deviation of the length of the audio in seconds.

**Default:** `0.0`<br>
</dd>


<dt><code>--audio-format</code> &lt;str&gt;</dt>

<dd>

The format of the audio files (wav or mp3).


**Choices:** `wav`, `mp3`<br>
**Default:** `wav`<br>
</dd>


<dt><code>--audio-depths</code> &lt;list&gt;</dt>

<dd>

A list of audio bit depths to randomly select from in bits.

**Default:** `[16]`<br>
</dd>


<dt><code>--audio-sample-rates</code> &lt;list&gt;</dt>

<dd>

A list of audio sample rates to randomly select from in kHz.
Common sample rates are 16, 44.1, 48, 96, etc.

**Default:** `[16.0]`<br>
</dd>


<dt><code>--audio-num-channels</code> &lt;int&gt;</dt>

<dd>

The number of audio channels to use for the audio data generation.

**Default:** `1`<br>
</dd>

</dl>

## Image Input Options

<dl>

<dt><code>--image-width-mean</code> &lt;float&gt;</dt>

<dd>

The mean width of images when generating synthetic image data.

**Default:** `0.0`<br>
</dd>


<dt><code>--image-width-stddev</code> &lt;float&gt;</dt>

<dd>

The standard deviation of width of images when generating synthetic image data.

**Default:** `0.0`<br>
</dd>


<dt><code>--image-height-mean</code> &lt;float&gt;</dt>

<dd>

The mean height of images when generating synthetic image data.

**Default:** `0.0`<br>
</dd>


<dt><code>--image-height-stddev</code> &lt;float&gt;</dt>

<dd>

The standard deviation of height of images when generating synthetic image data.

**Default:** `0.0`<br>
</dd>


<dt><code>--image-batch-size</code>, <code>--batch-size-image</code> &lt;int&gt;</dt>

<dd>

The image batch size of the requests AIPerf should send.

**Default:** `1`<br>
</dd>


<dt><code>--image-format</code> &lt;str&gt;</dt>

<dd>

The compression format of the images.


**Choices:** `png`, `jpeg`, `random`<br>
**Default:** `png`<br>
</dd>

</dl>

## Video Input Options

<dl>

<dt><code>--video-batch-size</code>, <code>--batch-size-video</code> &lt;int&gt;</dt>

<dd>

The video batch size of the requests AIPerf should send.

**Default:** `1`<br>
</dd>


<dt><code>--video-duration</code> &lt;float&gt;</dt>

<dd>

Seconds per clip (default: 5.0).

**Default:** `5.0`<br>
</dd>


<dt><code>--video-fps</code> &lt;int&gt;</dt>

<dd>

Frames per second (default/recommended for Cosmos: 4).

**Default:** `4`<br>
</dd>


<dt><code>--video-width</code> &lt;int&gt;</dt>

<dd>

Video width in pixels.

</dd>


<dt><code>--video-height</code> &lt;int&gt;</dt>

<dd>

Video height in pixels.

</dd>


<dt><code>--video-synth-type</code> &lt;str&gt;</dt>

<dd>

Synthetic generator type.


**Choices:** `moving_shapes`, `grid_clock`<br>
**Default:** `moving_shapes`<br>
</dd>


<dt><code>--video-format</code> &lt;str&gt;</dt>

<dd>

The video format of the generated files.


**Choices:** `mp4`, `webm`<br>
**Default:** `webm`<br>
</dd>


<dt><code>--video-codec</code> &lt;str&gt;</dt>

<dd>

The video codec to use for encoding. Common options: libvpx-vp9 (CPU, BSD-licensed, default for WebM), libx264 (CPU, GPL-licensed, widely compatible), libx265 (CPU, GPL-licensed, smaller files), h264_nvenc (NVIDIA GPU), hevc_nvenc (NVIDIA GPU, smaller files). Any FFmpeg-supported codec can be used.

**Default:** `libvpx-vp9`<br>
</dd>

</dl>

## Prompt Options

<dl>

<dt><code>-b</code>, <code>--prompt-batch-size</code>, <code>--batch-size-text</code>, <code>--batch-size</code> &lt;int&gt;</dt>

<dd>

The batch size of text requests AIPerf should send.
This is currently supported with the embeddings and rankings endpoint types.

**Default:** `1`<br>
</dd>

</dl>

## Input Sequence Length (ISL) Options

<dl>

<dt><code>--prompt-input-tokens-mean</code>, <code>--synthetic-input-tokens-mean</code>, <code>--isl</code> &lt;int&gt;</dt>

<dd>

The mean of number of tokens in the generated prompts when using synthetic data.

**Default:** `550`<br>
</dd>


<dt><code>--prompt-input-tokens-stddev</code>, <code>--synthetic-input-tokens-stddev</code>, <code>--isl-stddev</code> &lt;float&gt;</dt>

<dd>

The standard deviation of number of tokens in the generated prompts when using synthetic data.

**Default:** `0.0`<br>
</dd>


<dt><code>--prompt-input-tokens-block-size</code>, <code>--synthetic-input-tokens-block-size</code>, <code>--isl-block-size</code> &lt;int&gt;</dt>

<dd>

The block size of the prompt.

**Default:** `512`<br>
</dd>


<dt><code>--seq-dist</code>, <code>--sequence-distribution</code> &lt;str&gt;</dt>

<dd>

Sequence length distribution specification for varying ISL/OSL pairs.

</dd>

</dl>

## Output Sequence Length (OSL) Options

<dl>

<dt><code>--prompt-output-tokens-mean</code>, <code>--output-tokens-mean</code>, <code>--osl</code> &lt;int&gt;</dt>

<dd>

The mean number of tokens in each output.

</dd>


<dt><code>--prompt-output-tokens-stddev</code>, <code>--output-tokens-stddev</code>, <code>--osl-stddev</code> &lt;float&gt;</dt>

<dd>

The standard deviation of the number of tokens in each output.

**Default:** `0`<br>
</dd>

</dl>

## Prefix Prompt Options

<dl>

<dt><code>--prompt-prefix-pool-size</code>, <code>--prefix-prompt-pool-size</code>, <code>--num-prefix-prompts</code> &lt;int&gt;</dt>

<dd>

The total size of the prefix prompt pool to select prefixes from.
If this value is not zero, these are prompts that are prepended to input prompts.
This is useful for benchmarking models that use a K-V cache.

**Default:** `0`<br>
</dd>


<dt><code>--prompt-prefix-length</code>, <code>--prefix-prompt-length</code> &lt;int&gt;</dt>

<dd>

The number of tokens in each prefix prompt.
This is only used if "num" is greater than zero.
Note that due to the prefix and user prompts being concatenated,
the number of tokens in the final prompt may be off by one.

**Default:** `0`<br>
</dd>

</dl>

## Conversation Input Options

<dl>

<dt><code>--conversation-num</code>, <code>--num-conversations</code>, <code>--num-sessions</code> &lt;int&gt;</dt>

<dd>

The total number of unique conversations to generate.
Each conversation represents a single request session between client and server.
Supported on synthetic mode and the custom random_pool dataset. The number of conversations
will be used to determine the number of entries in both the custom random_pool and synthetic
datasets and will be reused until benchmarking is complete.

</dd>


<dt><code>--num-dataset-entries</code>, <code>--num-prompts</code> &lt;int&gt;</dt>

<dd>

The total number of unique dataset entries to generate for the dataset.
Each entry represents a single turn used in a request.

**Default:** `100`<br>
</dd>


<dt><code>--conversation-turn-mean</code>, <code>--session-turns-mean</code> &lt;int&gt;</dt>

<dd>

The mean number of turns within a conversation.

**Default:** `1`<br>
</dd>


<dt><code>--conversation-turn-stddev</code>, <code>--session-turns-stddev</code> &lt;int&gt;</dt>

<dd>

The standard deviation of the number of turns within a conversation.

**Default:** `0`<br>
</dd>


<dt><code>--conversation-turn-delay-mean</code>, <code>--session-turn-delay-mean</code> &lt;float&gt;</dt>

<dd>

The mean delay between turns within a conversation in milliseconds.

**Default:** `0.0`<br>
</dd>


<dt><code>--conversation-turn-delay-stddev</code>, <code>--session-turn-delay-stddev</code> &lt;float&gt;</dt>

<dd>

The standard deviation of the delay between turns
within a conversation in milliseconds.

**Default:** `0.0`<br>
</dd>


<dt><code>--conversation-turn-delay-ratio</code>, <code>--session-delay-ratio</code> &lt;float&gt;</dt>

<dd>

A ratio to scale multi-turn delays.

**Default:** `1.0`<br>
</dd>

</dl>

## Output Options

<dl>

<dt><code>--output-artifact-dir</code>, <code>--artifact-dir</code> &lt;str&gt;</dt>

<dd>

The directory to store all the (output) artifacts generated by AIPerf.

**Default:** `artifacts`<br>
</dd>


<dt><code>--profile-export-prefix</code>, <code>--profile-export-file</code> &lt;str&gt;</dt>

<dd>

The prefix for the profile export file names. Will be suffixed with .csv, .json, .jsonl, and _raw.jsonl.If not provided, the default profile export file names will be used: profile_export_aiperf.csv, profile_export_aiperf.json, profile_export.jsonl, and profile_export_raw.jsonl.

</dd>


<dt><code>--export-level</code>, <code>--profile-export-level</code> &lt;str&gt;</dt>

<dd>

The level of profile export files to create.


**Choices:** `summary`, `records`, `raw`<br>
**Default:** `records`<br>
</dd>


<dt><code>--slice-duration</code> &lt;float&gt;</dt>

<dd>

The duration (in seconds) of an individual time slice to be used post-benchmark in time-slicing mode.

</dd>

</dl>

## Tokenizer Options

<dl>

<dt><code>--tokenizer</code> &lt;str&gt;</dt>

<dd>

The HuggingFace tokenizer to use to interpret token metrics from prompts and responses.
The value can be the name of a tokenizer or the filepath of the tokenizer.
The default value is the model name.

</dd>


<dt><code>--tokenizer-revision</code> &lt;str&gt;</dt>

<dd>

The specific model version to use.
It can be a branch name, tag name, or commit ID.

**Default:** `main`<br>
</dd>


<dt><code>--tokenizer-trust-remote-code</code></dt>

<dd>

Allows custom tokenizer to be downloaded and executed.
This carries security risks and should only be used for repositories you trust.
This is only necessary for custom tokenizers stored in HuggingFace Hub.

</dd>

</dl>

## Load Generator Options

<dl>

<dt><code>--benchmark-duration</code> &lt;float&gt;</dt>

<dd>

The duration in seconds for benchmarking.

</dd>


<dt><code>--benchmark-grace-period</code> &lt;float&gt;</dt>

<dd>

The grace period in seconds to wait for responses after benchmark duration ends. Only applies when --benchmark-duration is set. Responses received within this period are included in metrics.

**Default:** `30.0`<br>
</dd>


<dt><code>--concurrency</code> &lt;int&gt;</dt>

<dd>

The concurrency value to benchmark.

</dd>


<dt><code>--request-rate</code> &lt;float&gt;</dt>

<dd>

Sets the request rate for the load generated by AIPerf. Unit: requests/second.

</dd>


<dt><code>--request-rate-mode</code> &lt;str&gt;</dt>

<dd>

Sets the request rate mode for the load generated by AIPerf. Valid values: constant, poisson.
constant: Generate requests at a fixed rate.
poisson: Generate requests using a poisson distribution.

**Default:** `poisson`<br>
</dd>


<dt><code>--request-count</code>, <code>--num-requests</code> &lt;int&gt;</dt>

<dd>

The number of requests to use for measurement.

**Default:** `10`<br>
</dd>


<dt><code>--warmup-request-count</code>, <code>--num-warmup-requests</code> &lt;int&gt;</dt>

<dd>

The number of warmup requests to send before benchmarking.

**Default:** `0`<br>
</dd>


<dt><code>--request-cancellation-rate</code> &lt;float&gt;</dt>

<dd>

The percentage of requests to cancel.

**Default:** `0.0`<br>
</dd>


<dt><code>--request-cancellation-delay</code> &lt;float&gt;</dt>

<dd>

The delay in seconds before cancelling requests. This is used when --request-cancellation-rate is greater than 0.

**Default:** `0.0`<br>
</dd>

</dl>

## Telemetry Options

<dl>

<dt><code>--gpu-telemetry</code> &lt;list&gt;</dt>

<dd>

Enable GPU telemetry console display and optionally specify: (1) 'dashboard' for realtime dashboard mode, (2) custom DCGM exporter URLs (e.g., http://node1:9401/metrics), (3) custom metrics CSV file (e.g., custom_gpu_metrics.csv). Default endpoints localhost:9400 and localhost:9401 are always attempted. Example: --gpu-telemetry dashboard node1:9400 custom.csv.

</dd>

</dl>

## ZMQ Communication Options

<dl>

<dt><code>--zmq-host</code> &lt;str&gt;</dt>

<dd>

Host address for TCP connections.

**Default:** `127.0.0.1`<br>
</dd>


<dt><code>--zmq-ipc-path</code> &lt;str&gt;</dt>

<dd>

Path for IPC sockets.

</dd>

</dl>

## Workers Options

<dl>

<dt><code>--workers-max</code>, <code>--max-workers</code> &lt;int&gt;</dt>

<dd>

Maximum number of workers to create. If not specified, the number of workers will be determined by the formula `min(concurrency, (num CPUs * 0.75) - 1)`,  with a default max cap of `32`. Any value provided will still be capped by the concurrency value (if specified), but not by the max cap.

</dd>

</dl>

## Service Options

<dl>

<dt><code>--log-level</code> &lt;str&gt;</dt>

<dd>

Logging level.


**Choices:** `TRACE`, `DEBUG`, `INFO`, `NOTICE`, `WARNING`, `SUCCESS`, `ERROR`, `CRITICAL`<br>
**Default:** `INFO`<br>
</dd>


<dt><code>-v</code>, <code>--verbose</code></dt>

<dd>

Equivalent to --log-level DEBUG. Enables more verbose logging output, but lacks some raw message logging.

</dd>


<dt><code>-vv</code>, <code>--extra-verbose</code></dt>

<dd>

Equivalent to --log-level TRACE. Enables the most verbose logging output possible.

</dd>


<dt><code>--record-processor-service-count</code>, <code>--record-processors</code> &lt;int&gt;</dt>

<dd>

Number of services to spawn for processing records. The higher the request rate, the more services should be spawned in order to keep up with the incoming records. If not specified, the number of services will be automatically determined based on the worker count.

</dd>


<dt><code>--ui-type</code>, <code>--ui</code> &lt;str&gt;</dt>

<dd>

Type of UI to use.


**Choices:** `none`, `simple`, `dashboard`<br>
**Default:** `dashboard`<br>
</dd>

</dl>

