<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# CLI Options
The following options are available when profiling using AIPerf.

```
╭─ Endpoint ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *  MODEL-NAMES --model-names --model                    -m  Model name(s) to be benchmarked. Can be a comma-separated list or a single model name. [required]                         │
│    MODEL-SELECTION-STRATEGY --model-selection-strategy      When multiple models are specified, this is how a specific model should be assigned to a prompt. round_robin: nth prompt  │
│                                                             in the list gets assigned to n-mod len(models). random: assignment is uniformly random [choices: round-robin, random]     │
│                                                             [default: round-robin]                                                                                                    │
│    CUSTOM-ENDPOINT --custom-endpoint --endpoint             Set a custom endpoint that differs from the OpenAI defaults.                                                              │
│    ENDPOINT-TYPE --endpoint-type                            The endpoint type to send requests to on the server. [choices: openai-chat-completions, openai-completions,               │
│                                                             openai-embeddings, openai-responses] [default: openai-chat-completions]                                                   │
│    STREAMING --streaming                                    An option to enable the use of the streaming API. [default: False]                                                        │
│    URL --url                                            -u  URL of the endpoint to target for benchmarking. [default: localhost:8000]                                                 │
│    REQUEST-TIMEOUT-SECONDS --request-timeout-seconds        The timeout in floating points seconds for each request to the endpoint. [default: 600.0]                                 │
│    API-KEY --api-key                                        The API key to use for the endpoint. If provided, it will be sent with every request as a header: Authorization: Bearer   │
│                                                             <api_key>.                                                                                                                │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```
```
╭─ Input ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ EXTRA-INPUTS --extra-inputs                                    Provide additional inputs to include with every request. Inputs should be in an 'input_name:value' format.             │
│                                                                Alternatively, a string representing a json formatted dict can be provided. [default: []]                              │
│ HEADER --header                                            -H  Adds a custom header to the requests. Headers must be specified as 'Header:Value' pairs. Alternatively, a string       │
│                                                                representing a json formatted dict can be provided. [default: []]                                                      │
│ INPUT-FILE --input-file                                        The file or directory path that contains the dataset to use for profiling. This parameter is used in conjunction with  │
│                                                                the custom_dataset_type parameter to support different types of user provided datasets.                                │
│ FIXED-SCHEDULE --fixed-schedule                                Specifies to run a fixed schedule of requests. This is normally inferred from the --input-file parameter, but can be   │
│                                                                set manually here. [default: False]                                                                                    │
│ FIXED-SCHEDULE-AUTO-OFFSET --fixed-schedule-auto-offset        Specifies to automatically offset the timestamps in the fixed schedule, such that the first timestamp is considered 0, │
│                                                                and the rest are shifted accordingly. If disabled, the timestamps will be assumed to be relative to 0. [default:       │
│                                                                False]                                                                                                                 │
│ FIXED-SCHEDULE-START-OFFSET --fixed-schedule-start-offset      Specifies the offset in milliseconds to start the fixed schedule at. By default, the schedule starts at 0, but this    │
│                                                                option can be used to start at a reference point further in the schedule. This option cannot be used in conjunction    │
│                                                                with the --fixed-schedule-auto-offset. The schedule will include any requests at the start offset.                     │
│ FIXED-SCHEDULE-END-OFFSET --fixed-schedule-end-offset          Specifies the offset in milliseconds to end the fixed schedule at. By default, the schedule ends at the last timestamp │
│                                                                in the trace dataset, but this option can be used to only run a subset of the trace. The schedule will include any     │
│                                                                requests at the end offset.                                                                                            │
│ CUSTOM-DATASET-TYPE --custom-dataset-type                      The type of custom dataset to use. This parameter is used in conjunction with the --file parameter. [choices:          │
│                                                                single-turn, multi-turn, random-pool, mooncake-trace] [default: mooncake-trace]                                        │
│ RANDOM-SEED --random-seed                                      The seed used to generate random values. Set to some value to make the synthetic data generation deterministic. It     │
│                                                                will use system default if not provided.                                                                               │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```
```
╭─ Output ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ OUTPUT-ARTIFACT-DIR --output-artifact-dir --artifact-dir  The directory to store all the (output) artifacts generated by AIPerf. [default: artifacts]                                 │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```
```
╭─ Tokenizer ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ TOKENIZER --tokenizer                                      The HuggingFace tokenizer to use to interpret token metrics from prompts and responses. The value can be the name of a     │
│                                                            tokenizer or the filepath of the tokenizer. The default value is the model name.                                           │
│ TOKENIZER-REVISION --tokenizer-revision                    The specific model version to use. It can be a branch name, tag name, or commit ID. [default: main]                        │
│ TOKENIZER-TRUST-REMOTE-CODE --tokenizer-trust-remote-code  Allows custom tokenizer to be downloaded and executed. This carries security risks and should only be used for             │
│                                                            repositories you trust. This is only necessary for custom tokenizers stored in HuggingFace Hub. [default: False]           │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```
```
╭─ Load Generator ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ CONCURRENCY --concurrency                                          The concurrency value to benchmark.                                                                                │
│ REQUEST-RATE --request-rate                                        Sets the request rate for the load generated by AIPerf. Unit: requests/second                                      │
│ REQUEST-RATE-MODE --request-rate-mode                              Sets the request rate mode for the load generated by AIPerf. Valid values: constant, poisson. constant: Generate   │
│                                                                    requests at a fixed rate. poisson: Generate requests using a poisson distribution. [default: poisson]              │
│ REQUEST-COUNT --request-count --num-requests                       The number of requests to use for measurement. [default: 10]                                                       │
│ WARMUP-REQUEST-COUNT --warmup-request-count --num-warmup-requests  The number of warmup requests to send before benchmarking. [default: 0]                                            │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```
```
╭─ Conversation Input ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ CONVERSATION-NUM --conversation-num --num-conversations          The total number of unique conversations to generate. Each conversation represents a single request session between  │
│   --num-sessions --num-dataset-entries                           client and server. Supported on synthetic mode and the custom random_pool dataset. The number of conversations will  │
│                                                                  be used to determine the number of entries in both the custom random_pool and synthetic datasets and will be reused  │
│                                                                  until benchmarking is complete. [default: 100]                                                                       │
│ CONVERSATION-TURN-MEAN --conversation-turn-mean                  The mean number of turns within a conversation. [default: 1]                                                         │
│   --session-turns-mean                                                                                                                                                                │
│ CONVERSATION-TURN-STDDEV --conversation-turn-stddev              The standard deviation of the number of turns within a conversation. [default: 0]                                    │
│   --session-turns-stddev                                                                                                                                                              │
│ CONVERSATION-TURN-DELAY-MEAN --conversation-turn-delay-mean      The mean delay between turns within a conversation in milliseconds. [default: 0.0]                                   │
│   --session-turn-delay-mean                                                                                                                                                           │
│ CONVERSATION-TURN-DELAY-STDDEV --conversation-turn-delay-stddev  The standard deviation of the delay between turns within a conversation in milliseconds. [default: 0.0]              │
│   --session-turn-delay-stddev                                                                                                                                                         │
│ CONVERSATION-TURN-DELAY-RATIO --conversation-turn-delay-ratio    A ratio to scale multi-turn delays. [default: 1.0]                                                                   │
│   --session-delay-ratio                                                                                                                                                               │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```
```
╭─ Input Sequence Length (ISL) ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ PROMPT-INPUT-TOKENS-MEAN --prompt-input-tokens-mean              The mean of number of tokens in the generated prompts when using synthetic data. [default: 550]                      │
│   --synthetic-input-tokens-mean --isl                                                                                                                                                 │
│ PROMPT-INPUT-TOKENS-STDDEV --prompt-input-tokens-stddev          The standard deviation of number of tokens in the generated prompts when using synthetic data. [default: 0.0]        │
│   --synthetic-input-tokens-stddev --isl-stddev                                                                                                                                        │
│ PROMPT-INPUT-TOKENS-BLOCK-SIZE --prompt-input-tokens-block-size  The block size of the prompt. [default: 512]                                                                         │
│   --synthetic-input-tokens-block-size --isl-block-size                                                                                                                                │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```
```
╭─ Output Sequence Length (OSL) ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ PROMPT-OUTPUT-TOKENS-MEAN --prompt-output-tokens-mean      The mean number of tokens in each output.                                                                                  │
│   --output-tokens-mean --osl                                                                                                                                                          │
│ PROMPT-OUTPUT-TOKENS-STDDEV --prompt-output-tokens-stddev  The standard deviation of the number of tokens in each output. [default: 0]                                                │
│   --output-tokens-stddev --osl-stddev                                                                                                                                                 │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```
```
╭─ Prompt ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ PROMPT-BATCH-SIZE --prompt-batch-size --batch-size-text  -b  The batch size of text requests AIPerf should send. This is currently supported with the embeddings and rankings         │
│   --batch-size                                               endpoint types [default: 1]                                                                                              │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```
```
╭─ Prefix Prompt ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ PROMPT-PREFIX-POOL-SIZE --prompt-prefix-pool-size  The total size of the prefix prompt pool to select prefixes from. If this value is not zero, these are prompts that are prepended  │
│   --prefix-prompt-pool-size --num-prefix-prompts   to input prompts. This is useful for benchmarking models that use a K-V cache. [default: 0]                                        │
│ PROMPT-PREFIX-LENGTH --prompt-prefix-length        The number of tokens in each prefix prompt. This is only used if "num" is greater than zero. Note that due to the prefix and user  │
│   --prefix-prompt-length                           prompts being concatenated, the number of tokens in the final prompt may be off by one. [default: 0]                               │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```
```
╭─ Audio Input ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ AUDIO-BATCH-SIZE --audio-batch-size --batch-size-audio  The batch size of audio requests AIPerf should send. This is currently supported with the OpenAI chat endpoint type [default: │
│                                                         1]                                                                                                                            │
│ AUDIO-LENGTH-MEAN --audio-length-mean                   The mean length of the audio in seconds. [default: 0.0]                                                                       │
│ AUDIO-LENGTH-STDDEV --audio-length-stddev               The standard deviation of the length of the audio in seconds. [default: 0.0]                                                  │
│ AUDIO-FORMAT --audio-format                             The format of the audio files (wav or mp3). [choices: wav, mp3] [default: wav]                                                │
│ AUDIO-DEPTHS --audio-depths                             A list of audio bit depths to randomly select from in bits. [default: [16]]                                                   │
│ AUDIO-SAMPLE-RATES --audio-sample-rates                 A list of audio sample rates to randomly select from in kHz. Common sample rates are 16, 44.1, 48, 96, etc. [default: [16.0]] │
│ AUDIO-NUM-CHANNELS --audio-num-channels                 The number of audio channels to use for the audio data generation. [default: 1]                                               │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```
```
╭─ Image Input ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ IMAGE-WIDTH-MEAN --image-width-mean                     The mean width of images when generating synthetic image data. [default: 0.0]                                                 │
│ IMAGE-WIDTH-STDDEV --image-width-stddev                 The standard deviation of width of images when generating synthetic image data. [default: 0.0]                                │
│ IMAGE-HEIGHT-MEAN --image-height-mean                   The mean height of images when generating synthetic image data. [default: 0.0]                                                │
│ IMAGE-HEIGHT-STDDEV --image-height-stddev               The standard deviation of height of images when generating synthetic image data. [default: 0.0]                               │
│ IMAGE-BATCH-SIZE --image-batch-size --batch-size-image  The image batch size of the requests AIPerf should send. This is currently supported with the image retrieval endpoint type.  │
│                                                         [default: 1]                                                                                                                  │
│ IMAGE-FORMAT --image-format                             The compression format of the images. [choices: png, jpeg, random] [default: png]                                             │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```
```
╭─ Service ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ LOG-LEVEL --log-level                                                 Logging level [choices: trace, debug, info, notice, warning, success, error, critical] [default: info]          │
│ VERBOSE --verbose                                                -v   Equivalent to --log-level DEBUG. Enables more verbose logging output, but lacks some raw message logging.       │
│                                                                       [default: False]                                                                                                │
│ EXTRA-VERBOSE --extra-verbose                                    -vv  Equivalent to --log-level TRACE. Enables the most verbose logging output possible. [default: False]             │
│ RECORD-PROCESSOR-SERVICE-COUNT --record-processor-service-count       Number of services to spawn for processing records. The higher the request rate, the more services should be    │
│   --record-processors                                                 spawned in order to keep up with the incoming records. If not specified, the number of services will be         │
│                                                                       automatically determined based on the worker count.                                                             │
│ UI-TYPE --ui-type --ui                                                Type of UI to use [choices: dashboard, simple, none] [default: dashboard]                                       │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```
```
╭─ Workers ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ WORKERS-MAX --workers-max --max-workers  Maximum number of workers to create. If not specified, the number of workers will be determined by the smaller of (concurrency + 1) and (num │
│                                          CPUs - 1).                                                                                                                                   │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```