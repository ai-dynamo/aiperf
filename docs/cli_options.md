<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# CLI Options
Use these options to profile with AIPerf.

```
╭─ Endpoint ───────────────────────────────────────────────────────────────────╮
│ *  MODEL-NAMES --model-names     -m  Model name(s) to be benchmarked. Can be │
│      --model                         a comma-separated list or a single      │
│                                      model name. [required]                  │
│    MODEL-SELECTION-STRATEGY          When multiple models are specified,     │
│      --model-selection-strategy      this is how a specific model should be  │
│                                      assigned to a prompt. round_robin: nth  │
│                                      prompt in the list gets assigned to     │
│                                      n-mod len(models). random: assignment   │
│                                      is uniformly random [choices:           │
│                                      round-robin, random] [default:          │
│                                      round-robin]                            │
│    CUSTOM-ENDPOINT                   Set a custom endpoint that differs from │
│      --custom-endpoint               the OpenAI defaults.                    │
│      --endpoint                                                              │
│    ENDPOINT-TYPE                     The endpoint type to send requests to   │
│      --endpoint-type                 on the server. [choices: chat,          │
│                                      completions, cohere-rankings,           │
│                                      embeddings, hf-tei-rankings,            │
│                                      huggingface-generate, nim-rankings,     │
│                                      solido-rag, template] [default: chat]   │
│    STREAMING --streaming             An option to enable the use of the      │
│                                      streaming API. [default: False]         │
│    URL --url                     -u  URL of the endpoint to target for       │
│                                      benchmarking. [default: localhost:8000] │
│    REQUEST-TIMEOUT-SECONDS           The timeout in floating-point seconds   │
│      --request-timeout-seconds       for each request to the endpoint.       │
│                                      [default: 600.0]                        │
│    API-KEY --api-key                 The API key to use for the endpoint. If │
│                                      provided, it will be sent with every    │
│                                      request as a header: Authorization:     │
│                                      Bearer <api_key>.                       │
│    TRANSPORT --transport             The transport to use for the endpoint.  │
│      --transport-type                If not provided, it will be             │
│                                      auto-detected from the URL.This can     │
│                                      also be used to force an alternative    │
│                                      transport or implementation. [choices:  │
│                                      http]                                   │
╰──────────────────────────────────────────────────────────────────────────────╯
```
```
╭─ Input ──────────────────────────────────────────────────────────────────────╮
│ EXTRA-INPUTS --extra-inputs       Provide additional inputs to include with  │
│                                   every request. Inputs should be in an      │
│                                   'input_name:value' format. Alternatively,  │
│                                   a string representing a json formatted     │
│                                   dict can be provided. [default: []]        │
│ HEADER --header               -H  Adds a custom header to the requests.      │
│                                   Headers must be specified as               │
│                                   'Header:Value' pairs. Alternatively, a     │
│                                   string representing a json formatted dict  │
│                                   can be provided. [default: []]             │
│ INPUT-FILE --input-file           The file or directory path that contains   │
│                                   the dataset to use for profiling. This     │
│                                   parameter is used in conjunction with the  │
│                                   custom_dataset_type parameter to support   │
│                                   different types of user provided datasets. │
│ FIXED-SCHEDULE                    Specifies to run a fixed schedule of       │
│   --fixed-schedule                requests. This is normally inferred from   │
│                                   the --input-file parameter, but can be set │
│                                   manually here. [default: False]            │
│ FIXED-SCHEDULE-AUTO-OFFSET        Specifies to automatically offset the      │
│   --fixed-schedule-auto-offs      timestamps in the fixed schedule, such     │
│   et                              that the first timestamp is considered 0,  │
│                                   and the rest are shifted accordingly. If   │
│                                   disabled, the timestamps will be assumed   │
│                                   to be relative to 0. [default: False]      │
│ FIXED-SCHEDULE-START-OFFSET       Specifies the offset in milliseconds to    │
│   --fixed-schedule-start-off      start the fixed schedule at. By default,   │
│   set                             the schedule starts at 0, but this option  │
│                                   can be used to start at a reference point  │
│                                   further in the schedule. This option       │
│                                   cannot be used in conjunction with the     │
│                                   --fixed-schedule-auto-offset. The schedule │
│                                   will include any requests at the start     │
│                                   offset.                                    │
│ FIXED-SCHEDULE-END-OFFSET         Specifies the offset in milliseconds to    │
│   --fixed-schedule-end-offse      end the fixed schedule at. By default, the │
│   t                               schedule ends at the last timestamp in the │
│                                   trace dataset, but this option can be used │
│                                   to only run a subset of the trace. The     │
│                                   schedule will include any requests at the  │
│                                   end offset.                                │
│ PUBLIC-DATASET                    The public dataset to use for the          │
│   --public-dataset                requests. [choices: sharegpt]              │
│ CUSTOM-DATASET-TYPE               The type of custom dataset to use. This    │
│   --custom-dataset-type           parameter is used in conjunction with the  │
│                                   --input-file parameter. [choices:          │
│                                   single_turn, multi_turn, random_pool,      │
│                                   mooncake_trace]                            │
│ DATASET-SAMPLING-STRATEGY         The strategy to use for sampling the       │
│   --dataset-sampling-strateg      dataset. sequential: Iterate through the   │
│   y                               dataset sequentially, then wrap around to  │
│                                   the beginning. random: Randomly select a   │
│                                   conversation from the dataset. Will        │
│                                   randomly sample with replacement. shuffle: │
│                                   Shuffle the dataset and iterate through    │
│                                   it. Will randomly sample without           │
│                                   replacement. Once the end of the dataset   │
│                                   is reached, shuffle the dataset again and  │
│                                   start over. [choices: sequential, random,  │
│                                   shuffle]                                   │
│ RANDOM-SEED --random-seed         The seed used to generate random values.   │
│                                   Set to some value to make the synthetic    │
│                                   data generation deterministic. It will use │
│                                   system default if not provided.            │
│ GOODPUT --goodput                 Specify service level objectives (SLOs)    │
│                                   for goodput as space-separated 'KEY:VALUE' │
│                                   pairs, where KEY is a metric tag and VALUE │
│                                   is a number in the metric’s display unit   │
│                                   (falls back to its base unit if no display │
│                                   unit is defined). Examples:                │
│                                   'request_latency:250' (ms),                │
│                                   'inter_token_latency:10' (ms),             │
│                                   output_token_throughput_per_user:600       │
│                                   (tokens/s). Only metrics applicable to the │
│                                   current endpoint/config are considered.    │
│                                   For more context on the definition of      │
│                                   goodput, refer to DistServe paper:         │
│                                   https://arxiv.org/pdf/2401.09670 and the   │
│                                   blog:                                      │
│                                   https://hao-ai-lab.github.io/blogs/distser │
│                                   ve                                         │
╰──────────────────────────────────────────────────────────────────────────────╯
```
```
╭─ Output ─────────────────────────────────────────────────────────────────────╮
│ OUTPUT-ARTIFACT-DIR          The directory to store all the (output)         │
│   --output-artifact-dir      artifacts generated by AIPerf. [default:        │
│   --artifact-dir             artifacts]                                      │
│ PROFILE-EXPORT-PREFIX        The prefix for the profile export file names.   │
│   --profile-export-prefix    Will be suffixed with .csv, .json, .jsonl, and  │
│   --profile-export-file      _raw.jsonl.If not provided, the default profile │
│                              export file names will be used:                 │
│                              profile_export_aiperf.csv,                      │
│                              profile_export_aiperf.json,                     │
│                              profile_export.jsonl, and                       │
│                              profile_export_raw.jsonl.                       │
│ EXPORT-LEVEL --export-level  The level of profile export files to create.    │
│   --profile-export-level     [choices: summary, records, raw] [default:      │
│                              records]                                        │
╰──────────────────────────────────────────────────────────────────────────────╯
```
```
╭─ Tokenizer ──────────────────────────────────────────────────────────────────╮
│ TOKENIZER --tokenizer         The HuggingFace tokenizer to use to interpret  │
│                               token metrics from prompts and responses. The  │
│                               value can be the name of a tokenizer or the    │
│                               filepath of the tokenizer. The default value   │
│                               is the model name.                             │
│ TOKENIZER-REVISION            The specific model version to use. It can be a │
│   --tokenizer-revision        branch name, tag name, or commit ID. [default: │
│                               main]                                          │
│ TOKENIZER-TRUST-REMOTE-CODE   Allows custom tokenizer to be downloaded and   │
│   --tokenizer-trust-remote-c  executed. This carries security risks and      │
│   ode                         should only be used for repositories you       │
│                               trust. This is only necessary for custom       │
│                               tokenizers stored in HuggingFace Hub.          │
│                               [default: False]                               │
╰──────────────────────────────────────────────────────────────────────────────╯
```
```
╭─ Load Generator ─────────────────────────────────────────────────────────────╮
│ BENCHMARK-DURATION            The duration in seconds for benchmarking.      │
│   --benchmark-duration                                                       │
│ BENCHMARK-GRACE-PERIOD        The grace period in seconds to wait for        │
│   --benchmark-grace-period    responses after benchmark duration ends. Only  │
│                               applies when --benchmark-duration is set.      │
│                               Responses received within this period are      │
│                               included in metrics. [default: 30.0]           │
│ CONCURRENCY --concurrency     The concurrency value to benchmark.            │
│ REQUEST-RATE --request-rate   Sets the request rate for the load generated   │
│                               by AIPerf. Unit: requests/second               │
│ REQUEST-RATE-MODE             Sets the request rate mode for the load        │
│   --request-rate-mode         generated by AIPerf. Valid values: constant,   │
│                               poisson. constant: Generate requests at a      │
│                               fixed rate. poisson: Generate requests using a │
│                               poisson distribution. [default: poisson]       │
│ REQUEST-COUNT                 The number of requests to use for measurement. │
│   --request-count             [default: 10]                                  │
│   --num-requests                                                             │
│ WARMUP-REQUEST-COUNT          The number of warmup requests to send before   │
│   --warmup-request-count      benchmarking. [default: 0]                     │
│   --num-warmup-requests                                                      │
│ REQUEST-CANCELLATION-RATE     The percentage of requests to cancel.          │
│   --request-cancellation-rat  [default: 0.0]                                 │
│   e                                                                          │
│ REQUEST-CANCELLATION-DELAY    The delay in seconds before cancelling         │
│   --request-cancellation-del  requests. This is used when                    │
│   ay                          --request-cancellation-rate is greater than 0. │
│                               [default: 0.0]                                 │
╰──────────────────────────────────────────────────────────────────────────────╯
```
```
╭─ Conversation Input ─────────────────────────────────────────────────────────╮
│ CONVERSATION-NUM              The total number of unique conversations to    │
│   --conversation-num          generate. Each conversation represents a       │
│   --num-conversations         single request session between client and      │
│   --num-sessions              server. Supported on synthetic mode and the    │
│                               custom random_pool dataset. The number of      │
│                               conversations will be used to determine the    │
│                               number of entries in both the custom           │
│                               random_pool and synthetic datasets and will be │
│                               reused until benchmarking is complete.         │
│ NUM-DATASET-ENTRIES           The total number of unique dataset entries to  │
│   --num-dataset-entries       generate for the dataset. Each entry           │
│   --num-prompts               represents a single turn used in a request.    │
│                               [default: 100]                                 │
│ CONVERSATION-TURN-MEAN        The mean number of turns within a              │
│   --conversation-turn-mean    conversation. [default: 1]                     │
│   --session-turns-mean                                                       │
│ CONVERSATION-TURN-STDDEV      The standard deviation of the number of turns  │
│   --conversation-turn-stddev  within a conversation. [default: 0]            │
│   --session-turns-stddev                                                     │
│ CONVERSATION-TURN-DELAY-MEAN  The mean delay between turns within a          │
│   --conversation-turn-delay-  conversation in milliseconds. [default: 0.0]   │
│   mean                                                                       │
│   --session-turn-delay-mean                                                  │
│ CONVERSATION-TURN-DELAY-STDD  The standard deviation of the delay between    │
│   EV --conversation-turn-del  turns within a conversation in milliseconds.   │
│   ay-stddev --session-turn-d  [default: 0.0]                                 │
│   elay-stddev                                                                │
│ CONVERSATION-TURN-DELAY-RATI  A ratio to scale multi-turn delays. [default:  │
│   O --conversation-turn-dela  1.0]                                           │
│   y-ratio                                                                    │
│   --session-delay-ratio                                                      │
╰──────────────────────────────────────────────────────────────────────────────╯
```
```
╭─ Input Sequence Length (ISL) ────────────────────────────────────────────────╮
│ PROMPT-INPUT-TOKENS-MEAN      The mean of number of tokens in the generated  │
│   --prompt-input-tokens-mean  prompts when using synthetic data. [default:   │
│   --synthetic-input-tokens-m  550]                                           │
│   ean --isl                                                                  │
│ PROMPT-INPUT-TOKENS-STDDEV    The standard deviation of number of tokens in  │
│   --prompt-input-tokens-stdd  the generated prompts when using synthetic     │
│   ev --synthetic-input-token  data. [default: 0.0]                           │
│   s-stddev --isl-stddev                                                      │
│ PROMPT-INPUT-TOKENS-BLOCK-SI  The block size of the prompt. [default: 512]   │
│   ZE --prompt-input-tokens-b                                                 │
│   lock-size --synthetic-inpu                                                 │
│   t-tokens-block-size                                                        │
│   --isl-block-size                                                           │
│ SEQ-DIST --seq-dist           Sequence length distribution specification for │
│   --sequence-distribution     varying ISL/OSL pairs                          │
╰──────────────────────────────────────────────────────────────────────────────╯
```
```
╭─ Output Sequence Length (OSL) ───────────────────────────────────────────────╮
│ PROMPT-OUTPUT-TOKENS-MEAN     The mean number of tokens in each output.      │
│   --prompt-output-tokens-mea                                                 │
│   n --output-tokens-mean                                                     │
│   --osl                                                                      │
│ PROMPT-OUTPUT-TOKENS-STDDEV   The standard deviation of the number of tokens │
│   --prompt-output-tokens-std  in each output. [default: 0]                   │
│   dev --output-tokens-stddev                                                 │
│   --osl-stddev                                                               │
╰──────────────────────────────────────────────────────────────────────────────╯
```
```
╭─ Prompt ─────────────────────────────────────────────────────────────────────╮
│ PROMPT-BATCH-SIZE      -b  The batch size of text requests AIPerf should     │
│   --prompt-batch-size      send. This is currently supported with the        │
│   --batch-size-text        embeddings and rankings endpoint types [default:  │
│   --batch-size             1]                                                │
╰──────────────────────────────────────────────────────────────────────────────╯
```
```
╭─ Prefix Prompt ──────────────────────────────────────────────────────────────╮
│ PROMPT-PREFIX-POOL-SIZE      The total size of the prefix prompt pool to     │
│   --prompt-prefix-pool-size  select prefixes from. If this value is not      │
│   --prefix-prompt-pool-size  zero, these are prompts that are prepended to   │
│   --num-prefix-prompts       input prompts. This is useful for benchmarking  │
│                              models that use a K-V cache. [default: 0]       │
│ PROMPT-PREFIX-LENGTH         The number of tokens in each prefix prompt.     │
│   --prompt-prefix-length     This is only used if "num" is greater than      │
│   --prefix-prompt-length     zero. Note that due to the prefix and user      │
│                              prompts being concatenated, the number of       │
│                              tokens in the final prompt may be off by one.   │
│                              [default: 0]                                    │
╰──────────────────────────────────────────────────────────────────────────────╯
```
```
╭─ Audio Input ────────────────────────────────────────────────────────────────╮
│ AUDIO-BATCH-SIZE             The batch size of audio requests AIPerf should  │
│   --audio-batch-size         send. This is currently supported with the      │
│   --batch-size-audio         OpenAI chat endpoint type [default: 1]          │
│ AUDIO-LENGTH-MEAN            The mean length of the audio in seconds.        │
│   --audio-length-mean        [default: 0.0]                                  │
│ AUDIO-LENGTH-STDDEV          The standard deviation of the length of the     │
│   --audio-length-stddev      audio in seconds. [default: 0.0]                │
│ AUDIO-FORMAT --audio-format  The format of the audio files (wav or mp3).     │
│                              [choices: wav, mp3] [default: wav]              │
│ AUDIO-DEPTHS --audio-depths  A list of audio bit depths to randomly select   │
│                              from in bits. [default: [16]]                   │
│ AUDIO-SAMPLE-RATES           A list of audio sample rates to randomly select │
│   --audio-sample-rates       from in kHz. Common sample rates are 16, 44.1,  │
│                              48, 96, etc. [default: [16.0]]                  │
│ AUDIO-NUM-CHANNELS           The number of audio channels to use for the     │
│   --audio-num-channels       audio data generation. [default: 1]             │
╰──────────────────────────────────────────────────────────────────────────────╯
```
```
╭─ Image Input ────────────────────────────────────────────────────────────────╮
│ IMAGE-WIDTH-MEAN             The mean width of images when generating        │
│   --image-width-mean         synthetic image data. [default: 0.0]            │
│ IMAGE-WIDTH-STDDEV           The standard deviation of width of images when  │
│   --image-width-stddev       generating synthetic image data. [default: 0.0] │
│ IMAGE-HEIGHT-MEAN            The mean height of images when generating       │
│   --image-height-mean        synthetic image data. [default: 0.0]            │
│ IMAGE-HEIGHT-STDDEV          The standard deviation of height of images when │
│   --image-height-stddev      generating synthetic image data. [default: 0.0] │
│ IMAGE-BATCH-SIZE             The image batch size of the requests AIPerf     │
│   --image-batch-size         should send. [default: 1]                       │
│   --batch-size-image                                                         │
│ IMAGE-FORMAT --image-format  The compression format of the images. [choices: │
│                              png, jpeg, random] [default: png]               │
╰──────────────────────────────────────────────────────────────────────────────╯
```
```
╭─ Video Input ────────────────────────────────────────────────────────────────╮
│ VIDEO-BATCH-SIZE             The video batch size of the requests AIPerf     │
│   --video-batch-size         should send. [default: 1]                       │
│   --batch-size-video                                                         │
│ VIDEO-DURATION               Seconds per clip (default: 5.0). [default: 5.0] │
│   --video-duration                                                           │
│ VIDEO-FPS --video-fps        Frames per second (default/recommended for      │
│                              Cosmos: 4). [default: 4]                        │
│ VIDEO-WIDTH --video-width    Video width in pixels.                          │
│ VIDEO-HEIGHT --video-height  Video height in pixels.                         │
│ VIDEO-SYNTH-TYPE             Synthetic generator type. [choices:             │
│   --video-synth-type         moving-shapes, grid-clock] [default:            │
│                              moving-shapes]                                  │
│ VIDEO-FORMAT --video-format  The video format of the generated files.        │
│                              [choices: mp4] [default: mp4]                   │
│ VIDEO-CODEC --video-codec    The video codec to use for encoding. Common     │
│                              options: libx264 (CPU, widely compatible),      │
│                              libx265 (CPU, smaller files), h264_nvenc        │
│                              (NVIDIA GPU), hevc_nvenc (NVIDIA GPU, smaller   │
│                              files). Any FFmpeg-supported codec can be used. │
│                              [default: libx264]                              │
╰──────────────────────────────────────────────────────────────────────────────╯
```
```
╭─ Service ────────────────────────────────────────────────────────────────────╮
│ LOG-LEVEL --log-level              Logging level [choices: trace, debug,     │
│                                    info, notice, warning, success, error,    │
│                                    critical] [default: info]                 │
│ VERBOSE --verbose             -v   Equivalent to --log-level DEBUG. Enables  │
│                                    more verbose logging output, but lacks    │
│                                    some raw message logging. [default:       │
│                                    False]                                    │
│ EXTRA-VERBOSE                 -vv  Equivalent to --log-level TRACE. Enables  │
│   --extra-verbose                  the most verbose logging output possible. │
│                                    [default: False]                          │
│ RECORD-PROCESSOR-SERVICE-COU       Number of services to spawn for           │
│   NT --record-processor-serv       processing records. The higher the        │
│   ice-count                        request rate, the more services should be │
│   --record-processors              spawned in order to keep up with the      │
│                                    incoming records. If not specified, the   │
│                                    number of services will be automatically  │
│                                    determined based on the worker count.     │
│ UI-TYPE --ui-type --ui             Type of UI to use [choices: none, simple, │
│                                    dashboard] [default: dashboard]           │
╰──────────────────────────────────────────────────────────────────────────────╯
```
```
╭─ Telemetry ──────────────────────────────────────────────────────────────────╮
│ GPU-TELEMETRY      Enable GPU telemetry console display and optionally       │
│   --gpu-telemetry  specify custom DCGM exporter URLs (e.g.,                  │
│                    http://node1:9401/metrics http://node2:9401/metrics).     │
│                    Default localhost:9400 and localhost:9401 are always      │
│                    attempted                                                 │
╰──────────────────────────────────────────────────────────────────────────────╯
```
```
╭─ Workers ────────────────────────────────────────────────────────────────────╮
│ WORKERS-MAX --workers-max  Maximum number of workers to create. If not       │
│   --max-workers            specified, the number of workers will be          │
│                            determined by the formula min(concurrency, (num   │
│                            CPUs * 0.75) - 1),  with a default max cap of 32. │
│                            Any value provided will still be capped by the    │
│                            concurrency value (if specified), but not by the  │
│                            max cap.                                          │
╰──────────────────────────────────────────────────────────────────────────────╯
```
```
╭─ ZMQ Communication ──────────────────────────────────────────────────────────╮
│ ZMQ-HOST --zmq-host          Host address for TCP connections [default:      │
│                              127.0.0.1]                                      │
│ ZMQ-IPC-PATH --zmq-ipc-path  Path for IPC sockets                            │
╰──────────────────────────────────────────────────────────────────────────────╯
```