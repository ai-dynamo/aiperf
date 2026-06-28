---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Profile Text-to-Speech (TTS) Models with AIPerf
---

# Profile Text-to-Speech (TTS) Models with AIPerf

AIPerf benchmarks text-to-speech (TTS) models served over the OpenAI-compatible
`/v1/audio/speech` API. A text prompt is sent as the `input`, and the server
returns synthesized audio - either a full clip (non-streaming) or a stream of
audio chunks. AIPerf decodes the returned audio to measure how much speech was
produced and how fast.

Use `--endpoint-type speech`.

## TTS-specific metrics

| Metric | Meaning |
|---|---|
| Time to First Audio (TTFA) | Latency to the first audio chunk (streaming only) - the TTS analog of Time to First Token. |
| Output Audio Duration | Seconds of audio synthesized per request, decoded from the returned clip. |
| Real-Time Factor (RTF) | `request_latency / output_audio_duration`. Lower is better; RTF < 1.0 means faster than real-time. |
| Audio Throughput | Seconds of audio synthesized per wall-clock second across the run. Higher is better; exceeds 1.0 under concurrency. |

Standard request-latency and request-throughput metrics are also reported.
Token-based metrics (TTFT, ITL, output token throughput) do not apply because a
TTS endpoint produces audio, not tokens.

> [!NOTE]
> Audio duration is decoded with `soundfile`/libsndfile, which supports
> self-describing containers (`wav`, `flac`, `opus`, and `mp3` on builds with
> MP3 support). A headerless `pcm` response cannot be decoded for duration.

---

## Basic Usage

Profile a TTS server with synthetic text prompts:

```bash
aiperf profile \
    --model tts-1 \
    --endpoint-type speech \
    --tokenizer gpt2 \
    --url http://localhost:8000 \
    --extra-inputs voice:alloy \
    --extra-inputs response_format:mp3 \
    --request-count 10 \
    --concurrency 2
```

The `--tokenizer` is only needed for input token counting (Input Sequence
Length); the audio metrics do not require it.

## Streaming and Time to First Audio

Add `--streaming` to request incremental audio and measure Time to First Audio:

```bash
aiperf profile \
    --model gpt-4o-mini-tts \
    --endpoint-type speech \
    --streaming \
    --url http://localhost:8000 \
    --extra-inputs voice:alloy \
    --request-count 10 \
    --concurrency 4
```

When streaming, AIPerf requests Server-Sent Events (`stream_format: sse`) so
each `speech.audio.delta` chunk is timestamped on arrival. Servers that instead
stream a raw chunked audio body are also supported - AIPerf records per-chunk
timing for both shapes, so TTFA is always measured.

## Passing voice, format, and speed

TTS parameters are passed through `--extra-inputs` and merged into the request
body (per-request `extra` in a dataset overrides these):

```bash
aiperf profile \
    --model tts-1 \
    --endpoint-type speech \
    --url http://localhost:8000 \
    --extra-inputs voice:echo \
    --extra-inputs response_format:wav \
    --extra-inputs speed:1.25 \
    --request-count 10
```

## Custom input text

Synthetic prompts are used by default. To benchmark specific sentences, supply a
custom dataset or input file (any of AIPerf's text dataset paths work), where
each record's text becomes the speech `input`. See
[Custom Dataset](custom-dataset.md) and [Inline Datasets](inline-datasets.md).

## References

- [OpenAI Create Speech API](https://platform.openai.com/docs/api-reference/audio/createSpeech)
