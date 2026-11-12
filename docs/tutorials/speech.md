---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Profile Speech Generation
---

# Profile Speech Generation

Use `--endpoint-type speech` to benchmark OpenAI-compatible
`POST /v1/audio/speech` servers. The endpoint accepts one text input per request
and validates complete PCM or uncompressed PCM WAV responses. It reports request
latency, request throughput, output audio duration, output audio real-time factor
(RTF), and aggregate output audio throughput.

The HTTP client reads the complete response before recording its timestamp.
These metrics describe full-response performance; time to first audio and
playback continuity are not measured. Do not interpret them as streaming latency.
Token metrics and ASR input-audio metrics do not apply to this endpoint.

## Start a Phonon server

This example uses Gradium's Phonon CPU server. The public image is pinned to the
multi-platform digest of version 0.6.0 and the model to a Hugging Face revision.
The first start downloads model assets. No API key or GPU is required.

<!-- setup-phonon-speech-openai-endpoint-server -->
```bash
docker run --rm -p 8880:8880 -e RAYON_NUM_THREADS=4 \
    ghcr.io/gradium-ai/ptts-openai-server@sha256:442230282e5966813e352036d9f6a82ba1727a9e07c09bb9938b68ed7b026ad1 \
    --model hf://gradium/phonon-31m-en@92741c5b07d085ff840e26fea8ce43110b87852f \
    --lang en --device cpu --quant q8 --voice-catalog "" \
    --max-concurrent-requests 1
```
<!-- /setup-phonon-speech-openai-endpoint-server -->

Wait for the server to load:

<!-- health-check-phonon-speech-openai-endpoint-server -->
```bash
for attempt in $(seq 1 180); do
    if curl -fsS http://localhost:8880/health; then exit 0; fi
    sleep 2
done
exit 1
```
<!-- /health-check-phonon-speech-openai-endpoint-server -->

## Profile PCM output

Headerless PCM cannot identify its own rate or channel count. Set
`benchmark.endpoint.pcmOutput` in a config file to the server's actual format.
The supported sample format is signed 16-bit little-endian (`s16le`). This
metadata is retained in the benchmark configuration and is **not sent** to the
server. Other PCM encodings must be converted by the server or use a supported
WAV response instead.

The request's voice, response format, speed and other server-specific inputs
can be supplied through `endpoint.extra`, `--extra-inputs`, or each dataset
row's `extra` object. Per-row values override endpoint extras. A voice must be
specified; AIPerf does not assume that different servers share voice names.

<!-- aiperf-run-phonon-speech-openai-endpoint-server weight=60 -->
```bash
cat > speech.yaml <<EOF_CONFIG
schemaVersion: "2.0"
benchmark:
  model: phonon-31m-en
  endpoint:
    type: speech
    url: http://localhost:8880
    pcmOutput:
      sampleRate: 24000
      channels: 1
      sampleFormat: s16le
    extra:
      voice: default
      response_format: pcm
  dataset:
    type: file
    format: single_turn
    records:
      - text: "Your order is ready. Thank you for waiting."
  warmup:
    type: concurrency
    concurrency: 1
    requests: 3
  profiling:
    type: concurrency
    concurrency: 1
    requests: 20
EOF_CONFIG

aiperf profile --config speech.yaml
```
<!-- /aiperf-run-phonon-speech-openai-endpoint-server -->

Change the URL, model and voice for another compatible server. For WAV output,
set `response_format: wav` and remove `pcmOutput`: duration is derived from the
WAV header and validated sample frames. WAV is the endpoint's default response
format. Only finite-length uncompressed PCM WAV is supported; unknown-length
streaming WAV headers, compressed WAV, MP3, Opus, SSE audio deltas and other
codecs are not supported by this initial endpoint.

## Interpret the results

| Metric | Meaning |
|---|---|
| Request Latency | Time to receive the complete response |
| Output Audio Duration | Complete sample frames / sample rate, in seconds |
| Output Audio Real-Time Factor | Request latency in seconds / output duration; lower is better |
| Output Audio Throughput | Total successful output audio seconds / benchmark observation seconds |
| Request Error Rate | Includes HTTP failures and invalid or incomplete audio responses |

For example, generating two seconds of audio in 0.1 seconds gives an RTF of
0.05. With concurrent requests, aggregate throughput is computed from total
successful audio and the observation wall time, not an average of per-request
inverse RTFs. Failed and cancelled responses do not contribute audio duration.

PCM bodies must contain at least one complete interleaved sample frame. WAV
headers and sample counts must agree with the received body. A headerless PCM
body that ends cleanly on a sample boundary cannot reveal whether the server
stopped synthesis early; evaluating linguistic completeness requires a separate
quality evaluation. These checks do not measure intelligibility or speaker quality.

Use natural text with representative lengths and voice frequencies for meaningful
performance comparisons. Keep model revision, sampling parameters, hardware,
thread counts, concurrency and warm-up fixed, and retain errors and per-request
exports. Increase client concurrency only together with the server admission
limit when measuring admitted capacity; Phonon returns HTTP 429 for excess load.
