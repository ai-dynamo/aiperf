# Native FFmpeg Codec Allowlist

## Goal

Make the current AIPerf image and native synthetic-video defaults conform to
origin/main commit `e659d2a95a`: the image guarantees VP8/VP9 video plus
Vorbis/Opus audio, and MP4 audio defaults to Opus.

## Contract

The image installs and copies libopus, builds FFmpeg with `--disable-everything`
and the exact upstream encoder, decoder, muxer, demuxer, parser, protocol,
filter, and bitstream-filter allowlists. It remains LGPL/non-free disabled.
The shipped command must contain `ffmpeg` and `ffprobe`, but it must not acquire
an unrestricted codec set through autodetection.

When a native synthetic video enables audio without explicitly choosing a codec,
WebM uses `libvorbis` and MP4 uses `libopus`. Explicit strings remain passed to
the FFmpeg executable: AIPerf does not make its container allowlist a global
validation restriction on custom local installations. Because Opus muxing emits
48 kHz, MP4 auto-audio stream checks expect 48,000 Hz even if the generated WAV
input used a different valid rate.

## Boundaries

This changes only the distributed container and default selection. It does not
remove AAC/H.264/NVENC options from the CLI/YAML public model, add a codec
probing process to the request path, or change explicit-codec behavior.

## Evidence

Native binary E2E must inspect actual media streams. A source-level Dockerfile
contract test must assert the intentional configure and copied-runtime-library
surface, so it catches future regression without requiring an expensive image
build.
