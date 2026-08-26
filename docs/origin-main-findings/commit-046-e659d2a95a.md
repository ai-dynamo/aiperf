# Commit 046 — FFmpeg codec allowlist

## Upstream intent

`e659d2a95a` makes the distributed FFmpeg intentionally small: VP8/VP9 video
and Vorbis/Opus audio only. It changes the MP4 automatic audio selection from
AAC to Opus, keeps user-selected codecs valid against a user-provided FFmpeg,
and documents that the container cannot encode H.264/AAC/NVENC.

## Merge ancestry

The exact upstream commit is already the second parent of the existing actual
two-parent merge `d6c2b69a49` (`6e1c6f024b` + `e659d2a95a`). A new merge of
the same commit into the current port branch is therefore correctly a no-op;
this port supplies the subsequently missing native and current-Dockerfile
semantics rather than manufacturing a duplicate merge.

## Native/shared comparison

The current Dockerfile had regressed to an unrestricted libvpx/vorbis build.
Native Rust chose `aac` for MP4 when audio codec was omitted, and its native
product E2E still required `libx264`/AAC. That cannot run against the shipped
allowlisted image. The native CLI/YAML models intentionally keep explicit
codec strings unconstrained: validation would incorrectly reject a valid
user-owned FFmpeg installation.

## Required result

- Ship FFmpeg with exactly the upstream dependency and configure allowlist,
  including libopus runtime attribution.
- Select `libopus` automatically for MP4 audio, retain `libvorbis` for WebM,
  and document the 48 kHz Opus output rule.
- Exercise real native profile generation for WebM/Vorbis and MP4/VP9/Opus,
  including stream codec/sample-rate inspection.
- Add an executable Dockerfile contract that prevents an unrestricted codec
  build or stale H.264/AAC default from returning.
