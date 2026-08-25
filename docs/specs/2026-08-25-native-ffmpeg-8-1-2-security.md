# FFmpeg 8.1.2 container security port

## Context

The native AIPerf distribution packages Rust media generation with the shared
runtime container. Upstream commit `ffc943a9fe` upgrades the FFmpeg source
archive from 8.1.1 to 8.1.2 for CVE-2026-8461.

## Requirements

- Build the shipped container from FFmpeg 8.1.2.
- Keep attribution metadata and its source URL at the same version as the
  Docker build argument.
- Add a focused regression that detects drift between those two contracts.
- Preserve the existing FFmpeg configure flags and media-generator behavior.

## Non-goals

- Changing codecs, build flags, or the Rust media-generator interfaces.
- Claiming a rebuilt image without a successful image build.
