# Origin/main commit 010 finding: FFmpeg 8.1.2 security update

Upstream `ffc943a9fe1a1b6ecf88b64ee2ad2b60a622e696` changes the shipped
container's FFmpeg source from 8.1.1 to 8.1.2 for CVE-2026-8461 and updates
the corresponding attribution version and release URL. The native CLI image
uses the same top-level `Dockerfile`; its Rust media generators invoke the
image's `ffmpeg` executable. This is a shared-product security action, not a
runtime-source behavior change, but it is applicable to the native product and
must be merged and verified at the container-contract level.

The port must retain the coupled Dockerfile and attribution update, with a
focused static regression that fails if either version diverges. A full image
build is the final integration verification when the current build capacity is
available.
