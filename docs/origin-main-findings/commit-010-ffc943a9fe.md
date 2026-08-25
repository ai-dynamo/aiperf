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

## Port closure

The exact upstream commit was incorporated by non-fast-forward merge
`830f579589`. The focused contract regression (`609e5a982a`) establishes that
the Docker build argument is 8.1.2 and that both the FFmpeg attribution version
and official source archive use that same release. It was written RED against
8.1.1, then passed after restoring the coupled upstream values.

Focused `pytest`, Ruff, formatting, and diff checks passed. Direct whole-range
review passed and the direct Graham-rubric review is approved (no findings).

The isolated `docker build --target test` reached the cached FFmpeg 8.1.2 build
layer, then failed in the unchanged wheel-builder stage because `Dockerfile`
copies `rust/` but not the tracked `contracts/native-k8s/v1/*.schema.json`
files required by `cli/src/kube/contract.rs` `include_str!` calls. This
pre-existing build-context defect is outside the FFmpeg port, so no
rebuilt-image claim is made here.
