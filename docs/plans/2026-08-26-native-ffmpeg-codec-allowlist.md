# Native FFmpeg Codec Allowlist Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the upstream FFmpeg allowlist and make native MP4 automatic audio compatible with it.

**Architecture:** Keep codec selection in the native video generator, retain explicit user codec passthrough, and make the Dockerfile the shipped-image policy boundary. Product E2E verifies generated streams; a Dockerfile contract test verifies the expensive image build inputs.

**Tech Stack:** Rust 2024, native `aiperf` binary E2E, FFmpeg/ffprobe, Dockerfile shell configuration.

**Spec:** `docs/specs/2026-08-26-native-ffmpeg-codec-allowlist.md`

## Global Constraints

- Preserve explicit codec strings for a caller-provided FFmpeg.
- Use `/mnt/4tb/aiperf-target-port046`, `sccache`, clang/lld, and `tokio_unstable` for Rust verification.
- Keep the upstream explicit FFmpeg allowlist and libopus attribution intact.

---

### Task 1: Native automatic audio parity

**Files:**
- Modify: `rust/runtime/src/dataset/generator/video.rs`
- Modify: `rust/runtime/src/dataset/generator/mod.rs`
- Test: `rust/e2e-tests/tests/test_video.rs`

**Interfaces:**
- Consumes: `SyntheticVideoAudioConfig { codec: Option<String> }` and `SyntheticVideoFormat`.
- Produces: MP4 omitted-codec generation invokes FFmpeg with `libopus`; WebM invokes it with `libvorbis`.

- [ ] **Step 1: Write the failing product test**

```rust
video_with_audio_embeds_correct_stream("mp4", "libvpx-vp9", "opus", 48_000).await;
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cargo test -p aiperf-e2e-tests --test test_video test_video_with_audio_embeds_correct_stream_mp4 -- --nocapture`

Expected: FAIL because omitted MP4 audio selects AAC or the selected video codec is unavailable in the allowlisted contract.

- [ ] **Step 3: Make the minimal implementation**

```rust
SyntheticVideoFormat::Mp4 => "libopus",
SyntheticVideoFormat::WebM => "libvorbis",
```

- [ ] **Step 4: Re-run the product test**

Run the command in Step 2. Expected: PASS with an Opus stream at 48,000 Hz.

- [ ] **Step 5: Commit**

```bash
git add rust/runtime/src/dataset/generator/{mod.rs,video.rs} rust/e2e-tests/tests/test_video.rs
git commit -m "fix(dataset): align native video defaults with ffmpeg allowlist"
```

### Task 2: Shipped-image codec policy

**Files:**
- Modify: `Dockerfile`
- Modify: `ATTRIBUTIONS-container.md`
- Test: `rust/e2e-tests/tests/test_video.rs`

**Interfaces:**
- Consumes: Docker `env-builder` FFmpeg configure command.
- Produces: a reproducible minimal FFmpeg codec/runtime dependency surface.

- [ ] **Step 1: Write the failing Dockerfile contract test**

```rust
assert!(dockerfile.contains("--disable-everything"));
assert!(dockerfile.contains("--enable-encoder=libvpx_vp8,libvpx_vp9,libvorbis,libopus"));
assert!(dockerfile.contains("libopus-dev"));
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cargo test -p aiperf-e2e-tests --test test_video test_ffmpeg_container_allowlist_contract -- --nocapture`

Expected: FAIL because the current image builder has neither libopus nor `--disable-everything`.

- [ ] **Step 3: Implement the upstream allowlist**

Copy the exact upstream `e659d2a95a` FFmpeg configure, runtime-copy, package-attribution, and attribution-document policy into the current Dockerfile while retaining current surrounding stages.

- [ ] **Step 4: Re-run the contract and affected native E2E suite**

Run: `cargo test -p aiperf-e2e-tests --test test_video -- --nocapture`

Expected: PASS; WebM remains Vorbis/44,100 Hz and MP4 becomes VP9/Opus/48,000 Hz.

- [ ] **Step 5: Commit**

```bash
git add Dockerfile ATTRIBUTIONS-container.md rust/e2e-tests/tests/test_video.rs
git commit -m "build: restore minimal ffmpeg codec allowlist"
```
