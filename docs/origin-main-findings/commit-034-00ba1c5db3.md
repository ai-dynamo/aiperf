# Commit 034 — `00ba1c5db3`

Status: complete native behavioral port.

Upstream adds an OpenAI audio-transcription endpoint, multipart audio upload,
synthetic-audio coverage, response parsing, and a mock route. The native port
registers `audio_transcription`, selects `/v1/audio/transcriptions`, and lowers
the final turn's first audio item to multipart `file` metadata without requiring
an input tokenizer.

The port preserves turn-level model selection and lets endpoint/turn extras
override `model`, while `file` remains endpoint-owned. It maps the supported
WAV, MP3/MPGA/MPEG, FLAC, OGG, M4A/MP4, and WebM filename/MIME pairs, preserves
unknown audio formats as `audio/<format>`, and rejects missing or empty
`<fmt>,<b64>` input. Response parsing accepts raw text only when no decoded JSON
body exists; a decoded JSON body without `text` is invalid even when raw bytes
are retained.

The Rust mock route validates the file, echoes authored language/temperature
fields, and returns multipart parser errors instead of treating a malformed
tail as a successful request. The real-binary E2E covers synthetic audio,
native multipart dispatch, response parsing, and raw artifacts without a
tokenizer.

The two-line `capture_endpoint_policy` visibility dependency in the branch is
identical to the fix already integrated by origin-port #36. Keep one copy when
this branch is integrated into the shared history.

Validation uses `sccache` with
`CARGO_TARGET_DIR=/mnt/4tb/aiperf-origin-port-034-target`:

- focused runtime endpoint tests: 5 passed;
- focused mock-server integration tests: 2 passed;
- real-binary audio-transcription E2E: 13 passed.

Final Graham re-review: approved with no Critical or Important findings after
two passes over the complete Rust branch diff.
