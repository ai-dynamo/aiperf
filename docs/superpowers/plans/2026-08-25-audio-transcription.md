# Audio transcription endpoint

- [x] Audit and merge upstream `00ba1c5db3` with exact ancestry.
- [x] Register the native endpoint descriptor, parser, multipart formatter, and
  mock route.
- [x] Mark the endpoint input as non-tokenized and run the real binary without a
  tokenizer.
- [x] Preserve JSON-versus-raw response precedence.
- [x] Apply turn model precedence and endpoint/turn extra overrides while
  reserving only the multipart file.
- [x] Map supported and unknown audio formats and reject malformed/empty audio
  encodings.
- [x] Propagate mock multipart parser errors and remove the tautological fixture
  test and dead image chain.
- [x] Correct the malformed duplicated CLI-option constraint markup.
- [x] Record the #36 endpoint-policy visibility dependency as inherited scope to
  deduplicate during shared-history integration.
- [x] Run focused runtime, mock-server, and real-binary E2E coverage with the
  isolated shared target.
- [ ] Complete final Graham review with no Important or Critical findings.
