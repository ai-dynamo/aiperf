# AWS SigV4 signing and native SageMaker support

**Date:** 2026-09-11
**Issues:** AIP-1181 (SigV4), AIP-1177 (SageMaker)
**Status:** Design, approved

## Why this document exists

Two related pieces of work were conflated into one PR. PR #1408 was built on a plan
that treated SigV4 signing as "step 1 of 3" of SageMaker support. It is not: SigV4
already had its own Linear issue (AIP-1181), owner (Anthony Casagrande), and open PR
(#771) with 25 reviews. #1408 therefore reimplemented work already in review, and did
so less completely — one signing call site against #771's eight.

Anthony has handed AIP-1181 over. This document defines the boundary between the two
issues, what each delivers, and the order they land in.

## The boundary

The dividing line is **who owns the credential contract**.

- **AIP-1181** owns how a request is *authenticated*: the signer, its lifecycle, which
  requests get signed, and the configuration surface that describes signing.
- **AIP-1177** owns where a request *goes* and how its response is *framed*: URL
  construction, service headers, and eventstream decoding.

Three things sit in AIP-1181 that look like SageMaker concerns but are not, because
they are signing-layer API that SageMaker merely consumes:

1. The rule for which transports may sign.
2. The TLS requirement and its exemptions.
3. How the SigV4 credential scope (signing name) is determined.

If these lived in AIP-1177, AIP-1181 would ship a signer structurally incapable of
serving a second transport, and Bedrock would repeat the work.

## AIP-1181 — completing #771

`ajc/sigv4-support`, currently a draft, 3 commits behind `main`.

### Already done (do not re-port)

Verified present on the branch: out-of-band signing across readiness probe,
control-plane hooks, timing manager and video submit/poll/download (8 call sites total,
via a `sign_request()` helper and an `endpoint_signer()` lifecycle context manager);
HTTPS enforcement (`31731ac35`); off-loop initial credential resolution (`53fd2c84d`);
signed-readiness 401/403 handling (`a2d92f3f9`); control-plane signer failures converted
to `ControlPlaneHttpError` (`2ba313664`); multipart rejection; `x-api-key`/Bearer
suppression under an active signer; the `aws` extra in the Dockerfile.

**AIP-1181's description is stale** and says HTTPS enforcement is pending. Correct it on
takeover.

### Five changes to finish it

**1. Relax the transport gate — required for AIP-1177.**

`config/endpoint.py` currently rejects any transport that is not `TransportType.HTTP`:

```python
if self.transport is not None and self.transport != TransportType.HTTP:
    raise ValueError("--auth-type sigv4 requires HTTP transport; ...")
```

Its stated reason is that only `aiohttp_transport` calls `_sign_if_needed`. That
reasoning is sound today but the condition is wrong: `SageMakerTransport` subclasses
`AioHttpTransport` and inherits the signing call site. The gate must test *the
capability* — that the selected transport signs — not identity with `HTTP`. Left
unchanged, every SageMaker configuration is rejected at startup.

**2. Exempt loopback from the HTTPS requirement — required for AIP-1177.**

The same commit requires `https://` for every configured URL, with no exemption. This
makes the AWS path untestable locally and breaks aiperf's own mock-server workflow,
which the SageMaker tutorial documents and which is the end-to-end merge gate for
AIP-1177. Exempt `localhost`, `127.0.0.0/8` and `::1`: headers that never leave the
machine are not disclosed by the absence of TLS. Remote cleartext stays rejected.

**3. Rename `--aws-service` to `--aws-signing-service` and make it derivable — required
for AIP-1177.**

The value is the SigV4 *signing name*, i.e. the credential scope, which is not the API
identifier: the `sagemaker-runtime` API signs as `sagemaker`, and `bedrock-runtime` as
`bedrock`. The current flag name invites using it as an adapter selector, which is the
confusion that made the original proof of concept branch on a magic string.

Transports declare a `botocore_service_id` class attribute; the signer resolves the
signing name from botocore's own service model, falling back to the service id if the
lookup fails. `--aws-signing-service` becomes an override rather than a requirement.
This is what reduces SageMaker's common case to one flag, and it makes Bedrock correct
without further work. Nothing has shipped, so the rename costs nothing now and cannot be
done later.

**4. Do not follow redirects on signed requests — security.**

aiohttp strips `Authorization` when a redirect crosses origins, but applies no such rule
to custom headers, so `X-Amz-Security-Token` — a bearer credential — would be replayed
to the redirect target. Set `allow_redirects=False` when a signer is active. Nothing is
lost: SigV4 signs the `Host` header, so a replayed signature is invalid at the new origin
regardless.

This also requires fixing `post_request`, which drops `**kwargs` on its cancellation
branch; without that fix the guard silently does not apply to cancellable requests. A
3xx reaching the caller should carry a message explaining the refusal, since "302 Found"
alone does not.

**5. Add `aiperf[aws]` to the `test` extra — test integrity.**

The AWS suite self-skips via `pytest.importorskip("botocore")`, and the Windows-on-ARM
CI job installs `--extra test --no-dev`. That job therefore skips the entire AWS suite
and reports green, so its tick proves nothing about this code. botocore ships a
pure-Python wheel, so unlike the dependencies `dev` pulls it installs there without
issue.

### Gate

#771 is ready for review only when the AWS test suite demonstrably executes on every CI
platform, not merely passes.

## AIP-1177 — SageMaker, stacked on AIP-1181

Contains everything from #1408 except the authentication layer.

- `transports/aws/eventstream.py` — the reader, `AwsEventStreamMessage`, content-type
  dispatch, typed errors carrying AWS's `:exception-type`/`:error-code` label, raw-line
  fidelity, and one timestamp per network read (matching `sse_utils.py`, so eventstream
  ITL stays comparable to SSE ITL).
- `transports/aws/sagemaker.py` and `regions.py` — the transport subclass, header
  constants taken from botocore's service model, and a pure DNS-suffix helper with no
  botocore dependency.
- `config/sagemaker.py`, the derivation validators, and the six-hop CLI-to-`EndpointInfo`
  threading.
- CRD rules permitting a SageMaker resource to omit `urls`, and requiring `awsRegion`
  alongside `sagemaker.endpointName`.
- Mock-server routes and frame encoder, the tutorial, and the test suite.

It declares `botocore_service_id = "sagemaker-runtime"` and otherwise knows nothing about
signing.

### Design decisions carried forward from #1408

These were validated in review and should not be relitigated:

- **A transport subclass, not a new plugin category.** SageMaker differs from a plain
  OpenAI-compatible server only in auth, URL, headers and response framing — never in
  the request body. `url_schemes` is empty so URL-scheme auto-detection can never select
  it.
- **No endpoint plugin.** `AwsEventStreamMessage` satisfies the existing structural
  `InferenceServerResponse` protocol, which is why no parser and no metrics code changed.
  `--endpoint-type chat`/`completions`/`embeddings` all keep working.
- **The endpoint plugin's path is deliberately ignored.** SageMaker routes on the request
  body, so every endpoint type is invoked through `/endpoints/{name}/invocations`. This
  is the single most surprising thing for a first-time user and is called out in the
  tutorial and pinned by a test.
- **SageMaker settings are a nested config section**, following the
  `reset_kv_cache`/`server_profiler` precedent, because `EndpointConfig` otherwise
  exceeds the repo's field-count guardrail. CLI flags stay flat.

## Sequencing

1. Take over #771: rebase onto `main`, reassign AIP-1181, correct its description.
2. Land the five changes. Verify the gate.
3. Mark #771 ready; merge. AIP-1181 closes.
4. Branch AIP-1177 from #771's head, port the SageMaker content, open its PR with the
   base set to #771's branch so reviewers see only the SageMaker delta.
5. After #771 merges, rebase onto `main`; merge. AIP-1177 closes.
6. Set AIP-1177 `blockedBy` AIP-1181. Close #1408 with pointers to both successors.

#1408 is closed, not merged. Its resolved review findings travel with the content.

## Verification

Each phase keeps its own gate.

- **AIP-1181:** full unit suite; `make check-ergonomics` and `make check-ruff-baselined`
  with no new baseline entries; the AWS suite observed executing (not skipping) on every
  CI platform.
- **AIP-1177:** the same, plus the end-to-end gate that has already caught real defects —
  drive the mock SageMaker route with the real CLI and confirm from the mock server's own
  metrics that traffic reached `/endpoints/{name}/invocations-response-stream`, with TTFT
  and ITL decoded from real binary eventstream frames. This depends on change 2 above.

Real-endpoint validation stays with Salesforce, per the Linear thread; mock-server
coverage is the merge gate on our side.

## Out of scope

Bedrock — the seams are in place (content-type dispatch, `botocore_service_id`
derivation) but no transport ships. SageMaker Asynchronous and Serverless Inference,
which exchange payloads through S3 rather than the HTTP body.

## A note on overriding prior decisions

Changes 1 and 2 modify reasoning Anthony recorded deliberately: he restricted signing to
the HTTP transport and required HTTPS everywhere, both with comments explaining why. Both
are correct for a world without a second transport. They are being changed because that
world is ending, not because they were wrong. This is worth telling him rather than
changing silently.

## Open decisions

1. The seven files staged locally against #1408: four duplicate work #771 already has and
   should be dropped; two (`plugin-system.md` rows, converter coverage tests) carry
   forward to AIP-1177.
2. Whether #1408 is kept rebased and green as a fallback until phase 4, or allowed to go
   stale.
3. Whether all five changes land before #771 is marked ready, or only the two security
   fixes.
