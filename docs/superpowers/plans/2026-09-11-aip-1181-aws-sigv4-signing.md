# AIP-1181: AWS SigV4 Request Signing — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish and land PR #771 so aiperf can sign requests to any SigV4-protected endpoint, and so a second transport (SageMaker, AIP-1177) can use the signer.

**Architecture:** #771 already implements the signer, the `request_signer` plugin category, and signing across eight call sites (inference, video submit/poll/download, readiness probe, control-plane hooks, timing manager). This plan makes five changes: two that unblock a second transport, one that turns the signing-scope flag into a derivable override, one security fix, and one that makes CI actually execute the AWS suite.

**Tech Stack:** Python 3.11–3.13, botocore (optional `aiperf[aws]` extra), aiohttp, Pydantic v2, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-09-11-aws-sigv4-and-sagemaker-design.md`

## Global Constraints

- Work on `ajc/sigv4-support` (same repo, not a fork). Do not create a parallel branch.
- botocore floor is `>=1.34.0`. Do not raise it; nothing in production needs newer.
- Commits are GPG-signed and require `Signed-off-by:` (repo convention — every commit on `main` has one).
- Gates for every task: `make check-ergonomics` and `make check-ruff-baselined` must report `0 new`, and `pre-commit run --all-files` must pass.
- Never hand-edit generated artifacts. Regenerate with `make generate-all-plugin-files`, `make generate-all-docs`, `make generate-config-schema`, `make generate-crd`.
- No `Any`. Use `X | Y`, never `Optional`/`Union`. Every Pydantic field needs `Field(description=...)`.

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `pyproject.toml` | dependency extras | add `aiperf[aws]` to `test` |
| `uv.lock` | resolution | regenerate |
| `src/aiperf/config/endpoint.py` | signing config + validation | transport gate, loopback exemption, rename |
| `src/aiperf/config/flags/cli_config.py` | CLI surface | rename flag |
| `src/aiperf/config/flags/_section_fields.py`, `_converter_endpoint.py` | flag threading | rename |
| `src/aiperf/common/models/model_endpoint_info.py` | runtime model | rename |
| `src/aiperf/auth/sigv4_signer.py` | signing name resolution | derive from transport |
| `src/aiperf/transports/aiohttp_transport.py` | request send path | redirect guard |
| `src/aiperf/transports/aiohttp_client.py` | HTTP client | kwargs forwarding |
| `tests/unit/config/test_endpoint_sigv4_validation.py` | config validation | new cases |
| `tests/unit/transports/test_aiohttp_transport_signing.py` | signing behaviour | new cases |

---

### Task 1: Take over the branch

**Files:**
- Modify: none (git + Linear bookkeeping)

**Interfaces:**
- Consumes: nothing
- Produces: `ajc/sigv4-support` rebased onto current `main`, green CI

- [ ] **Step 1: Fetch and inspect**

```bash
git fetch origin main ajc/sigv4-support
git log --oneline origin/ajc/sigv4-support..origin/main   # commits to absorb
git log --oneline -5 origin/ajc/sigv4-support
```

- [ ] **Step 2: Create a local tracking branch and rebase**

```bash
git checkout -b ajc/sigv4-support origin/ajc/sigv4-support
git rebase origin/main
```

Expected: clean, or conflicts only in generated artifacts. If a generated file conflicts, take `origin/main`'s version and regenerate in Step 3 — never merge it by hand.

- [ ] **Step 3: Regenerate artifacts and verify**

```bash
make install
make generate-all-plugin-files && make generate-all-docs
make generate-config-schema && make generate-crd
.venv/bin/python -m pytest tests/unit tests/aiperf_mock_server -n auto -q
```

Expected: full suite passes.

- [ ] **Step 4: Push and confirm CI**

```bash
git push --force-with-lease origin ajc/sigv4-support
gh pr checks 771 --repo ai-dynamo/aiperf
```

- [ ] **Step 5: Linear bookkeeping**

Reassign AIP-1181 to yourself. Replace the stale line
`HTTPS enforcement follow-up pending` — commit `31731ac35` already implemented it.
Set AIP-1177 `blockedBy` AIP-1181.

---

### Task 2: Make CI actually run the AWS suite

Do this **first**. Until it lands, the Windows-on-ARM job skips every AWS test and reports green, so every later task is unverified on that platform.

**Files:**
- Modify: `pyproject.toml`, `uv.lock`

**Interfaces:**
- Consumes: nothing
- Produces: botocore present wherever the `test` extra is installed

- [ ] **Step 1: Confirm the gap**

```bash
uv export --extra test --no-dev --no-emit-project | grep -c '^botocore'
```

Expected: `0` — botocore absent from the `test` extra.

- [ ] **Step 2: Add the extra**

In `pyproject.toml`, inside `test = [`, as the first entry:

```toml
test = [
  # The AWS suite self-skips via pytest.importorskip("botocore"). The
  # Windows-on-ARM job installs "--extra test --no-dev", so without botocore
  # here it skips that entire suite and still reports green. botocore is a
  # pure-Python (py3-none-any) wheel, so unlike the deps `dev` pulls it
  # installs there fine.
  "aiperf[aws]",
  "httpx>=0.27.0",
```

- [ ] **Step 3: Relock and verify**

```bash
uv lock
uv lock --check
uv export --extra test --no-dev --no-emit-project | grep -c '^botocore'
```

Expected: `uv lock --check` passes; grep reports `1`.

- [ ] **Step 4: Run the suite**

```bash
.venv/bin/python -m pytest tests/unit tests/aiperf_mock_server -n auto -q
```

Expected: passes, with more tests collected than before (the AWS suite now runs locally under the `test` extra too).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock
git commit -S -m "test(aws): run the AWS suite wherever the test extra is installed

The suite self-skips via pytest.importorskip(\"botocore\") and the
Windows-on-ARM job installs --extra test --no-dev, so it skipped every
AWS test and still reported green.

Signed-off-by: <your name> <your email>"
```

---

### Task 3: Admit signing-capable transports, not just HTTP

**Files:**
- Modify: `src/aiperf/config/endpoint.py`
- Test: `tests/unit/config/test_endpoint_sigv4_validation.py`

**Interfaces:**
- Consumes: nothing
- Produces: a config that accepts `transport` values other than `HTTP` when that transport signs. AIP-1177 depends on this.

- [ ] **Step 1: Write the failing test**

```python
def test_a_signing_capable_transport_is_accepted() -> None:
    """The gate must test capability, not identity with HTTP. A transport that
    subclasses AioHttpTransport inherits _sign_if_needed and does sign."""
    from unittest.mock import patch

    from aiperf.config.endpoint import EndpointConfig
    from aiperf.transports.aiohttp_transport import AioHttpTransport

    class _SigningTransport(AioHttpTransport):
        pass

    with patch(
        "aiperf.plugin.plugins.get_class", return_value=_SigningTransport
    ):
        cfg = EndpointConfig.model_validate(
            {
                "type": "chat",
                "urls": ["https://x.example.com"],
                "transport": "http",
                "auth_type": "sigv4",
                "aws_region": "us-west-2",
                "aws_service": "execute-api",
            }
        )

    assert cfg.auth_type is not None


def test_a_transport_that_cannot_sign_is_still_rejected() -> None:
    """The original protection stands: a transport with no signing call site
    would resolve credentials and sign nothing."""
    import pytest
    from pydantic import ValidationError

    from aiperf.config.endpoint import EndpointConfig

    class _NonSigningTransport:
        pass

    from unittest.mock import patch

    with patch("aiperf.plugin.plugins.get_class", return_value=_NonSigningTransport):
        with pytest.raises(ValidationError, match="sign"):
            EndpointConfig.model_validate(
                {
                    "type": "chat",
                    "urls": ["https://x.example.com"],
                    "transport": "http",
                    "auth_type": "sigv4",
                    "aws_region": "us-west-2",
                    "aws_service": "execute-api",
                }
            )
```

- [ ] **Step 2: Run to verify it fails**

```bash
.venv/bin/python -m pytest tests/unit/config/test_endpoint_sigv4_validation.py -k signing_capable -v
```

Expected: FAIL — the current gate rejects on identity with `TransportType.HTTP`.

- [ ] **Step 3: Replace the gate**

In `src/aiperf/config/endpoint.py`, replace:

```python
        # Transport None means auto-detect HTTP from URL — allowed. Only the
        # HTTP transport (aiohttp_transport.py) calls _sign_if_needed; any other
        # transport would resolve AWS credentials and sign nothing, silently
        # producing unauthenticated requests.
        if self.transport is not None and self.transport != TransportType.HTTP:
            raise ValueError(
                "--auth-type sigv4 requires HTTP transport; unsupported transport "
                f"{self.transport!r}"
            )
```

with:

```python
        # Transport None means auto-detect HTTP from URL — allowed. Otherwise the
        # selected transport must actually sign: one that lacks the signing call
        # site would resolve AWS credentials and sign nothing, silently producing
        # unauthenticated requests. Tested by capability rather than identity so a
        # second signing transport (SageMaker) is not excluded by construction.
        if self.transport is not None and not _transport_signs(self.transport):
            raise ValueError(
                f"--auth-type {self.auth_type} requires a transport that signs "
                f"requests; {self.transport!r} does not. Signing is implemented by "
                "the HTTP transport and anything deriving from it."
            )
```

And add, above `class EndpointConfig`:

```python
def _transport_signs(transport: TransportType) -> bool:
    """Whether the named transport applies the configured request signer.

    Signing lives on ``AioHttpTransport._sign_if_needed``; anything deriving
    from it inherits the call site. Resolved through the plugin registry rather
    than an enum comparison so adding a signing transport needs no edit here.
    """
    from aiperf.plugin import plugins
    from aiperf.plugin.enums import PluginType
    from aiperf.transports.aiohttp_transport import AioHttpTransport

    try:
        transport_cls = plugins.get_class(PluginType.TRANSPORT, str(transport))
    except Exception:
        return False
    return isinstance(transport_cls, type) and issubclass(
        transport_cls, AioHttpTransport
    )
```

- [ ] **Step 4: Run to verify it passes**

```bash
.venv/bin/python -m pytest tests/unit/config/test_endpoint_sigv4_validation.py -v
.venv/bin/python -m pytest tests/unit/config tests/unit/transports -n auto -q
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/aiperf/config/endpoint.py tests/unit/config/test_endpoint_sigv4_validation.py
git commit -S -m "fix(config): gate sigv4 on whether the transport signs, not on HTTP

Signing lives on AioHttpTransport._sign_if_needed, which subclasses
inherit. Comparing against TransportType.HTTP excluded any future signing
transport by construction.

Signed-off-by: <your name> <your email>"
```

---

### Task 4: Exempt loopback from the HTTPS requirement

**Files:**
- Modify: `src/aiperf/config/endpoint.py`
- Test: `tests/unit/config/test_endpoint_sigv4_validation.py`

**Interfaces:**
- Consumes: nothing
- Produces: signed runs against `http://localhost` are permitted. The AIP-1177 end-to-end mock-server gate depends on this.

- [ ] **Step 1: Write the failing test**

```python
import pytest
from pydantic import ValidationError
from pytest import param

from aiperf.config.endpoint import EndpointConfig


def _signed(urls: list[str]) -> EndpointConfig:
    return EndpointConfig.model_validate(
        {
            "type": "chat",
            "urls": urls,
            "auth_type": "sigv4",
            "aws_region": "us-west-2",
            "aws_service": "execute-api",
        }
    )


@pytest.mark.parametrize(
    "url",
    [
        param("http://localhost:8765", id="localhost"),
        param("http://127.0.0.1:8765", id="ipv4-loopback"),
        param("http://[::1]:8765", id="ipv6-loopback"),
    ],
)
def test_loopback_over_http_is_allowed(url: str) -> None:
    """Credentials that never leave the machine are not disclosed by the
    absence of TLS, and the documented mock-server workflow signs against
    http://localhost."""
    assert _signed([url]).urls == [url]


def test_remote_cleartext_is_still_rejected() -> None:
    with pytest.raises(ValidationError, match="https"):
        _signed(["http://sagemaker.example.com"])


def test_one_remote_cleartext_url_among_several_is_rejected() -> None:
    with pytest.raises(ValidationError, match="https"):
        _signed(["https://good.example.com", "http://bad.example.com"])
```

- [ ] **Step 2: Run to verify it fails**

```bash
.venv/bin/python -m pytest tests/unit/config/test_endpoint_sigv4_validation.py -k loopback -v
```

Expected: FAIL — the current rule requires `https` for every URL.

- [ ] **Step 3: Add the exemption**

Replace the scheme loop in `src/aiperf/config/endpoint.py`:

```python
        for url in self.urls:
            scheme = urlparse(url).scheme.lower()
            if scheme != "https":
                raise ValueError(
                    f"--auth-type sigv4 requires https:// URLs; URL {url!r} uses "
                    f"scheme {scheme!r}."
                )
```

with:

```python
        insecure = [url for url in self.urls if _is_cleartext_remote(url)]
        if insecure:
            raise ValueError(
                f"--auth-type {self.auth_type} puts the signature and any session "
                "token into the request headers; sending those over plain HTTP "
                f"would disclose them: {', '.join(insecure)}. Use https, or drop "
                "--auth-type."
            )
```

And add, beside `_transport_signs`:

```python
def _is_cleartext_remote(url: str) -> bool:
    """Whether ``url`` would send request headers unencrypted off-box.

    Loopback is exempt: the documented mock-server workflow signs against
    ``http://localhost``, and headers that never leave the machine are not
    disclosed by the absence of TLS.
    """
    parsed = urlparse(url if "://" in url else f"http://{url}")
    if parsed.scheme != "http":
        return False
    host = (parsed.hostname or "").lower()
    return not (host in {"localhost", "::1"} or host.startswith("127."))
```

- [ ] **Step 4: Run to verify it passes**

```bash
.venv/bin/python -m pytest tests/unit/config/test_endpoint_sigv4_validation.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/aiperf/config/endpoint.py tests/unit/config/test_endpoint_sigv4_validation.py
git commit -S -m "fix(config): allow signed requests to loopback over http

Requiring https for every URL made the AWS path untestable locally and
broke the documented mock-server workflow. Remote cleartext stays
rejected; headers that never leave the machine are not disclosed.

Signed-off-by: <your name> <your email>"
```

---

### Task 5: Make the signing scope derivable — *amended 2026-09-21*

> **Amended after execution.** This task originally renamed `--aws-service` to
> `--aws-signing-service`. **The rename was reconsidered and dropped; the flag keeps its
> name.** Reasoning is recorded in the design doc under "Five changes to finish it",
> change 3 — chiefly that `--aws-region` names the same credential scope without a
> `signing` qualifier, so qualifying only `--aws-service` is less coherent, and that the
> adapter-selector risk the rename guarded against no longer exists.
>
> This task's original Step 4 and Step 5 also **seeded review finding F5**: they
> specified an unannotated `def _transport_botocore_service_id(transport)` and told you
> to paste a *second copy* into `config/endpoint.py`, justified by "config validation
> must not import the signer, which requires botocore". That justification is false —
> every botocore import in `sigv4_signer.py` is lazy, and the module imports fine with
> botocore blocked. The real constraint is an import cycle: `aiperf/auth/__init__.py`
> imports `base_signer`, which reaches `aiperf.config` via `aiperf.common.models`.
>
> Steps below reflect what actually shipped.

**Files:**
- Add: `src/aiperf/auth/_transport_scope.py`
- Modify: `src/aiperf/auth/sigv4_signer.py`, `src/aiperf/config/endpoint.py`
- Test: `tests/unit/auth/test_sigv4_signer.py`,
  `tests/unit/config/test_endpoint_sigv4_validation.py`

**Interfaces:**
- Consumes: nothing
- Produces: transports may declare `botocore_service_id: ClassVar[str]`; the signer
  resolves the signing name from botocore's service model. `--aws-service` stays an
  optional override. AIP-1177's transport declares `"sagemaker-runtime"`.

- [x] **Step 1: One implementation, in a leaf module**

`src/aiperf/auth/_transport_scope.py` imports nothing at module scope (the plugin
lookup is lazy inside the function), so it is safe to import from `aiperf.config` and
stays importable without the optional `aiperf[aws]` extra:

```python
def transport_botocore_service_id(transport: TransportType | str | None) -> str | None:
    if transport is None:
        return None
    from aiperf.plugin import plugins
    from aiperf.plugin.enums import PluginType

    try:
        transport_cls = plugins.get_class(PluginType.TRANSPORT, str(transport))
    except Exception:
        return None
    return getattr(transport_cls, "botocore_service_id", None)
```

Annotate the parameter. AGENTS.md requires type hints on all functions, and the missing
annotation was half of F5.

- [x] **Step 2: Signer resolves the name from botocore's model**

In `__init__`, alongside the existing region/profile reads:

```python
        self.botocore_service_id: str | None = transport_botocore_service_id(
            model_endpoint.transport
        )
```

In `_reresolve_credentials`, immediately before `session.get_credentials()`:

```python
        if not self.service and self.botocore_service_id:
            # Resolve the signing name from AWS's own service model rather than
            # mapping it here: sagemaker-runtime signs as 'sagemaker' and
            # bedrock-runtime as 'bedrock', and only the model knows that. Falls
            # back to the service id, which is right where the two coincide.
            try:
                self.service = session.get_service_model(
                    self.botocore_service_id
                ).signing_name
            except Exception:
                self.service = self.botocore_service_id
```

- [x] **Step 3: Config requires the flag only when nothing supplies the scope**

In `_validate_sigv4_auth`, import the **same** helper — inside the function, because a
module-scope import is the cycle described above — and never a second copy:

```python
        from aiperf.auth._transport_scope import transport_botocore_service_id

        aws_flags["--aws-service"] = self.aws_service or transport_botocore_service_id(
            self.transport
        )
```

- [x] **Step 4: Pin that the two can never drift**

Two guards, each verified to fail when a duplicate is reintroduced:

- config side, behavioural: patch `_transport_scope.transport_botocore_service_id` and
  assert validation stops demanding `--aws-service`. A private copy makes the patch
  unobserved and the config raises.
- signer side, structural: assert `src/` defines the lookup exactly once, matching
  **indented** definitions too — nesting one inside a function is exactly how this comes
  back, and a `startswith("def ")` check misses it.

- [x] **Step 5: Regenerate and run**

```bash
make generate-all-docs && make generate-config-schema && make generate-crd
.venv/bin/python -m pytest tests/unit -n auto -q
```

---

### Task 6: Do not follow redirects on signed requests

**Files:**
- Modify: `src/aiperf/transports/aiohttp_transport.py`, `src/aiperf/transports/aiohttp_client.py`
- Test: `tests/unit/transports/test_aiohttp_transport_signing.py`

**Interfaces:**
- Consumes: nothing
- Produces: signed requests sent with `allow_redirects=False` on both the normal and cancellation paths

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_redirects_are_disabled_for_signed_requests() -> None:
    """aiohttp strips Authorization across origins but not custom headers, so
    X-Amz-Security-Token -- a bearer credential -- would be replayed to the
    redirect target. Nothing is lost: SigV4 signs Host, so a replayed
    signature is invalid there anyway."""
    transport = await _signed_transport()
    await _send(transport)

    assert transport.aiohttp_client.post_request.call_args.kwargs[
        "allow_redirects"
    ] is False


@pytest.mark.asyncio
async def test_redirects_are_disabled_on_the_cancellation_path_too() -> None:
    """post_request routes cancellable requests through a different helper, so
    the guard has to survive that branch as well."""
    transport = await _signed_transport()
    request_info = _request_info(transport.model_endpoint)
    request_info.cancel_after_ns = 10_000_000_000

    await transport.send_request(request_info, {"messages": []})

    assert transport.aiohttp_client.post_request.call_args.kwargs[
        "allow_redirects"
    ] is False
```

- [ ] **Step 2: Run to verify it fails**

```bash
.venv/bin/python -m pytest tests/unit/transports/test_aiohttp_transport_signing.py -k redirect -v
```

Expected: FAIL — `allow_redirects` is not passed.

- [ ] **Step 3: Pass the flag when a signer is active**

In `aiohttp_transport.py`, immediately before the `post_request` call in
`send_request`:

```python
            # A redirect would replay the signed headers at the new origin.
            # aiohttp drops Authorization when the origin changes but has no such
            # rule for custom headers, so X-Amz-Security-Token would follow.
            redirect_kwargs: dict[str, Any] = (
                {"allow_redirects": False} if self.request_signer else {}
            )
```

and add `**redirect_kwargs,` to that call.

- [ ] **Step 4: Fix the dropped kwargs**

In `aiohttp_client.py`, `post_request` currently calls
`_request_with_cancellation(...)` without `**kwargs`. Add `**kwargs,` to that call,
add `**kwargs: Any,` to `_request_with_cancellation`'s signature, and add `**kwargs,`
to the `_request(...)` call inside it. Without all three the guard silently does not
reach cancellable requests.

- [ ] **Step 5: Explain the refusal**

After `record.request_headers = redact_headers(headers)` in `send_request`:

```python
            # Signed requests are sent with allow_redirects=False, so a 3xx
            # reaches the caller rather than being followed. "302 Found" alone
            # does not say why, and the reason is deliberate.
            if (
                self.request_signer is not None
                and record.error is not None
                and record.status is not None
                and 300 <= record.status < 400
            ):
                record.error.message = (
                    f"{record.error.message} "
                    "(aiperf does not follow redirects on signed requests: the "
                    "SigV4 signature covers the Host header, so it would be "
                    "invalid at the redirect target, and credential headers such "
                    "as x-amz-security-token would be replayed there.)"
                ).strip()
```

- [ ] **Step 6: Run to verify it passes**

```bash
.venv/bin/python -m pytest tests/unit/transports -n auto -q
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/aiperf/transports/ tests/unit/transports/
git commit -S -m "fix(auth): stop signed requests following redirects

X-Amz-Security-Token is a bearer credential and aiohttp does not strip
custom headers across origins. Also fixes post_request dropping **kwargs
on its cancellation branch, which would have made the guard skip
cancellable requests.

Signed-off-by: <your name> <your email>"
```

---

### Task 7: Verify and mark ready

**Files:** none

**Interfaces:**
- Consumes: Tasks 1–6
- Produces: #771 out of draft, ready for review

- [ ] **Step 1: Full verification**

```bash
.venv/bin/python -m pytest tests/unit tests/aiperf_mock_server -n auto -q
make check-ergonomics && make check-ruff-baselined
.venv/bin/pre-commit run --all-files
```

Expected: suite green; both gates report `0 new`; all hooks pass.

- [ ] **Step 2: Prove the AWS suite executes rather than skips**

```bash
.venv/bin/python -m pytest tests/unit/auth tests/unit/transports -q -rs | grep -c "importorskip"
```

Expected: `0` — no AWS module skipped when the `test` extra is installed.

- [ ] **Step 3: Rebase, push, confirm CI**

```bash
git fetch origin main && git rebase origin/main
git push --force-with-lease origin ajc/sigv4-support
gh pr checks 771 --repo ai-dynamo/aiperf
```

Expected: all checks pass. Inspect a Windows-ARM job log and confirm AWS tests ran.

- [ ] **Step 4: Mark ready and note the overrides**

```bash
gh pr ready 771 --repo ai-dynamo/aiperf
```

Comment on the PR recording that Tasks 3 and 4 deliberately change reasoning Anthony
documented — signing restricted to HTTP transport, and HTTPS required everywhere — and
why: a second signing transport is arriving, and the mock-server workflow needs
loopback. Tag him.

---

## Self-Review

**Spec coverage:** all five changes map to Tasks 2–6; takeover to Task 1; the gate to
Task 7. Sequencing steps 1–3 of the spec are covered; steps 4–6 belong to the AIP-1177
plan.

**Placeholders:** none. `<your name> <your email>` in commit templates is a deliberate
substitution, not a TODO.

**Type consistency:** ~~`_transport_botocore_service_id` is defined twice on purpose —
in `sigv4_signer.py` and in `config/endpoint.py` — because config validation must not
import the signer module, which requires botocore.~~ **Wrong on both counts, and this
became review finding F5.** The premise is false: every botocore import in
`sigv4_signer.py` is lazy, so the module imports fine without the `aiperf[aws]` extra.
The real constraint is an import *cycle* (`aiperf.auth.__init__` → `base_signer` →
`aiperf.common.models` → `aiperf.config`), which a lazy import at the one config call
site solves without a second copy. Two copies could drift, and a drifted pair lets
validation accept a scope the signer resolves to `None` — a mid-run 403 instead of a
config error. Now defined once in `auth/_transport_scope.py`. `_transport_signs` and
`_is_cleartext_remote` are defined once each, in `config/endpoint.py`.

**Known risk:** Task 5 no longer renames anything, so the six-hop rename risk below is
moot as written — but it still applies to AIP-1177, which must revert #1408's
`aws_signing_service` back to `aws_service` across those same hops. The hop that fails
silently is
`EndpointInfo` and its `from_run` — a missing field there reads back fine in a
hand-built object and is absent everywhere that matters. `test_every_cli_config_field_is_classified` catches the CLI hop but not that one.
