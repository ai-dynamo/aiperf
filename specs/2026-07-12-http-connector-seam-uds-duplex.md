# HTTP Transport `Connector` Seam: TCP / UDS / Duplex Stream Factories

**Status:** Designed, compile-verified (scratch project), not yet implemented.
**Date:** 2026-07-12
**Crate:** the seam lives in the `aiperf_runtime::transport_http` module (formerly the standalone `aiperf-transport-http` crate, now inlined into `aiperf-runtime`).
**Verification:** `~/tmp/connector-spec/` — trait + all three impls build on the workspace dep majors (hyper 1.10, hyper-util 0.1, tokio 1, async-trait 0.1) AND round-trip end-to-end over `TcpStream`, `UnixStream`, and `tokio::io::DuplexStream` behind one object-safe `#[async_trait(?Send)]` trait held as `Rc<dyn Connector>`.

---

## 1. Motivation

### The mode-branch smell (real code)

Connection establishment lives in
`rust/runtime/src/transport_http/client/connection.rs`. The un-timed body is
`establish_inner` (`connection.rs:419-509`). Its first act is a mode branch on a
config flag:

```rust
// connection.rs:426-444
#[cfg(unix)]
if let Some(path) = &cfg.uds_path {
    trace.tcp_connect_start_ns = Some(clock.now_ns());
    let stream = tokio::net::UnixStream::connect(path)
        .await
        .map_err(ErrorDetails::from)?;
    trace.tcp_connect_end_ns = Some(clock.now_ns());
    let sender = handshake(TokioIo::new(stream), false).await?;
    let dummy = SocketAddr::from(([127, 0, 0, 1], 0));
    return Ok((sender, SocketInfo { local: dummy, remote: dummy }));
}
// ...falls through to the TCP + DNS + TLS + handshake path (connection.rs:446-508)
```

This is the project's documented "branching on a mode = trait waiting to be
extracted" smell (CLAUDE.md, *Extensibility & porting discipline*). The stream
acquisition strategy (UDS vs TCP+TLS) is hardcoded as an `if let`, and the
selector (`ClientConfig::uds_path`, `config/defaults.rs:202`) is a per-config
flag threaded from the top rather than a first-class implementation choice.

`uds_path` is **not product-reachable** — no Python/CLI path and no
`EndpointProfileConfigV2` field assigns it, so on the product path it is always
`None`. **Correction (previously mis-stated as "never assigned anywhere"):** the
graph transport benchmark **does** assign and depend on it. `transport_bench.rs`
parses a `unix:` base-URL prefix into `Some(p.to_string())`
(`transport_bench.rs:462–474`) and constructs `ClientConfig { http_version,
uds_path, .. }` with that non-`None` value (`:502–506`), which drives the
`establish_inner` UDS branch (`connection.rs:426–444`) for co-located UDS
benchmarking. A blanket `grep -rn "uds_path:"` misses this because the field is
set via struct-shorthand (`uds_path,` not `uds_path:`).

So UDS is **not** dead code — it is live in the benchmark harness but unwired to
the product. That changes the deletion plan: dropping the `ClientConfig::uds_path`
field or the `establish_inner` UDS branch **without migrating `transport_bench`
first breaks the graph bench** (see §4 and §6). The seam extraction is still the
right move, but `transport_bench` must be migrated to the new `unix:`-scheme
`Connector` selection in the same change.

### Scheme-driven selection

Today an endpoint is a `url::Url`; whether it uses UDS is decided by a *separate*
`ClientConfig` field, not by the URL. The natural, self-describing selector is
the URL scheme: `http(s)://host:port` → TCP, `unix:/path` → UDS, an in-memory
scheme (e.g. `mem://name`) → Duplex. Selection then happens once at
endpoint-prepare time and produces the right `Connector`, instead of a boolean
riding in the client config forever.

### Three-modes-for-free bonus

CLAUDE.md's north star is a single `{transport, clock}` seam that yields
online-real / online-mock / offline-mock "for free." A `DuplexConnector` — an
in-process `tokio::io::duplex` pipe to a handler, zero sockets — is the missing
seam that lets an **in-process online mock** run over the *exact* HTTP client
stack (handshake, SSE parse, first-token filter, reuse, tracing, observer) with
no loopback socket. It is also the first concrete step toward unifying
online/offline-mock, though full virtual-clock offline unification is a separate,
harder step (see §7).

---

## 2. The `Connector` trait (compile-verified final form)

The connector's job: **"given an endpoint URL + client config + clock, produce a
ready HTTP `Sender` and a `SocketInfo`, stamping connect timings into
`TraceData`."** Everything above the `Sender` (request build, send, SSE parse,
reuse accounting, tracing, observer) is unchanged.

```rust
use std::rc::Rc;
use url::Url;
use crate::clock::Clock;
use crate::client::connection::{Sender, SocketInfo};
use crate::config::ClientConfig;
use crate::models::{ErrorDetails, TraceData};

/// How to obtain a ready HTTP `Sender` + `SocketInfo` from an endpoint address.
///
/// `?Send`: the crate is thread-per-core `Rc`/`RefCell`; the returned `Sender`
/// and the connection future spawned via `spawn_local` are `!Send`.
/// Object-safe: concrete/`dyn` params, single async method boxed by `async_trait`.
///
/// Extension point: a new byte-stream source (QUIC, vsock, a second in-memory
/// dialect) implements this trait; nothing above the `Sender` changes.
#[async_trait::async_trait(?Send)]
pub trait Connector {
    async fn connect(
        &self,
        url: &Url,
        cfg: &ClientConfig,
        clock: Rc<dyn Clock>,
        trace: &mut TraceData,
    ) -> Result<(Sender, SocketInfo), ErrorDetails>;
}
```

**Why this exact shape (each choice is compiler-forced or code-grounded):**

- **Return `(Sender, SocketInfo)`, not a raw IO stream.** The h2-vs-h1 decision
  is intertwined with the stream type: UDS is h1-only (`connection.rs:435`), TCP
  derives it from TLS ALPN + force flags (`connection.rs:468-500`), Duplex is
  configurable. So the connector owns the `handshake(io, use_h2)` call
  (`connection.rs:511-540`, unchanged) rather than returning an
  `enum{Tcp,Uds,Duplex}` IO for a caller to match on — which would just relocate
  the mode branch. `Sender` is the existing enum (`connection.rs:203-206`):
  `H1(http1::SendRequest<TimedBody>)` / `H2(http2::SendRequest<TimedBody>)`.
  `SocketInfo` is the existing `{ local: SocketAddr, remote: SocketAddr }`
  (`connection.rs:265-269`).
- **`#[async_trait(?Send)]` is REQUIRED, not optional.** Verified: a native
  `async fn` in a trait is not dyn-compatible, so `Rc<dyn Connector>` fails to
  compile (`error[E0038]: the trait Connector is not dyn compatible ... consider
  moving connect to another trait`). `async_trait` boxes the returned future
  (`Pin<Box<dyn Future + '_>>`) to restore object-safety. `(?Send)` is required
  because the future captures `!Send` state and the produced `Sender`/connection
  future are `!Send`. This matches the crate's existing `DnsResolver`,
  `HostLookup`, and `ConnectionManager` traits, all `#[async_trait(?Send)]`
  (`resolver.rs:24,34,51,105`; `pool.rs:129,388`).
- **`Rc<dyn Connector>`, not `Arc`.** The crate is `Rc`/`RefCell` throughout; the
  pool already holds `resolver: Rc<dyn DnsResolver>` (`pool.rs:116`) and clocks
  flow as `Rc<dyn Clock>`. Each thread-per-core worker builds its own transport
  stack, so no cross-thread sharing is needed. `Arc<dyn Connector>` would force
  `Send + Sync` bounds the impls (holding `Rc<dyn DnsResolver>`) cannot satisfy.
  **Verified constraint:** `Rc<dyn Connector>` compiles and is the correct
  handle; `Arc` is wrong here.
- **Manual `-> Pin<Box<dyn Future + '_>>` is an equivalent alternative** to
  `async_trait` and avoids the macro, but is strictly more verbose and buys
  nothing over the established in-crate `async_trait(?Send)` convention. Chosen:
  `async_trait(?Send)`.

---

## 3. The three impls (compile-verified sketches)

All three build and round-trip in `~/tmp/connector-spec/`. Field-for-field they
fold in today's `establish_inner` logic, split by stream source.

### 3.1 `TcpConnector` — folds in `DnsResolver` + TLS + socket opts

Owns the resolver (previously threaded into `establish_with_resolver` by the
pool, `pool.rs:309-315`). This is where the `DnsResolver` seam moves.

```rust
pub struct TcpConnector {
    resolver: Rc<dyn DnsResolver>,   // was pool.inner.resolver
}

#[async_trait(?Send)]
impl Connector for TcpConnector {
    async fn connect(&self, url, cfg, clock, trace) -> Result<(Sender, SocketInfo), ErrorDetails> {
        // Verbatim relocation of connection.rs:446-508:
        //   host/port/is_tls from url
        //   let remote = self.resolver.resolve(host, port, cfg, &clock, trace).await?;
        //   trace.tcp_connect_start_ns = Some(clock.now_ns());
        //   let tcp = TcpStream::connect(remote).await?;
        //   apply_socket_opts(&SockRef::from(&tcp));
        //   if is_tls { rustls_config + TlsConnector + ALPN -> use_h2; handshake(TokioIo::new(tls), use_h2) }
        //   else      { trace.tcp_connect_end_ns; use_h2 = force_h2; handshake(TokioIo::new(tcp), use_h2) }
        //   fill trace.local_ip/port + remote_ip/port
        // Ok((sender, SocketInfo { local, remote }))
    }
}
```

One-line essence: **today's TCP path, verbatim, with `resolver` + `rustls_config`
+ `apply_socket_opts` as owned collaborators.**

### 3.2 `UdsConnector` — the extracted `if let Some(uds_path)` branch

```rust
pub struct UdsConnector { path: std::path::PathBuf }

#[async_trait(?Send)]
impl Connector for UdsConnector {
    async fn connect(&self, _url, _cfg, clock, trace) -> Result<(Sender, SocketInfo), ErrorDetails> {
        trace.tcp_connect_start_ns = Some(clock.now_ns());
        let stream = tokio::net::UnixStream::connect(&self.path).await.map_err(ErrorDetails::from)?;
        trace.tcp_connect_end_ns = Some(clock.now_ns());
        let sender = handshake(TokioIo::new(stream), /*use_h2=*/ false).await?;
        let dummy = SocketAddr::from(([127, 0, 0, 1], 0));
        Ok((sender, SocketInfo { local: dummy, remote: dummy }))
    }
}
```

One-line essence: **`connection.rs:428-444` moved intact; h1-only; synthesized
loopback `SocketInfo`.**

> **CORRECTION (Finding 4) — a `unix:` URL CANNOT also supply the HTTP request
> line + Host; those must come from a SEPARATE route URL that is NOT deleted with
> `uds_path`.** `build_request_with_method` derives the HTTP request line from
> `url.path()` (+query) and the `HOST` header from `url.authority()` — **both from
> the same `url`** (`http_client.rs:408-416`). Today UDS deliberately keeps two
> URLs apart: `uds_path` holds the *socket* path while a **distinct** normal URL
> `http://localhost/v1/chat/completions` supplies the HTTP *route* + Host
> (`transport_bench.rs:462-466`). If scheme-driven selection makes the endpoint
> URL `unix:/run/x.sock`, then `url.path() = "/run/x.sock"` becomes the (wrong)
> request line and `url.authority() = ""` becomes an **empty Host** — the real API
> route (`/v1/chat/completions`) and a valid Host are lost, so the server 404s/400s.
>
> **Design fix:** the socket path and the HTTP request URL are two different
> things and must both survive. Concretely, one of:
> 1. **`UdsConnector { path }` for connect, plus a retained HTTP request URL**
>    (`http://<host>/<route>`) that `HttpClient`/`build_request_with_method` keeps
>    using for the request line + Host. `unix:` selects the connector and provides
>    the socket path; a companion route URL (the endpoint's normal
>    `http://host/route`, or a synthesized `http://localhost/<route>`) drives
>    request building — exactly today's `transport_bench` split, just chosen by
>    scheme instead of the `uds_path` flag. This means the "delete `uds_path`"
>    step (§4/§5) removes only the *connect selector* flag; a route/Host source is
>    still required and must not be dropped.
> 2. A composite `unix:` URL that encodes both socket path and route (e.g.
>    `unix:/run/x.sock` + an explicit route/Host carried alongside), with
>    `build_request_with_method` taught to pull the request line + Host from the
>    route component, never from `url.path()`/`url.authority()` of the raw `unix:`
>    URL. This requires changing `build_request_with_method`, so option 1 (retain a
>    separate route URL) is the lower-risk choice.
>
> Either way, the assertion "the URL still supplies the HTTP path + Host,
> unchanged" is FALSE for a `unix:` URL and is replaced by the above.

### 3.3 `DuplexConnector` — NEW, in-memory, zero sockets

Reaches an in-process server/handler through a small `DuplexEndpoint` handle the
connector holds. `open()` returns the client half of a fresh
`tokio::io::duplex(cap)` pair and is responsible for driving the server half
(e.g. `spawn_local`).

```rust
pub trait DuplexEndpoint {
    /// Client end of a new in-memory connection; impl drives the server end.
    fn open(&self) -> tokio::io::DuplexStream;
}

pub struct DuplexConnector { endpoint: Rc<dyn DuplexEndpoint>, use_h2: bool }

#[async_trait(?Send)]
impl Connector for DuplexConnector {
    async fn connect(&self, _url, _cfg, clock, trace) -> Result<(Sender, SocketInfo), ErrorDetails> {
        trace.tcp_connect_start_ns = Some(clock.now_ns());
        let client_io = self.endpoint.open();               // spawns in-proc server internally
        trace.tcp_connect_end_ns = Some(clock.now_ns());
        let sender = handshake(TokioIo::new(client_io), self.use_h2).await?;
        let dummy = SocketAddr::from(([0, 0, 0, 0], 0));
        Ok((sender, SocketInfo { local: dummy, remote: dummy }))
    }
}
```

Concrete stream → hyper handshake: `tokio::io::DuplexStream` implements
`AsyncRead + AsyncWrite`; wrapping in `hyper_util::rt::TokioIo` yields
`hyper::rt::Read + Write + Unpin + 'static`, which the shared `handshake<I>` for
both http1/http2 accepts. **Verified** by the e2e test: a hyper `http1` server
`serve_connection` on the server half answers the client-half request `pong`.

One-line essence: **`tokio::io::duplex` pipe to a `spawn_local`'d in-process
handler; the exact HTTP client stack runs with no loopback socket.**

---

## 4. Integration diff-plan (file:line touch points)

### `client/connection.rs`
- **Keep unchanged:** `handshake<I>` (`:511-540`), `Sender` (`:203-262`),
  `SocketInfo` (`:265-269`), `LocalExec` (`:37-48`), `TimedBody`/`SendCompletion`
  (`:56-200`), `with_timeout` (`:358-376`), `rustls_config`/
  `NoCertificateVerification` (`:271-350`).
- **Delete the mode branch:** remove `establish_inner`'s
  `#[cfg(unix)] if let Some(path) = &cfg.uds_path { … }` block (`:426-444`).
- **Refactor the free functions into `TcpConnector`:** the DNS→TCP→TLS→handshake
  body (`:446-508`) becomes `TcpConnector::connect`. `establish` (`:382-390`) and
  `establish_with_resolver` (`:397-416`) either (a) become thin shims that build
  a `TcpConnector` and call `connect` under `with_timeout`, or (b) are removed in
  favor of the connector + a `connect_with_timeout` helper. Recommended: keep a
  `connect_timeout` wrapper that races any `&dyn Connector` against the
  `cfg.connect_timeout_ns` Clock timer (the `with_timeout` logic at `:404-415`
  is connector-agnostic and stays).
- **New:** `Connector` trait (new `client/connector.rs` module, or in
  `connection.rs`), plus `TcpConnector`, `UdsConnector`, `DuplexConnector`,
  `DuplexEndpoint`.
- **Trace preservation:** each impl sets `tcp_connect_start_ns`/`_end_ns` and
  (TCP) `local_ip/port`, `remote_ip/port`, `tls_connect_*` exactly as the current
  code does (`:430-434,456-506`). This must remain byte-identical — the pool's
  `copy_socket` (`pool.rs:43-48`) and the sweep-line consumers read these.

### `client/pool.rs`
- **`PoolInner`:** replace `resolver: Rc<dyn DnsResolver>` (`:116`) with
  `connector: Rc<dyn Connector>`. `ConnectionPool::with_resolver`
  (`:159-168`) becomes `with_connector`; `ConnectionPool::new` (`:154-156`)
  defaults to `TcpConnector { resolver: CachingDnsResolver::default() }`.
- **Establish call sites:** the two `establish_with_resolver(url, cfg, clock,
  trace, self.inner.resolver.as_ref())` calls (`:309-315` in `acquire_managed`
  and `:400-407` in the `Never` branch of `acquire`) become
  `self.inner.connector.connect(url, cfg, clock, trace)`.
- **`origin_key` (pool key) — IMPORTANT:** `origin_key` (`:28-35`) keys on
  `scheme://host:port`. For UDS/Duplex the host is synthetic (URL is
  `unix:/path` or `mem://name`), so pooling MUST key on the path/name, not host.
  Fix: derive the pool key from the URL such that `unix:/run/x.sock` and
  `mem://engine-a` produce distinct, stable keys. Simplest: `origin_key` returns
  the full `scheme + scheme-specific-part` for non-`http(s)` schemes (e.g.
  `url.scheme()` + `url.path()` for `unix`, `url.scheme()` + `url.host_str()` for
  `mem`). The sticky-session binding check (`:191-203`) already compares the same
  key string, so it inherits the fix. Two different UDS paths must not collide;
  two requests to the same UDS path must share one pool entry.

### `client/http_client.rs`
- `HttpClient` (`:202-210`) holds `clock` + `cfg`. Its two `establish(...)` calls
  (`:276`, `:330`) are on the direct (non-pooled) `request`/`request_cancellable`
  paths. Give `HttpClient` a `connector: Rc<dyn Connector>` field and replace
  `establish(url, &self.cfg, self.clock.clone(), &mut trace)` with
  `self.connector.connect(&url, &self.cfg, self.clock.clone(), &mut trace)`
  (wrapped in the same `connect_timeout` helper to preserve
  `connect_timeout_ns`). The `dispatch*` methods (`:438-792`) take an established
  `&mut Sender` and are **entirely unchanged**.
- The transport facade `HttpTransport` (`transport/http_transport.rs:254`) calls
  `self.connections.acquire(...)` on the pool; since the pool now owns the
  connector, this call is unchanged. Only the pool's construction site (wherever
  `ConnectionPool::new`/`with_resolver` is built) picks the connector.

### `config/defaults.rs`
- **Drop `uds_path`:** remove the field (`:199-202`) and its `Default` (`:220`).
  UDS is now expressed by a `unix:` URL selecting `UdsConnector` at prepare time,
  not by a client-config flag. **But (Finding 4) the socket path is only the
  connect target — the HTTP route + Host still come from a separate normal-`http`
  route URL that must be retained** (the `HttpClient`/prepared-endpoint keeps a
  route URL distinct from the `unix:` connect URL, mirroring today's
  `transport_bench` split of `uds_path` vs `http://localhost/...`). Dropping
  `uds_path` removes the *connect-selector flag*, not the route/Host source.
- **BLOCKING PREREQUISITE — migrate `rust/runtime/src/graph/transport_bench.rs` in the
  same change:** it is the only live non-`None` writer of `uds_path`
  (`transport_bench.rs:462–474` parse, `:502–506` `ClientConfig { … uds_path … }`,
  `:516` `establish(&url, &cfg, …)`). Removing the field or the UDS branch of
  `establish_inner` without touching `transport_bench` **fails to compile
  the `aiperf-runtime` crate** (the `graph` module, formerly the `aiperf-graph` crate, no
  longer sees the removed field) or, worse, **silently reverts its UDS mode to a
  TCP connect against the dummy `http://localhost` URL** (branch removed but field
  kept) — breaking the co-located UDS benchmark that Harness C (§ regression plan)
  depends on. Migration: `transport_bench` must build the same `unix:`-scheme URL
  and route through `select_connector` / a `UdsConnector`, exactly like the
  product path, instead of the `uds_path` flag. Because `transport_bench` calls
  `establish(&url, &cfg, …)` directly (not through the pool), the `establish`
  shim must accept the selected connector (or expose a `connect_via(connector,
  …)` helper). Add `transport_bench.rs` to the compile-and-run verification set
  for this PR.
- `ClientConfig` otherwise unchanged (`ssl_verify`, `prepared_tls`,
  `http_version`, timeouts, pool bounds, DNS cache flags all still read by the
  connectors / dispatch). `apply_socket_opts` (`:229-240`) is unchanged and is
  called by `TcpConnector` only.

### Scheme → connector selection (where it lives)
At **endpoint-prepare time** (worker-local `PreparedEndpoint` construction in the
transport/endpoint-binding layer — the composition root that today builds the
`ConnectionPool` and `HttpClient`), parse the endpoint URL scheme once and choose:

| URL scheme        | Connector                                   | HTTP request line + Host source |
|-------------------|---------------------------------------------|---------------------------------|
| `http` / `https`  | `TcpConnector { resolver }`                 | the same URL (`url.path()` + `url.authority()`) |
| `unix`            | `UdsConnector { path = url.path() }`        | **a SEPARATE retained route URL** (Finding 4) — `url.path()` is the SOCKET path, not the HTTP route; do not feed the `unix:` URL to `build_request_with_method` |
| `mem` (in-proc)   | `DuplexConnector { endpoint, use_h2 }`      | a separate route URL (the `mem:` name is not an HTTP route either) |

**Finding 4 note on the table:** for `unix` (and `mem`), the connector's URL is
the *connect target*, but request building still needs an HTTP route + Host from a
**different** URL (see §3.2 correction). `select_connector` must therefore return
both the connector AND the route URL to use for `build_request_with_method` (or
the prepared endpoint must retain that route URL separately). Selecting a
`UdsConnector` while continuing to feed the `unix:` URL into request building
POSTs to the socket path with an empty Host — a broken request.

Selection is a small `fn select_connector(url, cfg, ...) -> Rc<dyn Connector>`
living beside the transport composition (analogous to the existing endpoint
factory/prepared-binding wiring), NOT inside the hot dispatch path. The chosen
`Rc<dyn Connector>` is injected into `ConnectionPool` and `HttpClient` once.

### Product reachability — Python-side gates (understated in earlier drafts)

The Rust seam alone does **not** make `unix:` reachable from `aiperf profile`.
The Python endpoint validator (`src/aiperf/config/endpoint.py`, ~460–475)
rejects a `unix:/run/x.sock` URL on **three** independent gates, not one:

1. `not parsed.netloc or not parsed.hostname` → `unix:/run/x.sock` has no
   netloc/hostname → "missing scheme or host" (endpoint.py:462–466);
2. `parsed.scheme.lower() not in ("http","https","grpc","grpcs")` → rejected
   as "unsupported scheme" (endpoint.py:467–471);
3. the port-parsing block that follows assumes a host:port authority.

So enabling `unix:` is **not** a one-line "add to the scheme whitelist"
(endpoint.py:467) change as PR3 currently scopes it. The Python plumbing work is:
relax the netloc/hostname requirement for `unix:` (path-only authority), add
`unix` to the scheme whitelist, skip port parsing for `unix:`, and thread the
`unix:` URL through `rust_wire`/`EndpointProfileConfigV2` unchanged. Budget this
as real Python work in PR3, not a whitelist tweak.

**`mem://` (Duplex) has NO product selection path.** `DuplexConnector` requires
an in-process `Rc<dyn DuplexEndpoint>` handle that only a test/bench composition
root can supply; there is no Config-v2 surface that produces one. `mem://` is
therefore **test-and-bench-only** by design in this spec — the scheme→connector
table's `mem` row is reachable from `select_connector` in tests, not from
`aiperf profile`. State this explicitly so the Duplex seam is not mistaken for a
product-reachable mode.

**UDS-win throughput harness needs an unbudgeted `rps_bench` rewire.** The
regression plan's UDS-vs-TCP win driver (Harness C) uses
`rust/runtime/examples/rps_bench.rs`, which calls `establish` directly
and is **not** wired to a `unix:`-scheme client (regression plan §1.4). Measuring
the headline UDS win requires rewiring `rps_bench` to select a `UdsConnector`
(the `fast_sse` example that earlier drafts cited for its `UDS_PATH` server has
since been REMOVED with the dissolution of `aiperf-core`, so a `unix:`-client rig
must be built rather than reused), which is additional
work PR3 must schedule; without it the connector seam risks landing as test-only
code with no measured product win.

---

## 5. What is explicitly UNCHANGED

`build_request*` and Host synthesis (`http_client.rs:388-427`); `Sender::send`
and readiness (`connection.rs:239-261`); SSE reader / `SseMessageFilter` /
first-token filter (`http_client.rs:134-200,689-712`); `ChunkTiming` and all
response-side trace fields (`http_client.rs:28-57`); connection reuse / lease /
`ProtocolState` / H2 multiplexing accounting (`pool.rs:50-111,436-641`); post-send
cancellation (`race_cancel_after_send`, `SendCompletion`); the observer/metrics
path above the transport. The seam is strictly *below the `Sender`*.

---

## 6. Migration & risk

- **`DnsResolver` seam:** not deleted — it becomes a `TcpConnector` field
  (previously `PoolInner::resolver`, `pool.rs:116`). `CachingDnsResolver` /
  `HostLookup` (`resolver.rs`) are untouched. UDS/Duplex simply never resolve.
- **TLS over UDS/Duplex:** N/A. `TcpConnector` alone constructs the
  `TlsConnector` from `rustls_config(cfg)` (`connection.rs:471-496`). `ssl_verify`
  / `prepared_tls` remain TCP-only inputs. UDS is h1 cleartext by construction;
  Duplex is cleartext in-memory.
- **`SocketInfo` for non-TCP:** synthesized dummy addresses (loopback:0 for UDS,
  as today `connection.rs:436`; 0.0.0.0:0 for Duplex). `copy_socket`
  (`pool.rs:43-48`) still runs, writing those dummies into `local_ip/port` /
  `remote_ip/port`. No consumer requires a real peer address; this matches
  current UDS behavior. A future refinement could make `SocketInfo` an enum
  (`Tcp{..} | Local`), but that is out of scope — keep the dummy to avoid a
  ripple through `pool.rs`/`trace.rs`.
- **Pool-key collision risk (highest-attention item):** covered in §4 —
  `origin_key` MUST key UDS/Duplex by path/name, not the synthetic host, or two
  distinct in-memory/UDS endpoints alias one pool entry (silent cross-talk) or
  one endpoint fails the sticky binding check. Add a unit test: two different
  `unix:` paths → two entries; same path twice → one shared entry.
- **`establish`/`establish_with_resolver` public API:** these `pub` fns
  (`connection.rs:382,397`) are used by the pool, tests, **and
  `rust/runtime/src/graph/transport_bench.rs:516` (a live out-of-crate caller)**. They
  MUST be kept as shims over `TcpConnector::connect` (or `transport_bench`
  rewired to the connector) — a bare removal breaks the graph bench compile.
  Grep found `transport_bench` because it calls `establish(` even though it sets
  `uds_path` via struct-shorthand; do not treat `establish` as pool-only.
- **Offline sim / virtual `Clock` complicates `DuplexConnector` — called out
  honestly:** `DuplexConnector` over `tokio::io::duplex` is clean for an
  **online in-process mock on a `RealClock`** (proven here). But `tokio::io::duplex`
  is a real tokio primitive driven by the tokio reactor; under a `SimClock`
  virtual-time run (the offline-mock third mode), the in-process server's
  progress is not scheduled by the DES pump and its byte availability does not
  advance virtual time. So `DuplexConnector` does **not** by itself unify
  online-real and offline-sim. Full offline-sim unification requires the
  server side to be a Clock-driven steppable (the `dynamo-offline`
  `RequestSink`/`GraphSink` substrate), which is a separate, harder step. The
  honest framing: `DuplexConnector` delivers the **online in-process mock** seam
  now and is a stepping stone, not the offline-sim endgame.

---

## 7. Verification evidence

Scratch project: `~/tmp/connector-spec/` (throwaway; the target repo was not
modified). Deps pinned to workspace majors: `hyper 1.10` (client+server+http1+http2),
`hyper-util 0.1` (TokioIo), `tokio 1`, `http-body-util 0.1`, `async-trait 0.1`,
`url 2`, `bytes 1` — matching `rust/runtime/Cargo.toml` (the transport is now a
module of the `aiperf-runtime` crate; its deps live in that manifest, not a per-crate one) and
`Cargo.lock` (hyper 1.10.1, hyper-util 0.1.20, tokio 1.48, async-trait 0.1.89).

### 7.1 Trait + impls compile

`src/lib.rs` defines `Connector` (`#[async_trait(?Send)]`), `TcpConnector` (owns
`Rc<dyn DnsResolver>`), `UdsConnector`, `DuplexConnector` (over
`Rc<dyn DuplexEndpoint>`), the shared `handshake<I>` (verbatim from the real
crate, incl. `LocalExec` + `spawn_local`), a `Sender` enum with `TimedBody`, and
`select_connector(scheme) -> Rc<dyn Connector>` + a `PoolInner { connector:
Rc<dyn Connector> }` holder proving object-safety and the pool refactor.

```
$ cargo build
   Compiling connector-spec v0.0.0 (/home/anthony/tmp/connector-spec)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 6.78s
```

### 7.2 All three connectors round-trip end-to-end

`tests/e2e.rs` drives each connector against an in-process hyper `http1` server
on a `current_thread` runtime + `LocalSet` (the crate's `!Send` model): a real
loopback `TcpListener` for `TcpConnector`, a real `UnixListener` for
`UdsConnector`, and a `tokio::io::duplex` pipe with a `spawn_local`'d server for
`DuplexConnector`. Each performs a full POST and asserts the `pong` body.

```
$ cargo test
     Running tests/e2e.rs
running 1 test
test all_three_connectors_round_trip ... ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

### 7.3 Forced constraints (recorded from the compiler)

- **`async_trait(?Send)` is mandatory for `Rc<dyn Connector>`.** A native
  `async fn` in the trait is not dyn-compatible:
  ```
  error[E0038]: the trait `Connector` is not dyn compatible
     = note: for a trait to be dyn compatible it needs to allow building a vtable
     = help: consider moving `connect` to another trait
  ```
  Boxing the future via `async_trait` (or a manual `-> Pin<Box<dyn Future+'_>>`)
  is required, not stylistic.
- **`Rc<dyn Connector>` is correct; `Arc` is not.** The impls hold
  `Rc<dyn DnsResolver>` / `Rc<dyn DuplexEndpoint>` (`!Send`), so an `Arc<dyn
  Connector>` requiring `Send + Sync` would not type-check. `Rc` matches the
  crate's existing `Rc<dyn DnsResolver>` (`pool.rs:116`) / `Rc<dyn Clock>`.
- **`(Sender, SocketInfo)` return keeps the h1/h2 decision inside the connector.**
  Because `handshake(io, use_h2)` and the ALPN/force-flag logic differ per stream
  source, returning a raw IO would relocate — not remove — the mode branch;
  returning the already-handshaked `Sender` removes it.

### 7.4 Reproduce

```bash
cd ~/tmp/connector-spec
cargo build      # trait + 3 impls compile on workspace dep majors
cargo test       # all_three_connectors_round_trip ... ok
```

---

## 8. Summary

- **Trait:** `#[async_trait(?Send)] trait Connector { async fn connect(&self,
  &Url, &ClientConfig, Rc<dyn Clock>, &mut TraceData) -> Result<(Sender,
  SocketInfo), ErrorDetails>; }`, held as `Rc<dyn Connector>`.
- **Impls:** `TcpConnector` (today's DNS+TLS+socket-opts TCP path, owning the
  `DnsResolver`) / `UdsConnector` (the extracted `if let uds_path` UDS h1 branch)
  / `DuplexConnector` (NEW in-memory `tokio::io::duplex` to a `spawn_local`'d
  in-process handler).
- **Selection:** by URL scheme at endpoint-prepare time (`http(s)`→Tcp,
  `unix`→Uds product-reachable after Python-side gates are relaxed; `mem`→Duplex
  **test/bench-only**, no product wiring); `ClientConfig::uds_path` is deleted.
- **Touch points:** `connection.rs:426-444` (delete branch), `:446-508` (→
  `TcpConnector::connect`), `:382-416` (shim/wrap — keep as shims,
  `transport_bench.rs:516` calls `establish` directly); `pool.rs:116` (resolver →
  connector field), `:309-315` & `:400-407` (establish → `connector.connect`),
  `:28-35` (`origin_key` keys UDS/Duplex by path/name); `http_client.rs:276` &
  `:330` (establish → connector), add `connector` field; `config/defaults.rs:199-202,220`
  (drop `uds_path`); **`rust/runtime/src/graph/transport_bench.rs:462-506,516`**
  (migrate the live `uds_path` writer to `unix:`-scheme connector selection — same
  change, or the `aiperf-runtime` crate's `graph` module fails to compile); **`src/aiperf/config/endpoint.py`
  ~460-475** (relax netloc/hostname + whitelist `unix` + skip port parse, three
  gates, to make `unix:` product-reachable).
- **Verified constraints:** `async_trait(?Send)` mandatory (native async fn not
  dyn-compatible), `Rc<dyn Connector>` (not `Arc`), `(Sender, SocketInfo)` return
  keeps handshake inside the connector. All three round-trip end-to-end.
- **Honest caveat:** `DuplexConnector` cleanly delivers the online in-process
  mock; full offline-sim (`SimClock`) unification needs a Clock-driven steppable
  server and is a separate step.
```
