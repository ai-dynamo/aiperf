// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Discovery-free connection seam for the velo cell transport.
//!
//! A cell reaches the controller with **`velo.connect(Endpoint)`** — velo's
//! address-first bootstrap handshake (`ajcasagrande/velo` `feat/connect-by-endpoint`):
//! it dials the controller's operator/launcher-injected endpoint and learns the
//! controller-presented `PeerInfo` via the unauthenticated `_hello` handshake. The
//! cell retains those exact bytes and the resolved dial target until the signed
//! registration reply binds them to its provisioned controller key. No discovery
//! backend or bootstrap side-channel is involved; the only a-priori network fact is
//! the endpoint (`AIPERF_CELL_CONTROLLER_ADDR`, `tcp://HOST:PORT`; also `uds://PATH`
//! for a pure-local run without HTTP artifact shipping).
//!
//! The coordinate stays a `tcp://HOST:PORT` string in every shipping deployment so
//! the HTTP artifact plane (`engine::artifact_shipping`, which derives its
//! authority by swapping the port on the same coordinate) keeps working.

use std::net::SocketAddr;
#[cfg(unix)]
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use velo::transports::tcp::TcpTransportBuilder;
use velo::{Endpoint, PeerInfo, Transport, Velo};

use super::CellTransportError;

/// Consume a launcher-inherited TCP listener named by `env_name`.
///
/// The environment value carries only a decimal descriptor. It is removed before
/// acquisition, making descriptor ownership one-shot; the returned listener is
/// marked close-on-exec before the controller can launch any other process.
pub(crate) fn take_inherited_tcp_listener(
    env_name: &'static str,
) -> Result<Option<std::net::TcpListener>> {
    let value = take_inherited_descriptor_value(env_name);
    let Some(value) = value else {
        return Ok(None);
    };
    #[cfg(unix)]
    {
        let fd = value
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("{env_name} must be a decimal file descriptor"))?
            .parse::<i32>()
            .map_err(|_| anyhow::anyhow!("{env_name} must be a decimal file descriptor"))?;
        if fd < 0 {
            anyhow::bail!("{env_name} must be a non-negative file descriptor");
        }
        validate_tcp_listener(env_name, fd)?;
        let owned = unsafe { OwnedFd::from_raw_fd(fd) };
        set_close_on_exec(env_name, &owned)?;
        let listener = std::net::TcpListener::from(owned);
        listener
            .local_addr()
            .with_context(|| format!("reading inherited {env_name} local address"))?;
        Ok(Some(listener))
    }
    #[cfg(not(unix))]
    {
        let _ = value;
        anyhow::bail!("{env_name} inherited listeners are unsupported on this platform")
    }
}

/// Consume a launcher-inherited pipe reader named by `env_name`.
///
/// The descriptor is a test/deployment lifetime gate, not a channel carrying
/// sensitive material. It must be a pipe endpoint and is made close-on-exec
/// before it becomes controller-owned.
pub(crate) fn take_inherited_pipe_reader(env_name: &'static str) -> Result<Option<std::fs::File>> {
    let value = take_inherited_descriptor_value(env_name);
    let Some(value) = value else {
        return Ok(None);
    };
    #[cfg(unix)]
    {
        let fd = parse_inherited_fd(env_name, value)?;
        validate_pipe_reader(env_name, fd)?;
        let owned = unsafe { OwnedFd::from_raw_fd(fd) };
        set_close_on_exec(env_name, &owned)?;
        set_nonblocking(env_name, &owned)?;
        Ok(Some(std::fs::File::from(owned)))
    }
    #[cfg(not(unix))]
    {
        let _ = value;
        anyhow::bail!("{env_name} inherited pipe readers are unsupported on this platform")
    }
}

/// Wait for the inherited scratch-lifetime gate to reach EOF.
///
/// Reading is readiness-driven and never occupies a Tokio worker thread.
#[cfg(unix)]
pub async fn wait_for_pipe_eof(reader: std::fs::File) -> Result<()> {
    let reader = tokio::io::unix::AsyncFd::new(reader)
        .context("registering inherited scratch hold pipe with Tokio")?;
    let mut byte = [0_u8; 1];
    loop {
        let mut readiness = reader
            .readable()
            .await
            .context("waiting for inherited scratch hold pipe")?;
        match readiness.try_io(|inner| {
            // SAFETY: the owned pipe descriptor and one-byte output buffer remain valid.
            let count = unsafe {
                libc::read(
                    inner.get_ref().as_raw_fd(),
                    byte.as_mut_ptr().cast(),
                    byte.len(),
                )
            };
            if count < 0 {
                Err(std::io::Error::last_os_error())
            } else {
                Ok(count)
            }
        }) {
            Ok(Ok(0)) => return Ok(()),
            Ok(Ok(_)) => anyhow::bail!("inherited scratch hold pipe unexpectedly carried data"),
            Ok(Err(error)) => return Err(error).context("reading inherited scratch hold pipe"),
            Err(_) => continue,
        }
    }
}

#[cfg(not(unix))]
pub async fn wait_for_pipe_eof(_reader: std::fs::File) -> Result<()> {
    anyhow::bail!("inherited scratch hold pipes are unsupported on this platform")
}

fn take_inherited_descriptor_value(env_name: &'static str) -> Option<std::ffi::OsString> {
    let value = std::env::var_os(env_name);
    // SAFETY: controller bootstrap runs before the Tokio runtime or worker threads.
    unsafe { std::env::remove_var(env_name) };
    value
}

#[cfg(unix)]
fn parse_inherited_fd(env_name: &'static str, value: std::ffi::OsString) -> Result<i32> {
    let fd = value
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("{env_name} must be a decimal file descriptor"))?
        .parse::<i32>()
        .map_err(|_| anyhow::anyhow!("{env_name} must be a decimal file descriptor"))?;
    if fd < 0 {
        anyhow::bail!("{env_name} must be a non-negative file descriptor");
    }
    Ok(fd)
}

#[cfg(unix)]
fn validate_tcp_listener(env_name: &'static str, fd: i32) -> Result<()> {
    if unsafe { libc::fcntl(fd, libc::F_GETFD) } < 0 {
        return Err(std::io::Error::last_os_error())
            .with_context(|| format!("opening inherited {env_name}"));
    }
    let mut kind: libc::c_int = 0;
    let mut kind_len = std::mem::size_of_val(&kind) as libc::socklen_t;
    // SAFETY: fd is a validated borrowed descriptor; output is initialized local storage.
    let result = unsafe {
        libc::getsockopt(
            fd,
            libc::SOL_SOCKET,
            libc::SO_TYPE,
            &mut kind as *mut _ as *mut _,
            &mut kind_len,
        )
    };
    if result != 0 || kind != libc::SOCK_STREAM {
        anyhow::bail!("{env_name} is not a TCP stream listener");
    }
    let mut accepting: libc::c_int = 0;
    let mut accepting_len = std::mem::size_of_val(&accepting) as libc::socklen_t;
    // SAFETY: fd is a validated borrowed descriptor; SO_ACCEPTCONN distinguishes a listener.
    let result = unsafe {
        libc::getsockopt(
            fd,
            libc::SOL_SOCKET,
            libc::SO_ACCEPTCONN,
            &mut accepting as *mut _ as *mut _,
            &mut accepting_len,
        )
    };
    if result != 0 || accepting != 1 {
        anyhow::bail!("{env_name} is not a TCP stream listener");
    }
    Ok(())
}

#[cfg(unix)]
fn validate_pipe_reader(env_name: &'static str, fd: i32) -> Result<()> {
    let mut status = std::mem::MaybeUninit::<libc::stat>::uninit();
    // SAFETY: `status` is initialized by fstat on success.
    if unsafe { libc::fstat(fd, status.as_mut_ptr()) } != 0 {
        return Err(std::io::Error::last_os_error())
            .with_context(|| format!("inspecting inherited {env_name}"));
    }
    // SAFETY: fstat succeeded above.
    let status = unsafe { status.assume_init() };
    if status.st_mode & libc::S_IFMT != libc::S_IFIFO {
        anyhow::bail!("{env_name} is not a pipe reader");
    }
    // SAFETY: fcntl only reads flags on the validated borrowed descriptor.
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
    if flags < 0 {
        return Err(std::io::Error::last_os_error())
            .with_context(|| format!("reading inherited {env_name} status flags"));
    }
    if flags & libc::O_ACCMODE != libc::O_RDONLY {
        anyhow::bail!("{env_name} is not a pipe reader");
    }
    Ok(())
}

#[cfg(unix)]
fn set_close_on_exec(env_name: &'static str, fd: &OwnedFd) -> Result<()> {
    // SAFETY: fcntl only reads/updates flags on the owned descriptor.
    let flags = unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_GETFD) };
    if flags < 0 {
        return Err(std::io::Error::last_os_error())
            .with_context(|| format!("reading inherited {env_name} descriptor flags"));
    }
    // SAFETY: as above, update only FD_CLOEXEC.
    if unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_SETFD, flags | libc::FD_CLOEXEC) } < 0 {
        return Err(std::io::Error::last_os_error())
            .with_context(|| format!("setting close-on-exec on inherited {env_name}"));
    }
    Ok(())
}

#[cfg(unix)]
fn set_nonblocking(env_name: &'static str, fd: &OwnedFd) -> Result<()> {
    // SAFETY: fcntl only reads/updates flags on the owned descriptor.
    let flags = unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_GETFL) };
    if flags < 0 {
        return Err(std::io::Error::last_os_error())
            .with_context(|| format!("reading inherited {env_name} status flags"));
    }
    // SAFETY: as above, update only O_NONBLOCK.
    if unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_SETFL, flags | libc::O_NONBLOCK) } < 0 {
        return Err(std::io::Error::last_os_error())
            .with_context(|| format!("setting nonblocking on inherited {env_name}"));
    }
    Ok(())
}

/// How long a cell keeps retrying `connect` before giving up (the controller may
/// not have bound its listener yet when a k8s cell pod starts first).
const CONNECT_TIMEOUT: Duration = Duration::from_secs(60);

/// One retry interval for `connect`.
const CONNECT_RETRY_INTERVAL: Duration = Duration::from_millis(200);

const HANDLER_READY_RETRY_INTERVAL: Duration = Duration::from_millis(10);

/// The exact address selected when resolving and dialing the controller coordinate.
#[derive(Clone, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub(crate) enum DialedControllerAddress {
    /// The resolved TCP socket address passed to Velo.
    Tcp(SocketAddr),
    /// The exact Unix-domain socket pathname bytes passed to Velo.
    #[cfg(unix)]
    Uds(Box<[u8]>),
}

/// Controller identity returned by Velo's endpoint handshake plus the exact dial target.
#[derive(Clone)]
pub(crate) struct ConnectedController {
    peer: PeerInfo,
    dialed: DialedControllerAddress,
}

impl ConnectedController {
    pub(crate) fn peer(&self) -> &PeerInfo {
        &self.peer
    }

    pub(crate) fn binding(&self) -> Result<ControllerPeerBinding, CellTransportError> {
        ControllerPeerBinding::new(&self.peer, self.dialed.clone())
    }

    #[cfg(test)]
    pub(crate) fn from_parts(peer: PeerInfo, dialed: DialedControllerAddress) -> Self {
        Self { peer, dialed }
    }
}

/// Exact controller identity and dial target covered by registration signatures.
#[derive(Clone, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub(crate) struct ControllerPeerBinding {
    instance_id: Vec<u8>,
    worker_address: Vec<u8>,
    dialed: DialedControllerAddress,
}

impl ControllerPeerBinding {
    pub(crate) fn new(
        peer: &PeerInfo,
        dialed: DialedControllerAddress,
    ) -> Result<Self, CellTransportError> {
        Ok(Self {
            instance_id: rmp_serde::to_vec(&peer.instance_id()).map_err(|_| {
                CellTransportError::Authentication("controller binding encoding failed")
            })?,
            worker_address: peer.worker_address().as_bytes().to_vec(),
            dialed,
        })
    }

    pub(crate) fn matches_peer(&self, peer: &PeerInfo) -> Result<bool, CellTransportError> {
        let instance_id = rmp_serde::to_vec(&peer.instance_id()).map_err(|_| {
            CellTransportError::Authentication("controller binding encoding failed")
        })?;
        Ok(self.instance_id == instance_id
            && self.worker_address.as_slice() == peer.worker_address().as_bytes())
    }

    pub(crate) fn append_transcript(&self, transcript: &mut Vec<u8>) {
        append_len_prefixed(transcript, &self.instance_id);
        append_len_prefixed(transcript, &self.worker_address);
        match &self.dialed {
            DialedControllerAddress::Tcp(address) => {
                transcript.push(1);
                let address_len = match address {
                    SocketAddr::V4(_) => 1_u64 + 4 + 2,
                    SocketAddr::V6(_) => 1_u64 + 16 + 2 + 4 + 4,
                };
                transcript.extend_from_slice(&address_len.to_le_bytes());
                match address {
                    SocketAddr::V4(address) => {
                        transcript.push(4);
                        transcript.extend_from_slice(&address.ip().octets());
                        transcript.extend_from_slice(&address.port().to_be_bytes());
                    }
                    SocketAddr::V6(address) => {
                        transcript.push(6);
                        transcript.extend_from_slice(&address.ip().octets());
                        transcript.extend_from_slice(&address.port().to_be_bytes());
                        transcript.extend_from_slice(&address.flowinfo().to_be_bytes());
                        transcript.extend_from_slice(&address.scope_id().to_be_bytes());
                    }
                }
            }
            #[cfg(unix)]
            DialedControllerAddress::Uds(path) => {
                transcript.push(2);
                append_len_prefixed(transcript, path);
            }
        }
    }
}

fn append_len_prefixed(output: &mut Vec<u8>, bytes: &[u8]) {
    output.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
    output.extend_from_slice(bytes);
}

/// Absolute bound shared by connection, readiness, registration, and attestation.
#[derive(Clone, Copy)]
pub(crate) struct RegistrationDeadline(tokio::time::Instant);

impl RegistrationDeadline {
    pub(crate) fn after(duration: Duration) -> Self {
        Self(tokio::time::Instant::now() + duration)
    }

    pub(crate) fn instant(self) -> tokio::time::Instant {
        self.0
    }

    pub(crate) fn is_elapsed(self) -> bool {
        tokio::time::Instant::now() >= self.0
    }

    pub(crate) fn for_registration() -> Self {
        let seconds = std::env::var("AIPERF_CELL_REGISTER_TIMEOUT_SECS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(300);
        Self::after(Duration::from_secs(seconds))
    }
}

/// How the controller or a cell binds its velo messaging transport.
pub enum BindSpec {
    /// Unix domain socket at this path (pure-local unix run without HTTP artifact
    /// shipping — the coordinate is then `uds://PATH`).
    #[cfg(unix)]
    UdsPath(PathBuf),
    /// TCP on an OS-assigned loopback port.
    TcpLoopback,
    /// TCP bound to a fixed address (the k8s controller at its operator-known port).
    TcpBind(SocketAddr),
    /// TCP over a caller-provided, already-bound listener — lets the caller read the
    /// OS-assigned port back (for the endpoint string) before handing it to velo.
    TcpListener(std::net::TcpListener),
}

/// Build a velo instance bound per `bind`. The cellular control plane is off the
/// per-request hot path, so a single messaging transport is sufficient.
pub async fn build_velo(bind: BindSpec) -> Result<Arc<Velo>> {
    let transport: Arc<dyn Transport> = match bind {
        #[cfg(unix)]
        BindSpec::UdsPath(path) => Arc::new(
            velo::transports::uds::UdsTransportBuilder::new()
                .socket_path(path)
                .build()
                .context("build uds transport")?,
        ),
        BindSpec::TcpLoopback => build_tcp_transport("127.0.0.1:0")?,
        BindSpec::TcpBind(addr) => build_tcp_transport(addr)?,
        BindSpec::TcpListener(listener) => Arc::new(
            TcpTransportBuilder::new()
                .from_listener(listener)
                .context("velo tcp from_listener")?
                .build()
                .context("build velo tcp transport")?,
        ),
    };
    Velo::builder()
        .add_transport(transport)
        .build()
        .await
        .context("build velo instance")
}

fn build_tcp_transport(addr: impl std::net::ToSocketAddrs) -> Result<Arc<dyn Transport>> {
    let listener = std::net::TcpListener::bind(addr).context("bind velo tcp listener")?;
    Ok(Arc::new(
        TcpTransportBuilder::new()
            .from_listener(listener)
            .context("velo tcp from_listener")?
            .build()
            .context("build velo tcp transport")?,
    ))
}

/// Parse a controller endpoint coordinate into a velo [`Endpoint`]: `tcp://HOST:PORT`
/// or `uds://PATH` (unix, pure-local).
async fn parse_endpoint_until(
    coordinate: &str,
    deadline: RegistrationDeadline,
) -> Result<(Endpoint, DialedControllerAddress), CellTransportError> {
    if let Some(addr) = coordinate.strip_prefix("tcp://") {
        if deadline.is_elapsed() {
            return Err(connection_deadline_error(None));
        }
        let mut addresses =
            tokio::time::timeout_at(deadline.instant(), tokio::net::lookup_host(addr))
                .await
                .map_err(|_| connection_deadline_error(None))?
                .map_err(|error| CellTransportError::Io(error.to_string()))?;
        let socket = addresses.next().ok_or_else(|| {
            CellTransportError::Io("controller endpoint resolved to no addresses".to_owned())
        })?;
        return Ok((Endpoint::Tcp(socket), DialedControllerAddress::Tcp(socket)));
    }
    if let Some(path) = coordinate.strip_prefix("uds://") {
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStrExt as _;

            let path = PathBuf::from(path);
            let dialed = DialedControllerAddress::Uds(
                path.as_os_str().as_bytes().to_vec().into_boxed_slice(),
            );
            return Ok((Endpoint::Uds(path), dialed));
        }
        #[cfg(not(unix))]
        {
            let _ = path;
            return Err(CellTransportError::Io(
                "uds controller endpoints are unix-only".to_owned(),
            ));
        }
    }
    Err(CellTransportError::Io(
        "unrecognized controller endpoint; expected tcp://HOST:PORT or uds://PATH".to_owned(),
    ))
}

fn connection_deadline_error(last_error: Option<&str>) -> CellTransportError {
    match last_error {
        Some(error) => CellTransportError::Io(format!(
            "controller connection deadline elapsed; last connection error: {error}"
        )),
        None => CellTransportError::Io("controller connection deadline elapsed".to_owned()),
    }
}

/// Connect to the controller at `coordinate`, retrying until it is reachable or
/// `CONNECT_TIMEOUT` elapses, and return its `PeerInfo`. Wraps `velo.connect`.
pub async fn connect_controller(velo: &Velo, coordinate: &str) -> Result<PeerInfo> {
    connect_controller_until(
        velo,
        coordinate,
        RegistrationDeadline::after(CONNECT_TIMEOUT),
    )
    .await
    .map(|controller| controller.peer)
    .map_err(anyhow::Error::new)
}

/// Connect to one resolved controller address before the shared registration deadline.
pub(crate) async fn connect_controller_until(
    velo: &Velo,
    coordinate: &str,
    deadline: RegistrationDeadline,
) -> Result<ConnectedController, CellTransportError> {
    let (endpoint, dialed) = parse_endpoint_until(coordinate, deadline).await?;
    let mut last_error = None;
    loop {
        match tokio::time::timeout_at(deadline.instant(), velo.connect(endpoint.clone())).await {
            Ok(Ok(peer)) => {
                return Ok(ConnectedController { peer, dialed });
            }
            Ok(Err(error)) => last_error = Some(error.to_string()),
            Err(_) => {
                return Err(connection_deadline_error(last_error.as_deref()));
            }
        }
        if tokio::time::timeout_at(
            deadline.instant(),
            tokio::time::sleep(CONNECT_RETRY_INTERVAL),
        )
        .await
        .is_err()
        {
            return Err(connection_deadline_error(last_error.as_deref()));
        }
    }
}

/// Wait for a named controller handler using Velo's typed handler inventory.
pub(crate) async fn await_handler_until(
    velo: &Velo,
    controller: &ConnectedController,
    handler: &'static str,
    deadline: RegistrationDeadline,
) -> Result<(), CellTransportError> {
    loop {
        tokio::time::timeout_at(
            deadline.instant(),
            velo.refresh_handlers(controller.peer.instance_id()),
        )
        .await
        .map_err(|_| CellTransportError::ReadinessTimeout { handler })?
        .map_err(|error| CellTransportError::Io(error.to_string()))?;
        let handlers = tokio::time::timeout_at(
            deadline.instant(),
            velo.available_handlers(controller.peer.instance_id()),
        )
        .await
        .map_err(|_| CellTransportError::ReadinessTimeout { handler })?
        .map_err(|error| CellTransportError::Io(error.to_string()))?;
        if handlers.iter().any(|candidate| candidate == handler) {
            return Ok(());
        }
        tokio::time::timeout_at(
            deadline.instant(),
            tokio::time::sleep(HANDLER_READY_RETRY_INTERVAL),
        )
        .await
        .map_err(|_| CellTransportError::ReadinessTimeout { handler })?;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(unix)]
    use std::os::fd::{AsRawFd, FromRawFd};
    #[cfg(unix)]
    use std::sync::{Mutex, OnceLock};

    #[cfg(unix)]
    static INHERITED_DESCRIPTOR_ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

    #[cfg(unix)]
    fn inherited_descriptor_env_lock() -> std::sync::MutexGuard<'static, ()> {
        INHERITED_DESCRIPTOR_ENV_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .expect("inherited descriptor environment lock")
    }

    #[cfg(unix)]
    #[test]
    fn inherited_listener_is_one_shot_stream_listener_with_close_on_exec() {
        let _guard = inherited_descriptor_env_lock();
        let source = std::net::TcpListener::bind("127.0.0.1:0").expect("listener");
        // SAFETY: the duplicate is transferred to the one-shot test environment.
        let duplicate = unsafe { libc::dup(source.as_raw_fd()) };
        assert!(duplicate >= 0, "duplicate listener");
        unsafe { std::env::set_var("AIPERF_TEST_INHERITED_LISTENER", duplicate.to_string()) };
        let listener = take_inherited_tcp_listener("AIPERF_TEST_INHERITED_LISTENER")
            .expect("consume listener")
            .expect("listener supplied");
        assert!(std::env::var_os("AIPERF_TEST_INHERITED_LISTENER").is_none());
        assert!(
            take_inherited_tcp_listener("AIPERF_TEST_INHERITED_LISTENER")
                .expect("second consumption")
                .is_none()
        );
        // SAFETY: fcntl only reads the returned listener's descriptor flags.
        let flags = unsafe { libc::fcntl(listener.as_raw_fd(), libc::F_GETFD) };
        assert_ne!(flags & libc::FD_CLOEXEC, 0);
        assert!(listener.local_addr().is_ok());
    }

    #[cfg(unix)]
    #[test]
    fn inherited_listener_rejects_non_socket_descriptor_and_clears_env() {
        let _guard = inherited_descriptor_env_lock();
        let mut fds = [0; 2];
        // SAFETY: pipe2 initializes the locally owned descriptor slots.
        assert_eq!(unsafe { libc::pipe2(fds.as_mut_ptr(), libc::O_CLOEXEC) }, 0);
        unsafe { std::env::set_var("AIPERF_TEST_INHERITED_LISTENER", fds[0].to_string()) };
        let error = take_inherited_tcp_listener("AIPERF_TEST_INHERITED_LISTENER")
            .expect_err("pipe must not be accepted as listener");
        assert!(error.to_string().contains("TCP stream listener"));
        assert!(std::env::var_os("AIPERF_TEST_INHERITED_LISTENER").is_none());
        // Prevalidation borrows rejected descriptors; the launcher retains ownership.
        unsafe {
            libc::close(fds[0]);
            libc::close(fds[1]);
        }
    }

    #[cfg(unix)]
    #[test]
    fn rejected_reused_fd_does_not_close_the_replacement() {
        let _guard = inherited_descriptor_env_lock();
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let replacement_source = std::fs::File::open("/dev/null").unwrap();
        let minimum_target = replacement_source.as_raw_fd().saturating_add(1).max(64);
        let stale =
            unsafe { libc::fcntl(listener.as_raw_fd(), libc::F_DUPFD_CLOEXEC, minimum_target) };
        assert!(stale >= 64, "create deterministic stale descriptor slot");
        assert_ne!(stale, replacement_source.as_raw_fd());
        assert!(
            unsafe { libc::fcntl(stale, libc::F_GETFD) } >= 0,
            "the target slot must remain owned until atomic replacement"
        );
        assert_eq!(
            unsafe { libc::dup2(replacement_source.as_raw_fd(), stale) },
            stale,
            "install deterministic replacement descriptor"
        );
        let replacement = unsafe { OwnedFd::from_raw_fd(stale) };
        unsafe { std::env::set_var("AIPERF_TEST_INHERITED_LISTENER", stale.to_string()) };
        assert!(take_inherited_tcp_listener("AIPERF_TEST_INHERITED_LISTENER").is_err());
        assert!(
            unsafe { libc::fcntl(replacement.as_raw_fd(), libc::F_GETFD) } >= 0,
            "rejected stale fd must not close replacement"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn inherited_pipe_requires_reader_and_waits_for_eof_without_blocking() {
        let guard = inherited_descriptor_env_lock();
        let mut fds = [0; 2];
        // SAFETY: pipe2 initializes the locally owned descriptor slots.
        assert_eq!(unsafe { libc::pipe2(fds.as_mut_ptr(), libc::O_CLOEXEC) }, 0);
        unsafe { std::env::set_var("AIPERF_TEST_INHERITED_PIPE", fds[1].to_string()) };
        assert!(take_inherited_pipe_reader("AIPERF_TEST_INHERITED_PIPE").is_err());
        assert!(std::env::var_os("AIPERF_TEST_INHERITED_PIPE").is_none());
        // Prevalidation borrows the rejected writer; both endpoints remain caller-owned.
        let reader = unsafe { std::fs::File::from_raw_fd(fds[0]) };
        let rejected_writer = unsafe { std::fs::File::from_raw_fd(fds[1]) };
        drop(rejected_writer);
        let mut fds = [0; 2];
        // SAFETY: create a distinct pipe for the valid reader/EOF path.
        assert_eq!(unsafe { libc::pipe2(fds.as_mut_ptr(), libc::O_CLOEXEC) }, 0);
        unsafe { std::env::set_var("AIPERF_TEST_INHERITED_PIPE", fds[0].to_string()) };
        let gate = take_inherited_pipe_reader("AIPERF_TEST_INHERITED_PIPE")
            .expect("consume pipe")
            .expect("pipe supplied");
        drop(guard);
        // SAFETY: fcntl only reads the returned descriptor flags.
        let descriptor_flags = unsafe { libc::fcntl(gate.as_raw_fd(), libc::F_GETFD) };
        let status_flags = unsafe { libc::fcntl(gate.as_raw_fd(), libc::F_GETFL) };
        assert_ne!(descriptor_flags & libc::FD_CLOEXEC, 0);
        assert_ne!(status_flags & libc::O_NONBLOCK, 0);
        let writer = unsafe { std::fs::File::from_raw_fd(fds[1]) };
        drop(writer);
        tokio::time::timeout(Duration::from_millis(200), wait_for_pipe_eof(gate))
            .await
            .expect("EOF wait must be readiness-driven")
            .expect("EOF wait");
        drop(reader);
    }

    // Catches omission of the fixed-width dial-address length from the signature transcript.
    #[test]
    fn tcp_controller_binding_has_canonical_literal_encoding() {
        let binding = ControllerPeerBinding {
            instance_id: Vec::new(),
            worker_address: Vec::new(),
            dialed: DialedControllerAddress::Tcp("127.0.0.1:9500".parse().unwrap()),
        };
        let mut transcript = Vec::new();
        binding.append_transcript(&mut transcript);

        assert_eq!(
            transcript,
            vec![
                0, 0, 0, 0, 0, 0, 0, 0, // empty instance id
                0, 0, 0, 0, 0, 0, 0, 0, // empty worker address
                1, // TCP dial variant
                7, 0, 0, 0, 0, 0, 0, 0, // fixed-width dial-address length
                4, 127, 0, 0, 1, 0x25, 0x1c,
            ]
        );
    }

    // Catches DNS resolution happening before the caller's registration deadline.
    #[tokio::test]
    async fn expired_registration_deadline_prevents_tcp_resolution() {
        let cell = build_velo(BindSpec::TcpLoopback).await.expect("cell velo");
        let error = match connect_controller_until(
            &cell,
            "tcp://not a socket address",
            RegistrationDeadline::after(Duration::ZERO),
        )
        .await
        {
            Ok(_) => panic!("an expired deadline must prevent resolution"),
            Err(error) => error,
        };

        assert_eq!(
            error,
            CellTransportError::Io("controller connection deadline elapsed".to_owned())
        );
    }

    #[tokio::test]
    async fn parse_recognizes_tcp_and_uds_endpoints() {
        let deadline = RegistrationDeadline::after(Duration::from_secs(1));
        assert!(matches!(
            parse_endpoint_until("tcp://127.0.0.1:9500", deadline)
                .await
                .unwrap(),
            (Endpoint::Tcp(_), DialedControllerAddress::Tcp(_))
        ));
        assert!(parse_endpoint_until("http://nope", deadline).await.is_err());
        assert!(
            parse_endpoint_until("tcp://not-an-addr", deadline)
                .await
                .is_err()
        );
        // `localhost` stands in for the DNS name supplied by Kubernetes.
        assert!(matches!(
            parse_endpoint_until("tcp://localhost:9500", deadline)
                .await
                .unwrap(),
            (Endpoint::Tcp(_), DialedControllerAddress::Tcp(_))
        ));
        #[cfg(unix)]
        assert!(matches!(
            parse_endpoint_until("uds:///tmp/controller.sock", deadline)
                .await
                .unwrap(),
            (Endpoint::Uds(_), DialedControllerAddress::Uds(_))
        ));
    }

    // A cell `connect`s the controller by TCP address alone (no PeerInfo), and the
    // returned peer is the exact identity presented by the endpoint handshake.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn connect_controller_bootstraps_by_endpoint() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let controller = build_velo(BindSpec::TcpListener(listener))
            .await
            .expect("controller velo");

        let cell = build_velo(BindSpec::TcpLoopback).await.expect("cell velo");
        let peer = connect_controller(&cell, &format!("tcp://{addr}"))
            .await
            .expect("connect");
        assert_eq!(peer.instance_id(), controller.instance_id());
        assert_eq!(
            peer.worker_address().as_bytes(),
            controller
                .messenger()
                .peer_info()
                .worker_address()
                .as_bytes()
        );
        assert_ne!(
            peer.worker_address().as_bytes(),
            controller.peer_info().worker_address().as_bytes(),
            "Velo::peer_info includes streaming addresses that _hello does not publish"
        );
    }

    // Catches readiness retries that infer state by parsing English send errors.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn registration_waits_for_typed_handler_publication_under_one_deadline() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let controller = build_velo(BindSpec::TcpListener(listener))
            .await
            .expect("controller velo");
        let cell = build_velo(BindSpec::TcpLoopback).await.expect("cell velo");
        let connected = connect_controller_until(
            &cell,
            &format!("tcp://{addr}"),
            RegistrationDeadline::after(Duration::from_millis(200)),
        )
        .await
        .expect("connect");

        let delayed_controller = Arc::clone(&controller);
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(40)).await;
            delayed_controller
                .register_handler(
                    velo::Handler::unary_handler_async(
                        super::super::HANDLER_REGISTER,
                        |_ctx| async { Ok(Some(bytes::Bytes::new())) },
                    )
                    .build(),
                )
                .expect("publish register handler");
        });
        await_handler_until(
            &cell,
            &connected,
            super::super::HANDLER_REGISTER,
            RegistrationDeadline::after(Duration::from_millis(200)),
        )
        .await
        .expect("published handler");

        let timeout = await_handler_until(
            &cell,
            &connected,
            "aiperf.cell.absent",
            RegistrationDeadline::after(Duration::from_millis(20)),
        )
        .await
        .unwrap_err();
        assert_eq!(
            timeout,
            CellTransportError::ReadinessTimeout {
                handler: "aiperf.cell.absent"
            }
        );
    }
}
