from __future__ import annotations

import contextlib
import json
import os
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from urllib.parse import urlparse


def find_toxiproxy_bin() -> str | None:
    override = os.environ.get("AIPERF_TOXIPROXY_BIN")
    if override and os.access(override, os.X_OK):
        return override
    return shutil.which("toxiproxy-server")


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


@dataclass(slots=True)
class Toxic:
    type: str
    attributes: dict[str, int]
    stream: str = "downstream"
    toxicity: float = 1.0


_NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class ToxiproxyClient:
    def __init__(self, admin_url: str, timeout: float = 5.0) -> None:
        self.admin_url = admin_url
        self.timeout = timeout

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            f"{self.admin_url}{path}",
            data=data,
            method=method,
            headers={"Content-Type": "application/json"} if data else {},
        )
        with _NO_PROXY_OPENER.open(req, timeout=self.timeout) as resp:
            payload = resp.read()
        if not payload:
            return {}
        return json.loads(payload)

    def add_proxy(self, name: str, listen: str, upstream: str) -> dict:
        return self._request(
            "POST",
            "/proxies",
            {"name": name, "listen": listen, "upstream": upstream, "enabled": True},
        )

    def add_toxic(self, proxy_name: str, toxic: Toxic) -> dict:
        return self._request(
            "POST",
            f"/proxies/{proxy_name}/toxics",
            {
                "type": toxic.type,
                "stream": toxic.stream,
                "toxicity": toxic.toxicity,
                "attributes": toxic.attributes,
            },
        )


@contextlib.contextmanager
def start_toxiproxy(bin_path: str | None = None) -> Iterator[ToxiproxyClient]:
    binary = bin_path or find_toxiproxy_bin()
    if binary is None:
        raise RuntimeError("toxiproxy-server not on PATH; set AIPERF_TOXIPROXY_BIN")
    admin_port = _free_port()
    proc = subprocess.Popen(
        [binary, "-host", "127.0.0.1", "-port", str(admin_port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    client = ToxiproxyClient(f"http://127.0.0.1:{admin_port}")
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                client._request("GET", "/version")
                break
            except (urllib.error.URLError, ConnectionError):
                time.sleep(0.1)
        else:
            raise RuntimeError(f"toxiproxy-server failed to come up on :{admin_port}")
        yield client
    finally:
        proc.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=3)
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=2)


@contextlib.contextmanager
def chaos_proxy(upstream_url: str, toxics: list[Toxic]) -> Iterator[str]:
    """Yield a local toxiproxy URL forwarding to ``upstream_url`` with toxics applied.

    Requires ``upstream_url`` to include host and port. Starts a temporary
    toxiproxy server, creates one proxy named ``aiperf-upstream``, installs each
    toxic, preserves the upstream scheme/path in the yielded URL, and tears the
    proxy server down on exit.
    """
    parsed = urlparse(upstream_url)
    if not parsed.hostname or not parsed.port:
        raise ValueError(f"upstream_url must include host:port; got {upstream_url!r}")
    upstream = f"{parsed.hostname}:{parsed.port}"
    listen_port = _free_port()
    with start_toxiproxy() as client:
        client.add_proxy(
            name="aiperf-upstream",
            listen=f"127.0.0.1:{listen_port}",
            upstream=upstream,
        )
        for toxic in toxics:
            client.add_toxic("aiperf-upstream", toxic)
        proxied = f"{parsed.scheme}://127.0.0.1:{listen_port}"
        if parsed.path:
            proxied = f"{proxied}{parsed.path}"
        yield proxied
