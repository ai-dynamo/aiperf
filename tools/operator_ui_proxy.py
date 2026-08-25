# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Serve local operator UI files while proxying operator API traffic."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
from dataclasses import dataclass
from pathlib import Path

from aiohttp import (
    ClientConnectionError,
    ClientSession,
    ClientWebSocketResponse,
    WSMsgType,
    web,
)


@dataclass(slots=True)
class DevReloadState:
    """Mutable dev-reload notification state stored before app startup."""

    condition: asyncio.Condition
    generation: int = 0


_UI_DIR = web.AppKey("ui_dir", Path)
_UPSTREAM = web.AppKey("upstream", str)
_SNAPSHOTS_DIR = web.AppKey("snapshots_dir", Path | None)
_DEV_RELOAD = web.AppKey("dev_reload", bool)
_DEV_RELOAD_STATE = web.AppKey("dev_reload_state", DevReloadState)
_DEV_RELOAD_TASK = web.AppKey("dev_reload_task", asyncio.Task[None])
_DROP_REQUEST_HEADERS = frozenset(
    {"host", "content-length", "connection", "transfer-encoding"}
)
_DROP_RESPONSE_HEADERS = frozenset({"transfer-encoding", "connection"})


def create_app(
    *,
    ui_dir: Path,
    upstream: str,
    snapshots_dir: Path | None = None,
    dev_reload: bool = False,
) -> web.Application:
    """Build the local operator UI proxy app."""
    resolved_ui_dir = ui_dir.resolve()
    if not resolved_ui_dir.is_dir():
        raise FileNotFoundError(f"operator UI directory not found: {resolved_ui_dir}")

    resolved_snapshots_dir = snapshots_dir.resolve() if snapshots_dir else None
    if resolved_snapshots_dir is not None and not resolved_snapshots_dir.is_dir():
        raise FileNotFoundError(
            f"operator UI snapshots directory not found: {resolved_snapshots_dir}"
        )

    app = web.Application()
    app[_UI_DIR] = resolved_ui_dir
    app[_UPSTREAM] = upstream.rstrip("/")
    app[_SNAPSHOTS_DIR] = resolved_snapshots_dir
    app[_DEV_RELOAD] = dev_reload
    app[_DEV_RELOAD_STATE] = DevReloadState(asyncio.Condition())
    if dev_reload:
        app.on_startup.append(_start_dev_reload_watcher)
        app.on_cleanup.append(_cleanup_dev_reload_watcher)
    app.router.add_route("*", "/api/{path:.*}", _proxy_api)
    app.router.add_get("/", _index)
    app.router.add_get("/live/{path:.*}", _serve_live)
    app.router.add_get(
        "/__dev_reload/events",
        _dev_reload_events if dev_reload else _dev_reload_not_found,
    )
    app.router.add_get("/{snapshot}/{path:.*}", _serve_snapshot)
    return app


async def _index(request: web.Request) -> web.Response:
    snapshots_dir = request.app[_SNAPSHOTS_DIR]
    links = ['<li><a href="/live/">live</a></li>']
    if snapshots_dir is not None:
        for child in sorted(path for path in snapshots_dir.iterdir() if path.is_dir()):
            links.append(f'<li><a href="/{child.name}/">{child.name}</a></li>')
    return web.Response(
        text=f"<html><body><h1>Operator UI Proxy</h1><ul>{''.join(links)}</ul></body></html>",
        content_type="text/html",
        headers={"Cache-Control": "no-store"},
    )


async def _serve_live(request: web.Request) -> web.StreamResponse:
    return await _static_response(
        request.app[_UI_DIR],
        request.match_info["path"],
        inject_reload=request.app[_DEV_RELOAD],
    )


async def _serve_snapshot(request: web.Request) -> web.StreamResponse:
    snapshots_dir = request.app[_SNAPSHOTS_DIR]
    if snapshots_dir is None:
        raise web.HTTPNotFound(text="snapshots directory is not configured")
    snapshot_dir = snapshots_dir / request.match_info["snapshot"]
    if not snapshot_dir.is_dir():
        raise web.HTTPNotFound(text="snapshot not found")
    return await _static_response(snapshot_dir, request.match_info["path"])


async def _static_response(
    root: Path,
    raw_path: str,
    *,
    inject_reload: bool = False,
) -> web.StreamResponse:
    relative = Path(raw_path or "index.html")
    candidate = (root / relative).resolve()
    if root not in candidate.parents and candidate != root:
        raise web.HTTPForbidden(text="path escapes static root")
    if candidate.is_dir():
        candidate = candidate / "index.html"
    if not candidate.exists():
        candidate = root / "index.html"
    if not candidate.exists():
        raise web.HTTPNotFound(text=f"index.html not found under {root}")
    if inject_reload and candidate.name == "index.html":
        html = await asyncio.to_thread(candidate.read_text, encoding="utf-8")
        return web.Response(
            text=_inject_dev_reload_script(html),
            content_type="text/html",
            headers={"Cache-Control": "no-store"},
        )
    return web.FileResponse(candidate, headers={"Cache-Control": "no-store"})


def _inject_dev_reload_script(html: str) -> str:
    script = """
<script type="module">
  const events = new EventSource('/__dev_reload/events');
  events.addEventListener('reload', () => window.location.reload());
</script>
""".strip()
    if "/__dev_reload/events" in html:
        return html
    if "</body>" in html:
        return html.replace("</body>", f"{script}</body>", 1)
    return f"{html}\n{script}\n"


async def _dev_reload_not_found(request: web.Request) -> web.Response:
    raise web.HTTPNotFound(text="dev reload is disabled")


async def _notify_dev_reload(app: web.Application) -> None:
    state = app[_DEV_RELOAD_STATE]
    async with state.condition:
        state.generation += 1
        state.condition.notify_all()


async def _dev_reload_events(request: web.Request) -> web.StreamResponse:
    if not request.app[_DEV_RELOAD]:
        raise web.HTTPNotFound(text="dev reload is disabled")
    response = web.StreamResponse(
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-store",
            "Connection": "keep-alive",
        }
    )
    await response.prepare(request)
    state = request.app[_DEV_RELOAD_STATE]
    async with state.condition:
        seen = state.generation
        await state.condition.wait_for(lambda: state.generation > seen)
    await response.write(b"event: reload\n")
    await response.write(b"data: changed\n\n")
    await response.write_eof()
    return response


async def _start_dev_reload_watcher(app: web.Application) -> None:
    app[_DEV_RELOAD_TASK] = asyncio.create_task(_watch_ui_files(app))


async def _cleanup_dev_reload_watcher(app: web.Application) -> None:
    task = app.get(_DEV_RELOAD_TASK)
    if task is None:
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def _watch_ui_files(app: web.Application) -> None:
    root = app[_UI_DIR]
    known_mtimes = await asyncio.to_thread(_collect_ui_file_mtimes, root)
    while True:
        await asyncio.sleep(0.5)
        await _scan_ui_files_once(app, known_mtimes)


async def _scan_ui_files_once(
    app: web.Application,
    known_mtimes: dict[Path, int],
) -> bool:
    current_mtimes = await asyncio.to_thread(_collect_ui_file_mtimes, app[_UI_DIR])
    if current_mtimes == known_mtimes:
        return False
    known_mtimes.clear()
    known_mtimes.update(current_mtimes)
    await _notify_dev_reload(app)
    return True


def _collect_ui_file_mtimes(root: Path) -> dict[Path, int]:
    mtimes: dict[Path, int] = {}
    for dirpath, _, filenames in os.walk(root):
        directory = Path(dirpath)
        for filename in filenames:
            candidate = directory / filename
            if candidate.suffix not in {".html", ".js", ".css"}:
                continue
            with contextlib.suppress(FileNotFoundError):
                mtimes[candidate] = candidate.stat().st_mtime_ns
    return mtimes


async def _proxy_api(request: web.Request) -> web.StreamResponse:
    if request.headers.get("Upgrade", "").lower() == "websocket":
        return await _proxy_websocket(request)

    upstream_url = _upstream_api_url(request)
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in _DROP_REQUEST_HEADERS
    }
    body = await request.read()

    try:
        async with (
            ClientSession(auto_decompress=False) as session,
            session.request(
                request.method, upstream_url, headers=headers, data=body
            ) as response,
        ):
            response_body = await response.read()
            response_headers = {
                key: value
                for key, value in response.headers.items()
                if key.lower() not in _DROP_RESPONSE_HEADERS
            }
            return web.Response(
                body=response_body,
                status=response.status,
                headers=response_headers,
            )
    except ClientConnectionError as exc:
        raise web.HTTPBadGateway(
            text=f"upstream API request failed for {upstream_url}: {exc}"
        ) from exc


async def _proxy_websocket(request: web.Request) -> web.WebSocketResponse:
    client_ws = web.WebSocketResponse()
    await client_ws.prepare(request)

    upstream_url = _websocket_url(_upstream_api_url(request))
    try:
        async with (
            ClientSession() as session,
            session.ws_connect(upstream_url) as upstream_ws,
        ):
            upstream_to_client = asyncio.create_task(_pipe_ws(upstream_ws, client_ws))
            client_to_upstream = asyncio.create_task(_pipe_ws(client_ws, upstream_ws))
            done, pending = await asyncio.wait(
                {upstream_to_client, client_to_upstream},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            for task in done:
                task.result()
    except ClientConnectionError:
        await client_ws.close()
    return client_ws


async def _pipe_ws(
    source: web.WebSocketResponse | ClientWebSocketResponse,
    target: web.WebSocketResponse | ClientWebSocketResponse,
) -> None:
    async for message in source:
        if message.type == WSMsgType.TEXT:
            await target.send_str(message.data)
        elif message.type == WSMsgType.BINARY:
            await target.send_bytes(message.data)
        elif message.type in {WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR}:
            await target.close()
            break


def _upstream_api_url(request: web.Request) -> str:
    path = request.match_info["path"]
    url = f"{request.app[_UPSTREAM]}/api/{path}"
    if request.query_string:
        return f"{url}?{request.query_string}"
    return url


def _websocket_url(url: str) -> str:
    if url.startswith("http://"):
        return "ws://" + url[len("http://") :]
    if url.startswith("https://"):
        return "wss://" + url[len("https://") :]
    return url


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve local operator UI files while proxying API traffic.",
    )
    parser.add_argument("--ui-dir", type=Path, default=Path("src/aiperf/operator/ui"))
    parser.add_argument("--snapshots-dir", type=Path)
    parser.add_argument("--upstream", default="http://127.0.0.1:8081")
    parser.add_argument(
        "--dev-reload",
        action="store_true",
        help="Inject a tiny EventSource reload script and watch UI files for changes.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8123)
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()
    app = create_app(
        ui_dir=args.ui_dir,
        upstream=args.upstream,
        snapshots_dir=args.snapshots_dir,
        dev_reload=args.dev_reload,
    )
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
