---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: UI Dev Iteration
---

# UI Dev Iteration (in-cluster overlay)

The operator's web UI (`/v1/`) is served as a static bundle from the
`results-server` container. In production the bundle is baked into the
image at install time; on a normal deployment, every UI change therefore
requires `make docker` + `make kube-push` + a `kubectl rollout restart`.

For fast iteration, the chart can mount a writable overlay volume
seeded from the bundled UI on pod boot, and a Makefile target streams
local changes into that overlay. Browser refresh = updated UI. No
image rebuild, no rollout.

## Threat model and scope

This is a **developer-only** mode. The overlay relaxes a property the
production deployment relies on (the UI bundle is immutable and tied to
the image SHA). Do not enable it in production:

- `developer.uiOverride.enabled=true` adds an `emptyDir` volume that is
  writable by the `results-server` container.
- A privilege-escalation in `results-server` could mutate the served UI
  and have it re-serve modified JS to every browser tab connected to
  the operator.
- The override is tied to pod lifetime (emptyDir); pod recreation
  re-seeds the bundled UI from the image, then `make ui-sync` re-applies
  your changes.

## Enable the overlay

Helm value (default `false`):

```yaml
developer:
  uiOverride:
    enabled: true
    # mountPath: /var/aiperf/ui-override   # optional, the default
```

Re-deploy:

```bash
helm upgrade aiperf-operator deploy/helm/aiperf-operator \
  --set developer.uiOverride.enabled=true \
  -n aiperf-system
```

What this wires:

1. An `emptyDir` volume `ui-override` on the operator Pod.
2. An `initContainer` (`ui-seed`) that copies the bundled
   `aiperf/operator/ui/` directory into the volume on every pod
   boot, so the pod works out-of-the-box even before any sync.
3. The volume is mounted writable on the `results-server` container at
   `developer.uiOverride.mountPath` (default `/var/aiperf/ui-override`).
4. The env var `AIPERF_DEV_UI_OVERRIDE_DIR` is set on
   `results-server` to that mount path. `results_server.py` reads the
   env, and falls back to the bundled package path when unset.

## Apply local changes

```bash
make ui-sync                 # default namespace: aiperf-system
NS=aiperf-dev make ui-sync   # override the namespace
DEST=/custom/path make ui-sync   # override the in-pod destination
```

What it does: tar-streams `src/aiperf/operator/ui/` to the
`results-server` container and extracts it into the overlay mount via
the in-container `python -c "import tarfile..."` (works in distroless,
no `tar` binary required on the remote side).

After sync, hard-refresh your browser (Ctrl-Shift-R / Cmd-Shift-R) so
cached `app.js` / `style.css` are not served stale by the browser.

## What gets reset when

| Event                            | Override survives? |
|----------------------------------|--------------------|
| Container restart (same Pod UID) | Yes — emptyDir persists. |
| Pod recreation (rollout, OOM, eviction) | No — emptyDir is fresh, init container re-seeds bundled UI. |
| Helm upgrade with overlay still enabled | Pod recreated → reset. |
| Disabling `developer.uiOverride.enabled` | Volume + initContainer + env removed; UI served from bundle. |

## Not covered

- **Live reload** — the browser must be manually refreshed.
- **Server-side code** (FastAPI routers, Pydantic models) — those live
  in the operator's Python source; changing them still requires a new
  image. Use `make ui-sync` only for static-bundle changes under
  `src/aiperf/operator/ui/`.
- **Multi-replica deployments** — the operator's `replicas` is
  effectively 1 due to leader-election; if you scale it up for some
  reason, `make ui-sync` only writes to one pod.
