<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
SPDX-License-Identifier: Apache-2.0
-->

# Benchmark Compose environments

## Purpose

This record defines the supported Docker Compose environment for native
benchmark tasks. It is a deliberately narrow sidecar facility: AIPerf owns the
agent `main` service and task authors may provide validated sidecar services.
It is not general Compose passthrough.

## Task contract

A standard schema-`1.0` task always provides `environment/Dockerfile`. Compose
is selected only by the exact additional file
`environment/docker-compose.yaml`; alternate Compose filenames are invalid.
AIPerf builds the Dockerfile and renders a private base Compose file for
`main`. The overlay may omit `main` or declare only its `depends_on` mapping.
It cannot replace the image, command, entrypoint, workdir, user, environment,
mounts, resources, labels, networks, healthcheck, or restart policy of `main`.
Before a project is started, AIPerf asks Compose for canonical JSON with
interpolation and env-file resolution disabled, and compares the result with
the generated-main authority.

Sidecars use a strict literal subset: image or local build, command,
entrypoint, workdir, user, environment, dependency names, healthcheck,
`expose`, project-owned volumes, read-only filesystems, tmpfs, init,
stop-grace period, non-reserved labels, and CPU/memory limits. The import path
rejects unknown fields, interpolation, bind mounts, ports, custom or external
networks and volumes, host namespaces, privileged mode, devices, secrets,
configs, env files, profiles, scaling, restart policies, remote builds, and
Compose top-level passthrough.

The effective environment network must be `public`. `no-network`, allowlists,
phase network transitions, and host-facing Compose networking are not
supported for a Compose task. The local sandbox rejects Compose before it
starts a process.

## Lifecycle and evidence

Preflight validates the owned source snapshot and the read-only canonical
Compose configuration before any build, pull, create, or `up` operation. A
project has a unique task-owned identity and AIPerf removes its labelled
containers, networks, and volumes at every terminal boundary, including
startup, agent, collection, verifier, and timeout failures.

The generated `main` runs the task healthcheck and agent phase on the project
network. Its Dockerfile image, workspace, user, public environment, secrets,
resources, and labels remain runtime-owned. A collection hook is an explicit
nonempty argv vector; AIPerf never supplies an implicit shell. Hooks default to
60 seconds and the combined collection window defaults to 120 seconds.

Strings and artifacts without a service refer to `main`. A sidecar artifact or
hook requires Compose and a separate verifier, and it is valid only on the
final effective step of an explicit multi-step task. At final collection,
AIPerf runs main hooks and captures main evidence, stops `main`, then runs
sidecar hooks and captures sidecar evidence. The separate verifier receives
only the declared frozen artifact transfer. It never joins the task's Compose
network or receives the mutable agent workspace, sidecar filesystem, or agent
secrets.

Compose plan material, sorted services, service-qualified artifacts, ordered
hooks, effective lifecycle timeouts, and the complete `environment/` tree are
part of native package identity. The generated private override and secret
values are deliberately excluded.

## Verification

Unit and fake-provider coverage enforce import policy, generated-main
authority, preflight ordering, service evidence ordering, deadline handling,
and labelled cleanup. Ignored serial product tests drive a freshly built
`aiperf` binary against Docker Compose and prove health-gated sidecar DNS,
non-root agent execution, secret and verifier isolation, sidecar evidence
transfer, and exact cleanup inventory restoration.
