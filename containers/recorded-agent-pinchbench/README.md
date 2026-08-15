# Recorded-agent PinchBench image

This is the neutral build context used by the recorded-agent PinchBench recipe.
The recipe mounts a digest-addressed task workspace at `/workspace` and executes
each recorded command with `bash -lc` and network disabled.

Builds must supply the digest-pinned `BASE_IMAGE` recorded by the workload
fixture; `BASE_IMAGE` defaults only to make local Dockerfile inspection usable.
The image tag `aiperf-recorded-agent-pinchbench:v1` is operational metadata;
the fixture's build-context and OCI digests establish replay comparability.
