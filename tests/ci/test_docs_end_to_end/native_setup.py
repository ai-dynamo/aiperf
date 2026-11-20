# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Extract a native vLLM launch from the supported documentation setup blocks."""

import shlex
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(slots=True)
class NativeServerSetup:
    image: str
    """Base image that must match the workflow job container."""
    args: tuple[str, ...]
    """Documented arguments to the vLLM OpenAI server."""
    env: dict[str, str]
    """Explicit environment assignments from the server setup."""
    requirements: tuple[str, ...] = ()
    """Python requirements from an inline Dockerfile."""


def compiled_server_constraints(versions: Mapping[str, str]) -> tuple[str, ...]:
    """Preserve the GPU ABI while allowing Python dependencies to be repaired."""
    compiled = {
        "vllm",
        "torch",
        "torchaudio",
        "torchvision",
        "triton",
        "numpy",
        "xformers",
    }
    prefixes = ("nvidia-", "cuda-", "flashinfer", "flash-attn", "vllm-")
    return tuple(
        f"{name}=={version}"
        for name, version in sorted(versions.items())
        if name.lower().replace("_", "-") in compiled
        or name.lower().replace("_", "-").startswith(prefixes)
    )


def parse_native_setup(command: str) -> NativeServerSetup:
    """Translate supported Docker wrappers without executing a Docker command.

    Only image pulls, inline Python dependency installs, and one foreground
    server launch are supported. Other shell or Docker semantics fail closed.
    """
    lines = iter(command.replace("\\\n", " ").splitlines())
    builds: dict[str, tuple[str, tuple[str, ...]]] = {}
    pulled_images: set[str] = set()
    setup: NativeServerSetup | None = None
    for line in lines:
        tokens = shlex.split(line, comments=True)
        if not tokens:
            continue
        if any("$" in token or "`" in token for token in tokens):
            raise ValueError(f"Unsupported shell expansion in server setup: {line}")
        if setup is not None:
            raise ValueError(f"Unsupported command after server launch: {line}")
        if tokens[:2] == ["docker", "pull"] and len(tokens) == 3:
            pulled_images.add(tokens[2])
        elif (
            len(tokens) == 7
            and tokens[:3] == ["docker", "build", "-t"]
            and tokens[4:6] == ["-", "<<"]
        ):
            image: str | None = None
            requirements: list[str] = []
            for instruction in lines:
                if instruction.strip() == tokens[6]:
                    break
                parts = shlex.split(instruction, comments=True)
                if not parts:
                    continue
                if parts[0] == "FROM" and len(parts) == 2 and image is None:
                    image = parts[1]
                elif parts[:3] == ["RUN", "pip", "install"] and len(parts) > 3:
                    if any(
                        p.startswith("-") or p in {"&&", ";", "|"} for p in parts[3:]
                    ):
                        raise ValueError(
                            f"Unsupported dependency install: {instruction}"
                        )
                    requirements.extend(parts[3:])
                else:
                    raise ValueError(
                        f"Unsupported Dockerfile instruction: {instruction}"
                    )
            else:
                raise ValueError("Unsupported unterminated inline Dockerfile")
            if image is None:
                raise ValueError("Unsupported Dockerfile without a base image")
            builds[tokens[3]] = (image, tuple(requirements))
        elif tokens[:2] == ["docker", "run"]:
            setup = _parse_run(tokens[2:], builds)
        else:
            raise ValueError(f"Unsupported server setup command: {line}")
    if setup is None:
        raise ValueError("Unsupported server setup without a foreground Docker run")
    if pulled_images - {setup.image}:
        raise ValueError("Unsupported pull of an image other than the server image")
    if builds and (setup.image, setup.requirements) not in builds.values():
        raise ValueError("Unsupported unused inline Dockerfile")
    return setup


def _parse_run(
    tokens: list[str], builds: dict[str, tuple[str, tuple[str, ...]]]
) -> NativeServerSetup:
    env: dict[str, str] = {}
    index = 0
    while index < len(tokens) and tokens[index].startswith("-"):
        option = tokens[index]
        if option not in {"--gpus", "-p", "--publish", "-e", "--env"}:
            raise ValueError(f"Unsupported Docker run option: {option}")
        if index + 1 >= len(tokens):
            raise ValueError(f"Unsupported Docker run option without a value: {option}")
        value = tokens[index + 1]
        if option == "--gpus" and value != "all":
            raise ValueError(f"Unsupported GPU selection: {value}")
        if option in {"-p", "--publish"}:
            ports = value.split(":")
            if len(ports) != 2 or ports[0] != ports[1] or not ports[0].isdigit():
                raise ValueError(f"Unsupported port mapping: {value}")
        if option in {"-e", "--env"} and "=" in value:
            name, assignment = value.split("=", 1)
            env[name] = assignment
        index += 2
    if index >= len(tokens) or not tokens[index + 1 :]:
        raise ValueError("Unsupported server launch without an image and arguments")
    args = tokens[index + 1 :]
    if any(token in {"&&", "||", ";", "|", "&", ">", "<"} for token in args):
        raise ValueError("Unsupported shell operator in server arguments")
    image, requirements = builds.get(tokens[index], (tokens[index], ()))
    if not image.startswith("vllm/vllm-openai:"):
        raise ValueError(f"Unsupported server image: {image}")
    return NativeServerSetup(image, tuple(args), env, requirements)


if __name__ == "__main__":
    from importlib.metadata import distributions

    print(
        "\n".join(
            compiled_server_constraints(
                {dist.metadata["Name"]: dist.version for dist in distributions()}
            )
        )
    )
