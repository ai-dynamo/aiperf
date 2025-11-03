<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Container Third-Party Software Attributions

This document provides attribution information for third-party software components included in the aiperf runtime container.

## Software Components

### FFmpeg

**Component Information:**
- **Software**: FFmpeg
- **Version**: 7.1
- **Website**: https://ffmpeg.org/
- **License**: LGPL v2.1+
- **Usage**: Video and audio processing library (included in runtime container)
- **Build Configuration**: Built without GPL components (`--disable-gpl --disable-nonfree --enable-libvpx`)

**License Text:**

> FFmpeg is licensed under the GNU Lesser General Public License (LGPL) version 2.1 or later.
>
> This library is free software; you can redistribute it and/or
> modify it under the terms of the GNU Lesser General Public
> License as published by the Free Software Foundation; either
> version 2.1 of the License, or (at your option) any later version.
>
> This library is distributed in the hope that it will be useful,
> but WITHOUT ANY WARRANTY; without even the implied warranty of
> MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
> Lesser General Public License for more details.
>
> Full license text: https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html

**Source Code Availability:**

The FFmpeg source code used to build this container is available at:
- Official release: https://ffmpeg.org/releases/ffmpeg-7.1.tar.xz
- Our build configuration is documented in the Dockerfile

**Compliance Notes:**

- FFmpeg is dynamically linked and can be replaced by users
- No FFmpeg source code modifications were made
- Build configuration excludes GPL-licensed components
- Apache 2.0 licensed code in this project remains separate from LGPL components

### libvpx

**Component Information:**
- **Software**: libvpx (VP8/VP9 Codec SDK)
- **Version**: 1.12.0 (from Debian Bookworm)
- **Source**: Debian Bookworm
- **Website**: https://www.webmproject.org/
- **License**: BSD 3-Clause
- **Usage**: VP9 video codec library (included in runtime container, used by FFmpeg)

**License Text:**

> Copyright (c) 2010, The WebM Project authors. All rights reserved.
>
> Redistribution and use in source and binary forms, with or without
> modification, are permitted provided that the following conditions are
> met:
>
>   * Redistributions of source code must retain the above copyright
>     notice, this list of conditions and the following disclaimer.
>
>   * Redistributions in binary form must reproduce the above copyright
>     notice, this list of conditions and the following disclaimer in
>     the documentation and/or other materials provided with the
>     distribution.
>
>   * Neither the name of Google, nor the WebM Project, nor the names
>     of its contributors may be used to endorse or promote products
>     derived from this software without specific prior written
>     permission.
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
> "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
> LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
> A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
> HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
> SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
> LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
> DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
> THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
> (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
> OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
>
> Full license text: https://chromium.googlesource.com/webm/libvpx/+/refs/heads/main/LICENSE

**Source Code Availability:**

The libvpx source code is available at:
- Debian package: https://packages.debian.org/bookworm/libvpx-dev
- Debian source package: `apt-get source libvpx` from Debian Bookworm repositories
- Upstream WebM Project source: https://chromium.googlesource.com/webm/libvpx/

**Compliance Notes:**

- libvpx binary is copied from Debian Bookworm base image
- No modifications were made to libvpx source code
- libvpx is dynamically linked with FFmpeg
- BSD license is compatible with Apache 2.0

### Bash

**Component Information:**
- **Software**: GNU Bash (Bourne Again SHell)
- **Version**: 5.2.15
- **Source**: Debian Bookworm
- **Website**: https://www.gnu.org/software/bash/
- **License**: GPL v3+
- **Usage**: Shell interpreter (included in runtime container for interactive use)

**License Text:**

> Bash is free software: you can redistribute it and/or modify
> it under the terms of the GNU General Public License as published by
> the Free Software Foundation, either version 3 of the License, or
> (at your option) any later version.
>
> This program is distributed in the hope that it will be useful,
> but WITHOUT ANY WARRANTY; without even the implied warranty of
> MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
> GNU General Public License for more details.
>
> Full license text: https://www.gnu.org/licenses/gpl-3.0.html

**Source Code Availability:**

The Bash source code is available at:
- Debian package: https://packages.debian.org/bookworm/bash
- Debian source package: `apt-get source bash` from Debian Bookworm repositories
- Upstream GNU source: https://ftp.gnu.org/gnu/bash/

**Compliance Notes:**

- Bash binary is copied from Debian Bookworm base image
- No modifications were made to Bash source code
- Bash is used as a separate executable and does not link with Apache 2.0 code
- Apache 2.0 licensed code in this project remains separate from GPL components

## License Compatibility

This project uses the Apache 2.0 license for its original code. Third-party components included in the runtime container have the following license compatibility considerations:

### FFmpeg (LGPL v2.1+)
LGPL is compatible with Apache 2.0 when:
- FFmpeg is dynamically linked (not statically linked)
- FFmpeg binaries can be replaced by users
- No modifications were made to FFmpeg source code
- Proper attribution is provided (as above)

### libvpx (BSD 3-Clause)
BSD 3-Clause is compatible with Apache 2.0:
- BSD is a permissive license that allows redistribution with minimal restrictions
- Attribution requirements are satisfied through this document
- No conflict with Apache 2.0 terms

### Bash (GPL v3+)
GPL is compatible with Apache 2.0 when:
- Bash runs as a separate executable and is not linked with Apache 2.0 code
- Bash is used as a shell interpreter, not as a library
- No modifications were made to Bash source code
- Proper attribution is provided (as above)

---
*Last updated: November 3, 2025*
