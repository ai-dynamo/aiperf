/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/** Foundation `.flow` source, duplicated from `@aiperf/flow-language`'s parser tests. */
export const FOUNDATION_SOURCE = `flow "Request flow" as request-flow {
  language 1
  require core.rect "^1.0.0"
  require core.text "^1.0.0"
  require core.connector "^1.0.0"
  token accent = "#7aa2f7"

  scene "Execution boundary" as execution {
    summary "The CLI starts a runtime that dispatches work."

    rect cli {
      x 40
      y 100
      width 160
      height 72
      fill token(accent)
      label "CLI"
      role "img"
      description "Command-line process"
      fallback "CLI"
    }

    rect runtime {
      x 300
      y 100
      width 180
      height 72
      fill "#244a35"
      label "Runtime"
      role "img"
      description "Execution runtime"
      fallback "Runtime"
    }

    connector spawn {
      from cli
      to runtime
      label "spawn --execute"
      stroke token(accent)
      fallback "CLI starts Runtime"
    }

    camera main {
      at 0 frame cli,runtime zoom 1
      at 2000 frame runtime zoom 1.4
    }

    timeline primary {
      at 0 reveal cli duration 400
      at 800 trace spawn duration 1200
      at 2200 reveal runtime duration 400
    }

    interaction inspect-runtime {
      on select runtime
      do inspect runtime
    }

    responsive compact when width < 720 {
      set runtime.x = 40
      set runtime.y = 240
    }

    narrate "The CLI starts a fresh runtime and dispatches work."
    reading-order cli,runtime,spawn
    fallback "CLI starts Runtime."
  }
}
`;
