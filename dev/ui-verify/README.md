# Operator UI visual verification

Renders the operator web UI, screenshots every page, and asserts the
presentation contracts that past regressions violated. Two modes, and the
difference between them matters.

    npm install playwright && npx playwright install chromium

## Mode 1 (preferred): drive the real results-server

The results-server mounts the UI at `/` and the API at `/api/v1/*`
(`operator/results_server.py:237-239`), so one process serves the whole stack
from source — no cluster, image build, or deploy:

    .venv/bin/python -c "import uvicorn; from pathlib import Path; \
      from aiperf.operator.results_server import create_app; \
      uvicorn.run(create_app(results_dir=Path('artifacts/ui-verify/pvc')), \
                  host='127.0.0.1', port=8098)"

    NAMESPACE=<namespace> node shoot.mjs
                            # BASE defaults to http://127.0.0.1:8098

Get a PVC tree by tarring the sweep dirs out of the operator pod:

    kubectl -n <ns> exec deploy/aiperf-operator -c results-server -- sh -c \
      'cd ${AIPERF_RESULTS_DIR:-/data} && tar czf - <ns>/sweeps/<sweep> <ns>/<sweep>-v*' \
      > pvc.tgz

**Use this mode.** It is the only one that can catch a wrong request.

## Mode 2 (offline): replay captured fixtures

    python3 capture_fixtures.py --base http://127.0.0.1:8098 \
        --namespace <ns> --sweep <sweep> > fixtures/api.json
    node serve.mjs                                  # http://127.0.0.1:8099
    BASE=http://127.0.0.1:8099 NAMESPACE=<namespace> node shoot.mjs

`serve.mjs` answers unmatched paths with an empty-but-valid body so the SPA
does not render an error card. That convenience is also this mode's blind spot:
**a page requesting the WRONG URL looks identical to one requesting the right
URL.** A real bug — every sweep fetching the *previous* sweep's epoch and
404ing, so the artifacts card claimed "No aggregate artifacts available" for
sweeps that had them — survived a full review cycle behind exactly that.

Fixtures also starve any page whose endpoints you did not capture: the
leaderboard renders 927 chars of empty state under replay versus 5624 against
the real server. That is why every page now carries a "did it render" check.

## Lessons encoded here

Each of these cost a real defect:

- **Assert positive content, not absence.** A full set of green assertions once
  passed against a winner card reading "No completed variation has a finite
  Output tok/s value yet". Every check was "the bad string is absent", which an
  empty page satisfies trivially.
- **Query the element, not body text.** Body text cannot distinguish "the
  planner cell id appears somewhere" (intended — it is kept as a subtitle
  because it is the artifact path) from "the cell id IS the headline" (the
  defect). Use `data-testid`.
- **Every page gets at least one render check.** Without them the run printed
  ALL CHECKS PASSED while two pages showed only their empty states, so the
  commits touching those pages had no verification at all.
- **Do not trust `fullPage: true`.** The app renders inside an inner scroll
  container, so `document.scrollHeight === window.innerHeight` and Playwright
  images only the first screen. Two pages with visibly different tables
  produced byte-identical PNGs. This harness uses a tall viewport instead.
- **Capture the per-child calls.** `/api/v1/jobs/{ns}/{child}` populates
  metrics and the per-epoch artifact listings populate the artifacts card.
  Omit them and the page renders structurally fine and completely empty.

## Notes

`compare` deliberately has no content check: it renders only a job selector
until two jobs are chosen, so an "empty" compare page is correct.

`NO_PROXY=127.0.0.1,localhost` is required where `HTTP_PROXY` is set, or
localhost requests are routed through the proxy and fail.

The UI loads its vendored Preact, HTM, Signals, and Chart.js runtime from the
same origin. The fixture server must therefore serve `.mjs` files as
`text/javascript`; `serve.mjs` includes that mapping.

Screenshots and their report default to gitignored `artifacts/ui-verify/shots`.
Set `SHOT_DIR` to use another output directory.
