# Graham Review — Native Baseten Outcome Fidelity

Reviewed range: `106019c5a1..ccb8c27c14`

## Findings

### [P1] Test-map and design prose overstate the missing-value fixture

The finding and design claim that a historical schema without outcome columns
produces no outcome metadata. The implemented fixture always declares
`duration_e2e_ms`, `duration_ttft_ms`, and `cached_tokens_reference`; the test
sets the latter two to null and retains E2E as `Some(0.0)`. This is useful
coverage, but it is not the evidence the records claim. Correct the prose to
describe the actual null-option behavior.

### [P2] Do not derive `Default` for a fixture whose core fields are required

`FixtureRow` contains required timestamp, prompt, token, session, and E2E
fields. A derived default makes omissions compile as empty or zero data and can
hide malformed future tests. Spell the two new optional fields explicitly in
existing fixture rows instead.

## Hot-path audit

The production change adds three scalar options and a straight-line projection.
It adds no synchronization, allocation beyond the existing turn, clones,
logging, tasks, channels, wall-clock calls, or request-path parsing. Recorded
outcomes remain outside `extra_body`, endpoint parameters, scheduling metadata,
and token accounting. No additional production finding.
