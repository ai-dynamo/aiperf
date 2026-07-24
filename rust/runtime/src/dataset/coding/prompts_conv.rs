// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! User-prompt, context, and coding-conversation renderers.
//!
//! This module also hosts the shared conversation scaffolding used by the two
//! conversation renderers (`conversations` and `conversations_advanced`): the
//! natural-language bridge/request pools, per-conversation identifiers,
//! placeholder-preserving fillers, and 18-way conversation dispatch.

use super::CodingCorpusError;
use super::templates::TemplateRenderer;
use super::vocab::*;
use super::{conversations, conversations_advanced};

// -- Natural-language pools --

/// `_USER_REQUESTS`: natural-language coding request templates.
const USER_REQUESTS: &[&str] = &[
    "Fix the failing test in {module} — it returns {error}",
    "Add retry logic to {cls}.{method}() with exponential backoff",
    "Refactor the {method} function to use async/await instead of callbacks",
    "The {cls} class is throwing {error} when {var} is None",
    "Add input validation for the {var} parameter in {method}()",
    "Write unit tests for {cls}.{method}() covering edge cases",
    "Optimize the {method} query — it's taking too long with large datasets",
    "Add logging to {cls} so we can debug {error} in production",
    "Move the {method} logic from {module} to a shared utility",
    "Implement caching for {cls}.{method}() to reduce database load",
    "Update the {module} config to support environment variable overrides",
    "Add a health check endpoint that verifies {cls} connectivity",
    "The CI is failing because {module} import is broken after the refactor",
    "Create a migration script for the {var} schema change",
    "Add rate limiting to the {method} endpoint — we're getting hammered",
    "Debug why {cls}.{method}() returns stale data after {method}()",
    "Add pagination support to the {method}() response",
    "Implement graceful shutdown for the {cls} worker pool",
    "The {module} integration test is flaky — fix the race condition",
    "Add type hints to all public methods in {cls}",
    "Refactor {module} to use dependency injection instead of globals",
    "Add metrics collection for {method}() latency and error rates",
    "Fix the memory leak in {cls} — it's not releasing {var} properly",
    "Implement {method} fallback when the primary {module} is unavailable",
    "Add request/response logging middleware for the {module} API",
    "Write a load test for {cls}.{method}() with concurrent connections",
    "Add circuit breaker pattern to {cls} for external API calls",
    "The {cls}.{method}() docstring is wrong — update it to match the code",
    "Implement batch processing for {method}() to handle bulk {var} updates",
    "Add WebSocket support to {cls} for real-time {var} updates",
    "Migrate {cls}.{method}() from sync to async — it's called in 3 places across {module} and needs backward compat",
    "Split the {cls} class into two: one for {method} and one for the {var} lifecycle management",
    "We need to add {method}() to {cls}, then wire it into the {module} pipeline and add an integration test",
    "Extract the {method} logic from {cls} into a standalone service, update all callers, and add a deprecation warning to the old path",
    "Rewrite the {module} retry logic: replace the sleep loop with a proper backoff strategy using {cls}",
    "Getting {error} after upgrading {module} to the latest version — only happens under load",
    "The {cls}.{method}() call started returning {error} after we merged the {var} migration PR",
    "Users are reporting {error} intermittently — the {module} logs show {var} is sometimes null",
    "After deploying the {method} change, we see {error} on about 5%% of requests to {cls}",
    "The staging environment throws {error} but prod is fine — suspect it's the {var} config difference",
    "Look at {module}/{cls}.{method}() — the {var} parameter is never validated before being passed to the database layer",
    "In the {module} service, the {method}() function at line ~200 has a subtle bug with {var} boundary handling",
    "The {cls} constructor in {module} initializes {var} too early — move it to the {method}() call site",
    "Add {method}() to {cls} without breaking the existing API contract — we have downstream consumers",
    "Optimize {cls}.{method}() for the case where {var} has over 10K entries, but keep the simple path fast too",
    "Fix the {error} in {module} — but don't change the public interface, we're in a code freeze for other modules",
    "Add telemetry to {cls}.{method}() without adding any new dependencies to the {module} package",
    "We profiled the {method} endpoint and {var} is growing unbounded in {cls}. We need to add eviction or cap the size. The 99th percentile latency spiked 3x last week.",
    "The {cls} pool keeps hitting {error} during peak hours. We scaled horizontally but the issue persists. I think {method}() is holding a lock too long.",
    "After the last {module} refactor, {cls}.{method}() no longer returns deterministic results. The old tests still pass but the integration tests are flaky. Might be a race condition on {var}.",
    "We're moving from REST to gRPC for the {module} service. Start by converting {cls}.{method}() — it's the most latency-sensitive endpoint. Keep the REST handler as a thin adapter for backward compat.",
    "Can you review the {cls}.{method}() implementation? I think the error handling around {var} is wrong",
    "Why does {cls} create a new {var} on every call to {method}()? Seems wasteful",
    "Walk me through the {method}() flow in {module} — I need to understand where {var} gets validated",
    "Is there a reason {cls}.{method}() catches Exception instead of the specific {error}?",
    "Add a Dockerfile for the {module} service that runs {cls} on port 8080 with health checks",
    "The k8s deployment for {module} keeps OOMKilling — add memory limits and check if {cls} leaks during {method}()",
    "Set up a GitHub Action that runs the {module} tests, lints with ruff, and blocks merge on failure",
    "Add Prometheus metrics for {cls}.{method}() — we need p50/p95/p99 latency and error rate by status code",
    "Add a new {var} column to the {module} table with a default value and backfill script",
    "The {cls} serializer is dropping {var} fields when they're empty lists — should preserve them as []",
    "Normalize the {var} schema in {module}: split the nested object into its own table with a foreign key",
];

/// `_BRIDGE_ANALYZE`: read/inspect bridge phrases.
pub(super) const BRIDGE_ANALYZE: &[&str] = &[
    "Let me look at the relevant code.",
    "I'll start by reading the file to understand the current implementation.",
    "Let me search for where this is defined.",
    "First, let me check the existing code.",
    "Let me examine the implementation.",
    "I'll read the source to understand what's happening.",
    "Let me look at the file to see the current state.",
    "I need to understand the existing logic first.",
    "Let me check where {cls} is defined.",
    "I'll look at the {method}() implementation first.",
    "Let me find all the callers of {method}() so we know the impact.",
    "I want to see the full {cls} class before making changes.",
];

/// `_BRIDGE_FIX`: apply-the-fix bridge phrases.
pub(super) const BRIDGE_FIX: &[&str] = &[
    "I can see the issue. Let me fix it.",
    "The problem is in the error handling. Here's the fix:",
    "This needs to be updated. Let me apply the change.",
    "Found it. The logic is incorrect here. Let me correct it.",
    "I see the bug. The condition is inverted. Here's the fix:",
    "The issue is a missing null check. Let me add it.",
    "This needs to be async. Let me update it.",
    "The root cause is a race condition on the shared state. Here's a fix:",
    "I see the problem -- {var} is being mutated after it's shared. Let me fix it.",
    "The issue is that {method}() doesn't account for the empty case. Here's the change:",
    "This is a classic off-by-one. Let me correct the boundary check.",
    "The lock ordering is wrong here. Let me restructure it.",
];

/// `_BRIDGE_TEST`: run-the-tests bridge phrases.
pub(super) const BRIDGE_TEST: &[&str] = &[
    "Let me run the tests to verify.",
    "Now let me check if the tests pass.",
    "Let me verify the fix with the test suite.",
    "Running the tests to confirm the change works.",
    "Let me make sure nothing else broke.",
    "I'll add a test for the new behavior and run the suite.",
    "Let me run just the relevant tests first.",
    "Let me verify with both unit and integration tests.",
];

/// `_BRIDGE_EXPLAIN`: explanation bridge phrases.
pub(super) const BRIDGE_EXPLAIN: &[&str] = &[
    "Here's what's happening in this code:",
    "The flow works like this:",
    "This is structured as follows:",
    "The key parts are:",
    "Let me walk through the logic:",
    "The architecture here is layered -- {cls} delegates to {module} for the heavy lifting.",
    "There are two paths through this code depending on whether {var} is set.",
    "The call chain is: {method}() -> {module}.{method}() -> the underlying store.",
];

/// `_BRIDGE_SUMMARY`: wrap-up bridge phrases.
pub(super) const BRIDGE_SUMMARY: &[&str] = &[
    "The fix adds proper error handling for the {var} case.",
    "I've updated {cls}.{method}() to handle the edge case.",
    "The change ensures {var} is validated before use.",
    "This should resolve the {error} issue. The root cause was missing validation on {var}.",
    "Done. The {method}() call now correctly handles the {var} boundary condition.",
    "Summary: added null check for {var} and updated the return type of {method}().",
    "All tests pass. The change is backward-compatible since {method}() still returns the same type.",
    "Fixed. The {cls} now properly cleans up {var} on both the happy path and the error path.",
    "To summarize: {cls}.{method}() was holding a reference to {var} after the connection closed. The fix moves the cleanup into a finally block.",
];

/// `_BRIDGE_SECURITY`: security-analysis bridge phrases.
pub(super) const BRIDGE_SECURITY: &[&str] = &[
    "This endpoint is vulnerable to SQL injection. The {var} parameter is interpolated directly into the query without sanitization.",
    "The JWT validation is missing the audience claim check. An attacker could use a token issued for a different service.",
    "Let me check the authentication middleware. The RBAC rules should prevent unauthorized access to {method}().",
    "The TLS certificate is using an insecure cipher suite. Let me update the configuration.",
    "I see the issue -- the CORS policy allows wildcard origins, which bypasses the CSRF protection.",
    "The API key is being logged in plaintext. Let me add a secrets filter to the logging configuration.",
    "The password hashing is using MD5. Let me migrate to bcrypt with a proper salt.",
    "Let me verify the OAuth2 authorization code flow. The redirect URI validation looks incomplete.",
];

/// `_BRIDGE_DISTRIBUTED`: distributed-systems bridge phrases.
pub(super) const BRIDGE_DISTRIBUTED: &[&str] = &[
    "The problem is a split-brain scenario. When the network partitions, both nodes think they're the leader.",
    "This needs eventual consistency. Let me add a vector clock to track causal ordering of {var} updates.",
    "The quorum calculation is wrong -- with 5 nodes you need at least 3 for a write quorum, not 2.",
    "Let me add a distributed lock with a TTL to prevent the {method}() race condition across replicas.",
    "The gossip protocol is flooding the network. Let me switch to a pull-based protocol with exponential backoff.",
    "I see the issue -- the Raft log is not being compacted, so leader election takes increasingly long.",
    "The shard rebalancing is not atomic. If it fails midway, some keys become unreachable.",
    "Let me add a read-repair mechanism so stale replicas converge after the partition heals.",
];

/// `_BRIDGE_OBSERVABILITY`: observability bridge phrases.
pub(super) const BRIDGE_OBSERVABILITY: &[&str] = &[
    "The trace spans are not being propagated across the {module} service boundary. Let me add the OpenTelemetry context injection.",
    "I'll add a histogram metric for {method}() latency with buckets at p50/p90/p99 to track the SLO.",
    "The structured logs are missing the correlation_id field, making it impossible to trace requests across services.",
    "Let me set up a Prometheus alert that fires when the error rate exceeds the SLI threshold for 5 minutes.",
    "The dashboard is missing the {cls} service panel. Let me add a Grafana query for the {method} latency distribution.",
    "I see the problem -- the span context is being dropped at the async boundary. Let me propagate it through the task.",
];

/// `_BRIDGE_DATA_ARCHITECTURE`: query/data-layer bridge phrases.
pub(super) const BRIDGE_DATA_ARCHITECTURE: &[&str] = &[
    "The EXPLAIN ANALYZE shows a sequential scan on {var} -- we need a composite index on ({var}, {method}).",
    "This is a classic N+1 query problem. The ORM is issuing a separate SELECT for each {var} in the loop.",
    "Let me batch the {method}() inserts into a single transaction. The current approach holds a lock per row.",
    "The connection pool is exhausted because {cls}.{method}() opens a new connection without releasing it on error.",
    "I'll denormalize the {var} join to avoid the cross-shard query. The read pattern is 100x more frequent than writes.",
    "The transaction isolation level needs to be SERIALIZABLE here to prevent phantom reads on {var}.",
    "Let me add a covering index so the query can be satisfied from the index alone without a table lookup.",
    "The partition key is wrong -- hashing by {var} creates hot spots because the distribution is skewed.",
];

/// `_BRIDGE_ARCHITECTURE_TRADEOFF`: long-form architecture tradeoff analyses.
pub(super) const BRIDGE_ARCHITECTURE_TRADEOFF: &[&str] = &[
    "There are two approaches here. Option A: add a caching layer in front of {cls}.{method}() with a TTL-based invalidation. This gives us sub-millisecond reads but introduces a consistency window where stale {var} can be returned. Option B: use a write-through cache that invalidates on every {method}() call. This maintains consistency but adds latency to writes and complexity to the error handling path. Given the read-heavy workload (100:1 ratio), I'd recommend Option A with a 30-second TTL and a manual invalidation endpoint for critical updates.",
    "The current architecture has {cls} calling {module} synchronously, which blocks the event loop during {method}(). We could switch to a message queue (Redis Streams or Kafka) to decouple the producer and consumer. The tradeoff is that we lose the synchronous error feedback -- if {method}() fails, the caller won't know until it polls for the result. We'd need to add a dead-letter queue and a retry policy with exponential backoff. For this use case, I think the decoupling is worth it because the {method}() latency varies 10x under load.",
    "Looking at this from a security perspective, the {var} field is user-controlled input that flows through {cls}.{method}() into a SQL query. The ORM provides parameterized queries, so SQL injection isn't a risk, but the {var} value is reflected in error messages which could leak internal table names. Additionally, the rate limiter on this endpoint uses a per-IP strategy, but behind a load balancer all requests share the same source IP. We should switch to a per-API-key rate limit and sanitize error responses.",
    "This is a classic CAP theorem tradeoff. The {module} service currently prioritizes consistency (CP) -- if a network partition occurs, the service rejects writes rather than risk divergent state. For the {cls} use case, availability matters more than strict consistency because {method}() is idempotent and clients already handle retries. I'd recommend switching to an AP model with conflict resolution via last-write-wins using the timestamp from the {var} field. We'd need to add a reconciliation job that runs hourly.",
];

/// `_BRIDGE_REFACTOR`: refactor bridge phrases.
pub(super) const BRIDGE_REFACTOR: &[&str] = &[
    "Let me extract this into a separate method for clarity.",
    "I'll restructure {cls} to separate the {method} concern from the lifecycle logic.",
    "The current approach mixes IO with business logic. Let me split them.",
    "I'll move {method}() into its own module since it's used across multiple services.",
    "Let me introduce an interface so we can swap the {module} implementation later.",
    "I'll consolidate the duplicate {method} logic into a shared helper.",
    "The {cls} class is doing too much. Let me split it along the {var}/{method} boundary.",
];

/// `_BRIDGE_PERF`: performance bridge phrases.
pub(super) const BRIDGE_PERF: &[&str] = &[
    "Let me profile {method}() to see where the time goes.",
    "The bottleneck is likely in the {var} allocation. Let me check.",
    "I'll add some timing instrumentation first.",
    "The issue is that {cls} creates a new {var} on every call. Let me add pooling.",
    "Let me check the query plan to see if we're missing an index.",
    "This is doing N+1 queries. Let me batch the {method}() calls.",
    "The {var} is being serialized on every request. Let me cache it.",
    "I see the problem -- {method}() is called inside the lock, blocking all other workers.",
];

/// `_BRIDGE_DEPLOY`: deployment bridge phrases.
pub(super) const BRIDGE_DEPLOY: &[&str] = &[
    "Let me check the deployment configuration.",
    "I'll look at the Dockerfile and the k8s manifests.",
    "Let me verify the environment variables are set correctly.",
    "I'll check the CI pipeline configuration.",
    "Let me look at the health check endpoint.",
    "I see the issue in the resource limits. Let me update the deployment.",
    "The liveness probe is too aggressive. Let me increase the timeout.",
];

/// `_BRIDGE_WRITE_TEST`: test-authoring bridge phrases.
pub(super) const BRIDGE_WRITE_TEST: &[&str] = &[
    "Let me write tests for the new behavior.",
    "I'll add test cases for both the happy path and the error cases.",
    "Let me add a parametrized test to cover all the edge cases.",
    "I'll write an integration test that exercises the full {method}() flow.",
    "Let me add a regression test for this specific bug.",
    "I'll mock the {module} dependency so the test is isolated.",
    "Here's a test that verifies the fix -- it would have caught the original bug.",
];

/// `_FOLLOWUP_QUESTIONS`: user follow-up prompts.
pub(super) const FOLLOWUP_QUESTIONS: &[&str] = &[
    "Can you also add a test for the edge case where {var} is None?",
    "What about the {method} path -- does it need the same fix?",
    "Should we add logging here too?",
    "Can you explain why {cls} uses {var} instead of a local?",
    "Is there a performance concern with this approach?",
    "Should we also update the {method} docstring?",
    "What happens if {var} is empty instead of None?",
    "Can you also check if {method}() handles concurrent access correctly?",
    "Does this need a database migration?",
    "Should we add a feature flag for this change?",
    "What about backward compatibility? The old callers pass {var} as a string.",
    "Can you check if the {module} service needs the same fix?",
    "Is this safe to deploy without a maintenance window?",
    "Can you run the integration tests too?",
    "Looks good. Can you also update the config to increase the default {var}?",
    "One more thing -- can you make {method}() idempotent?",
];

/// `_LANGUAGES`: the language pool driving per-conversation code renderers.
pub(super) const LANGUAGES: &[&str] = &["python", "go", "rust", "typescript"];

/// Per-conversation identifier bundle.
#[derive(Clone, Copy)]
pub(super) struct ConvIds {
    pub cls: &'static str,
    pub module: &'static str,
    pub method: &'static str,
    pub var: &'static str,
    pub error: &'static str,
}

/// Draw shared identifiers reused across a conversation's turns.
pub(super) fn conv_ids(r: &mut TemplateRenderer) -> Result<ConvIds, CodingCorpusError> {
    Ok(ConvIds {
        cls: r.pick(CLASSES)?,
        module: r.pick(MODULES)?,
        method: r.pick(METHODS)?,
        var: r.pick(VARS)?,
        error: r.pick(ERRORS)?,
    })
}

/// Fill the five conversation identifiers,
/// leaving any other `{...}` placeholder untouched.
fn safe_fill(template: &str, ids: &ConvIds) -> String {
    template
        .replace("{cls}", ids.cls)
        .replace("{module}", ids.module)
        .replace("{method}", ids.method)
        .replace("{var}", ids.var)
        .replace("{error}", ids.error)
}

/// Fill all identifiers used by a standalone prompt.
fn fill_all(
    template: &str,
    module: &str,
    cls: &str,
    method: &str,
    var: &str,
    error: &str,
) -> String {
    template
        .replace("{module}", module)
        .replace("{cls}", cls)
        .replace("{method}", method)
        .replace("{var}", var)
        .replace("{error}", error)
}

/// Pick one bridge phrase and fill shared identifiers.
pub(super) fn conv_bridge(
    r: &mut TemplateRenderer,
    pool: &'static [&'static str],
    ids: &ConvIds,
) -> Result<String, CodingCorpusError> {
    let template = r.pick(pool)?;
    Ok(safe_fill(template, ids))
}

/// Pick a user-request template and fill identifiers.
pub(super) fn conv_user_msg(
    r: &mut TemplateRenderer,
    ids: &ConvIds,
) -> Result<String, CodingCorpusError> {
    let template = r.pick(USER_REQUESTS)?;
    Ok(safe_fill(template, ids))
}

/// A single natural-language request, optionally suffixed
/// with a code/error/constraint context block ~30% of the time.
pub(super) fn user_prompt(r: &mut TemplateRenderer) -> Result<String, CodingCorpusError> {
    let template = r.pick(USER_REQUESTS)?;
    let module = r.pick(MODULES)?;
    let cls = r.pick(CLASSES)?;
    let method = r.pick(METHODS)?;
    let var = r.pick(VARS)?;
    let error = r.pick(ERRORS)?;
    // This unused draw is part of the deterministic RNG stream.
    let _type = r.pick(TYPES)?;

    let mut base = fill_all(template, module, cls, method, var, error);
    if r.random() < 0.3 {
        base.push_str("\n\n");
        base.push_str(&prompt_context(r)?);
    }
    Ok(base)
}

/// A short code snippet, error-output block, or a
/// one-line constraint appended to a user prompt.
fn prompt_context(r: &mut TemplateRenderer) -> Result<String, CodingCorpusError> {
    match r.index(3)? {
        0 => {
            let cls = r.pick(CLASSES)?;
            let m1 = r.pick(METHODS)?;
            let v = r.sample(VARS, 2)?;
            let (v1, v2) = (v[0], v[1]);
            let f = r.pick(FILES)?;
            Ok(format!(
                "Here's the relevant code from `{f}`:\n\n\
                 ```\n\
                 class {cls}:\n    \
                 def {m1}(self, {v1}):\n        \
                 {v2} = self._{v1}\n        \
                 return {v2}\n\
                 ```"
            ))
        }
        1 => {
            let err = r.pick(ERRORS)?;
            // Drawn for RNG parity; `cls` is unused in the error-output block.
            let _cls = r.pick(CLASSES)?;
            let m1 = r.pick(METHODS)?;
            let f = r.pick(FILES)?;
            let line = r.number(10, 300)?;
            Ok(format!(
                "Error output:\n\n\
                 ```\n  \
                 File \"{f}\", line {line}, in {m1}\n    \
                 raise RuntimeError(\"{err}\")\n\
                 RuntimeError: {err}\n\
                 ```"
            ))
        }
        _ => {
            let mod_a = r.pick(MODULES)?;
            let mod_b = r.pick(MODULES)?;
            let lat = r.number(5, 50)?;
            let options = [
                "Constraint: no new dependencies allowed in this PR.".to_string(),
                "This is on the hot path — keep allocations minimal.".to_string(),
                "Must remain backward-compatible with the v1 API.".to_string(),
                format!("The {mod_a} service is frozen — only touch {mod_b}."),
                format!("Target is under {lat}ms p99 latency."),
                "We need this for the release on Friday — keep it simple.".to_string(),
            ];
            let idx = r.index(6)?;
            Ok(options[idx].clone())
        }
    }
}

/// Dispatch across all 18 conversation renderers in stable index order.
pub(super) fn coding_conversation(r: &mut TemplateRenderer) -> Result<String, CodingCorpusError> {
    match r.index(18)? {
        0 => conversations::conv_bugfix(r),
        1 => conversations::conv_review(r),
        2 => conversations::conv_feature(r),
        3 => conversations::conv_debug(r),
        4 => conversations::conv_qa(r),
        5 => conversations::conv_refactor(r),
        6 => conversations::conv_perf(r),
        7 => conversations::conv_cicd(r),
        8 => conversations::conv_ml_debug(r),
        9 => conversations::conv_test_write(r),
        10 => conversations_advanced::conv_migration(r),
        11 => conversations_advanced::conv_deploy(r),
        12 => conversations_advanced::conv_security(r),
        13 => conversations_advanced::conv_distributed(r),
        14 => conversations_advanced::conv_observability(r),
        15 => conversations_advanced::conv_db_optimize(r),
        16 => conversations_advanced::conv_architecture_review(r),
        _ => conversations_advanced::conv_incident_response(r),
    }
}
