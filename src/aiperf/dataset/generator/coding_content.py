# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coding content generator for realistic coding trace replay.

Generates structurally plausible coding content (code, bash output, JSON,
errors, git diffs, CI output, configs, markdown, test output, user prompts)
using template-based generation with random identifiers.

Unlike PromptGenerator which uses Shakespeare as its corpus, this generator
builds two token pools from structural templates:
- text_pool: user prompts (natural language coding requests)
- tool_pool: mixed technical content (code, errors, diffs, configs, etc.)

Generation uses window slicing from pre-built token pools, same as PromptGenerator.
"""

from __future__ import annotations

from aiperf.common import random_generator as rng
from aiperf.common.exceptions import (
    ConfigurationError,
    InvalidStateError,
    NotInitializedError,
)
from aiperf.common.hash_id_random_generator import HashIdRandomGenerator
from aiperf.common.tokenizer import Tokenizer
from aiperf.config.dataset.content import PrefixPromptConfig, PromptConfig
from aiperf.dataset.generator.base import BaseGenerator
from aiperf.dataset.generator.prompt import sample_tokens_from_corpus

# fmt: off
# -- Vocabulary tuples for template fills --

_MODULES = (
    "auth", "cache", "config", "database", "events", "handler", "logger",
    "metrics", "middleware", "pipeline", "processor", "registry", "router",
    "scheduler", "serializer", "service", "storage", "transport", "validator",
    "worker", "adapter", "broker", "collector", "dispatcher", "encoder",
    "factory", "gateway", "indexer", "manager", "monitor", "notifier",
    "observer", "parser", "provider", "queue", "resolver", "scanner",
    "session", "sink", "source", "stream", "transformer", "uploader",
    # web / HTTP
    "api", "webhook", "cors", "oauth", "graphql", "grpc", "websocket",
    "rate_limiter", "proxy", "load_balancer", "reverse_proxy",
    # database / data
    "migration", "schema", "repository", "connection_pool", "query_builder",
    "data_loader", "orm", "replication", "sharding", "backup",
    # ML / data science
    "inference", "tokenizer", "embedding", "feature_store", "model_registry",
    "trainer", "evaluator", "dataset", "sampler", "checkpoint",
    # DevOps / infra
    "deployer", "provisioner", "orchestrator", "health_check", "autoscaler",
    "dns_resolver", "cert_manager", "secret_store", "telemetry", "alerter",
    # security
    "firewall", "encryptor", "key_manager", "audit", "compliance",
    # real libraries / frameworks
    "torch", "numpy", "pandas", "sqlalchemy", "fastapi", "pydantic",
    "celery", "redis", "boto3", "transformers", "datasets", "accelerate",
    "flask", "django", "requests",
)

_CLASSES = (
    "RequestHandler", "DataProcessor", "EventEmitter", "CacheManager",
    "ConnectionPool", "TaskScheduler", "MessageBroker", "StateManager",
    "ConfigLoader", "MetricsCollector", "RateLimiter", "CircuitBreaker",
    "RetryPolicy", "BatchProcessor", "StreamReader", "TokenValidator",
    "SessionStore", "PermissionChecker", "ResourceAllocator", "HealthMonitor",
    "LoadBalancer", "QueueConsumer", "IndexBuilder", "SchemaValidator",
    "PipelineStage", "WorkerPool", "ContextManager", "PluginLoader",
    "TemplateEngine", "SignalHandler", "ProtocolAdapter", "BufferManager",
    "ThrottleController", "RegistryClient", "LockManager", "SnapshotStore",
    "AuditLogger", "FeatureToggle", "MigrationRunner", "DeploymentManager",
    # HTTP layer
    "HttpClient", "RouteResolver", "CorsMiddleware", "AuthMiddleware",
    "ResponseSerializer", "RequestParser", "WebSocketManager", "ApiGateway",
    # data layer
    "QueryExecutor", "TransactionManager", "MigrationEngine", "PoolManager",
    "ReplicaSelector", "ShardRouter", "CursorIterator", "ChangeStream",
    # error types
    "RetryableError", "ValidationError", "TimeoutError", "QuotaExceeded",
    "ConflictError", "NotFoundError", "AuthorizationError", "RateLimitError",
    # ML / inference
    "ModelLoader", "TokenEncoder", "EmbeddingStore", "FeatureExtractor",
    "InferenceEngine", "BatchScheduler", "GradientAccumulator", "Checkpoint",
    # infra / orchestration
    "ServiceMesh", "HealthProbe", "AutoScaler", "SecretProvider",
    "CertRotator", "DnsCache", "TelemetryExporter", "AlertDispatcher",
    # real framework classes
    "Tensor", "DataFrame", "Series", "Session", "Engine", "Router",
    "Pipeline", "Trainer", "Dataset", "DataLoader", "Optimizer",
    "Tokenizer",
)

_METHODS = (
    "process", "handle", "validate", "transform", "execute", "initialize",
    "configure", "dispatch", "resolve", "serialize", "deserialize", "encode",
    "decode", "publish", "subscribe", "notify", "aggregate", "partition",
    "schedule", "allocate", "release", "acquire", "flush", "compress",
    "decompress", "authenticate", "authorize", "revoke", "checkpoint",
    "rollback", "migrate", "replicate", "synchronize", "reconcile",
    "invalidate", "prefetch", "evict", "rebalance", "throttle", "retry",
    "render", "persist", "hydrate", "prune", "drain", "backfill",
    "enqueue", "dequeue", "broadcast", "handshake", "negotiate", "probe",
    "rotate", "shard", "merge", "split", "compact", "snapshot",
    "finalize", "abort", "resume", "suspend", "escalate", "demote",
    "promote", "quarantine", "scrub", "warm_up", "cool_down", "heal",
    "reclaim", "tombstone", "seal", "unseal", "bootstrap", "teardown",
    # real library methods
    "forward", "backward", "train", "evaluate", "predict", "fit",
    "load_state_dict", "save_pretrained", "from_pretrained", "to_dict",
)

_TYPES = (
    "str", "int", "float", "bool", "bytes", "dict", "list", "tuple", "set",
    "None", "Any", "Optional", "Sequence", "Mapping", "Iterator", "Callable",
    "Awaitable", "Coroutine", "AsyncIterator", "Generator", "TypeVar",
    "Protocol", "ClassVar", "Final", "Literal", "Union", "Type",
    "NamedTuple", "TypedDict", "Annotated", "ParamSpec", "Self",
)

_VARS = (
    "result", "data", "config", "context", "payload", "response", "request",
    "buffer", "cursor", "offset", "count", "total", "index", "batch",
    "chunk", "token", "record", "entry", "item", "value", "key", "state",
    "status", "event", "message", "signal", "metric", "timestamp", "duration",
    "timeout", "retries", "threshold", "capacity", "interval", "priority",
    "sequence", "channel", "endpoint", "header", "session", "connection",
    "pipeline", "schema", "trace_id", "tenant_id", "batch_size", "page_size",
    "shard_key", "replica_id", "worker_id", "partition_key", "ttl",
    "max_retries", "backoff", "jitter", "watermark", "checkpoint_id",
    "correlation_id", "span_id", "parent_id", "depth", "fanout",
    "concurrency", "rate", "window", "lag", "drift", "skew",
    "epoch", "generation", "version", "revision", "digest", "nonce",
)

_FILE_PATHS = (
    "src/main.py", "src/config.py", "src/models.py", "src/routes.py",
    "src/utils.py", "src/middleware.py", "src/database.py", "src/auth.py",
    "tests/test_main.py", "tests/test_models.py", "tests/conftest.py",
    "lib/core.go", "lib/handler.go", "lib/service.go", "lib/types.go",
    "pkg/api/server.go", "pkg/api/client.go", "pkg/store/store.go",
    "cmd/server/main.go", "internal/config/config.go",
    "src/lib.rs", "src/main.rs", "src/config.rs", "src/error.rs",
    "src/handler.rs", "src/models.rs", "src/routes.rs",
    "src/index.ts", "src/app.ts", "src/types.ts", "src/api.ts",
    "src/components/App.tsx", "src/components/Form.tsx",
    "Dockerfile", "Makefile", "docker-compose.yml", "pyproject.toml",
    ".github/workflows/ci.yml", "kubernetes/deployment.yaml",
)

_HTTP_ROUTES = (
    "/api/v1/users", "/api/v1/items", "/api/v1/orders", "/api/v1/auth/login",
    "/api/v1/auth/refresh", "/api/v2/search", "/api/v2/analytics",
    "/health", "/ready", "/metrics", "/api/v1/webhooks", "/api/v1/uploads",
    "/api/v1/notifications", "/api/v1/settings", "/api/v1/billing",
    "/api/v1/teams/{team_id}/members", "/api/v1/projects/{project_id}/runs",
    "/api/v1/tenants/{tenant_id}/quota", "/internal/gc", "/internal/debug/pprof",
)

_DB_TABLES = (
    "users", "orders", "items", "sessions", "audit_log", "migrations",
    "api_keys", "rate_limits", "notifications", "webhooks", "tenants",
    "permissions", "invitations", "uploads", "billing_events",
    "job_queue", "dead_letter", "feature_flags", "schema_versions", "locks",
)

_STATUS_CODES = (
    "200 OK", "201 Created", "204 No Content", "301 Moved Permanently",
    "400 Bad Request", "401 Unauthorized", "403 Forbidden", "404 Not Found",
    "409 Conflict", "429 Too Many Requests", "500 Internal Server Error",
    "502 Bad Gateway", "503 Service Unavailable", "504 Gateway Timeout",
)

_LANG_FILE_PATHS: dict[str, tuple[str, ...]] = {
    "python": (
        "src/main.py", "src/config.py", "src/models.py", "src/routes.py",
        "src/utils.py", "src/middleware.py", "src/database.py", "src/auth.py",
        "tests/test_main.py", "tests/test_models.py", "tests/conftest.py",
        "pyproject.toml", "Dockerfile", "Makefile",
        "src/api/v1/endpoints.py", "src/api/v1/schemas.py", "src/api/deps.py",
        "src/core/security.py", "src/core/events.py", "src/services/worker.py",
        "src/repositories/base.py", "tests/integration/test_api.py",
    ),
    "go": (
        "lib/core.go", "lib/handler.go", "lib/service.go", "lib/types.go",
        "pkg/api/server.go", "pkg/api/client.go", "pkg/store/store.go",
        "cmd/server/main.go", "internal/config/config.go",
        "go.mod", "go.sum", "Makefile",
        "internal/middleware/auth.go", "internal/middleware/ratelimit.go",
        "internal/repository/postgres.go", "internal/service/worker.go",
        "pkg/api/middleware.go", "pkg/api/routes.go",
        "internal/telemetry/tracing.go", "internal/health/probe.go",
    ),
    "rust": (
        "src/lib.rs", "src/main.rs", "src/config.rs", "src/error.rs",
        "src/handler.rs", "src/models.rs", "src/routes.rs",
        "Cargo.toml", "Cargo.lock",
        "src/middleware/auth.rs", "src/middleware/tracing.rs",
        "src/repository/mod.rs", "src/repository/postgres.rs",
        "src/service/mod.rs", "src/service/worker.rs",
        "tests/integration/api_test.rs", "benches/throughput.rs",
    ),
    "typescript": (
        "src/index.ts", "src/app.ts", "src/types.ts", "src/api.ts",
        "src/components/App.tsx", "src/components/Form.tsx",
        "src/utils.ts", "src/middleware.ts", "src/routes.ts",
        "package.json", "tsconfig.json", "Dockerfile",
        "src/services/auth.service.ts", "src/services/worker.service.ts",
        "src/middleware/rate-limiter.ts", "src/middleware/error-handler.ts",
        "src/models/user.model.ts", "src/models/order.model.ts",
        "src/repositories/base.repository.ts", "tests/integration/api.test.ts",
    ),
}

_ERROR_MESSAGES = (
    "connection refused", "timeout exceeded", "permission denied",
    "resource not found", "invalid argument", "out of memory",
    "deadlock detected", "rate limit exceeded", "authentication failed",
    "schema validation error", "serialization error", "buffer overflow",
    "index out of range", "null pointer dereference", "type mismatch",
    "missing required field", "duplicate key", "constraint violation",
    "circular dependency detected", "maximum recursion depth exceeded",
    "transaction aborted", "lock timeout after 30s", "quota exceeded",
    "connection pool exhausted", "certificate expired", "DNS resolution failed",
    "checksum mismatch", "payload too large", "stale read",
    "leader election in progress", "shard unavailable", "replica lag exceeded",
    "write conflict detected", "token revoked", "session expired",
    "circuit breaker open", "backpressure applied", "partition offline",
    "consensus timeout", "snapshot corrupted", "migration in progress",
    # GPU / infra errors
    "CUDA out of memory", "NCCL timeout", "connection reset by peer",
    "relation does not exist", "broken pipe", "no route to host",
    "too many open files", "disk quota exceeded",
)

_CLI_COMMANDS = (
    "git status", "git diff HEAD~1", "git log --oneline -10",
    "docker build -t app .", "docker compose up -d",
    "kubectl get pods -n default", "kubectl apply -f deployment.yaml",
    "cargo build --release", "cargo test -- --nocapture",
    "go build ./...", "go test -v ./...", "go vet ./...",
    "npm run build", "npm test", "npx tsc --noEmit",
    "pytest -xvs tests/", "ruff check .", "mypy src/",
    "make build", "make test", "make lint",
    "curl -s http://localhost:8080/health",
    "ps aux | grep python", "top -bn1 | head -20",
    # k8s / infra
    "kubectl describe pod app-7d4b8f-xz9k", "kubectl logs -f deploy/api --tail=100",
    "kubectl rollout status deploy/worker", "kubectl top nodes",
    "helm upgrade --install app ./chart -f values.yaml",
    "terraform plan -out=tfplan", "terraform apply tfplan",
    # redis / data stores
    "redis-cli INFO memory", "redis-cli --latency-history -i 1",
    "pg_dump -Fc mydb > backup.dump", "mongosh --eval 'db.stats()'",
    # perf / profiling
    "perf stat -e cache-misses,cache-references ./bin/server",
    "strace -c -p $(pgrep server)", "valgrind --tool=memcheck ./bin/app",
    "pprof -http=:6060 http://localhost:6060/debug/pprof/heap",
    # load testing
    "wrk -t12 -c400 -d30s http://localhost:8080/api/v1/items",
    "hey -n 10000 -c 100 http://localhost:8080/health",
    "ab -n 5000 -c 50 http://localhost:8080/",
    # misc dev
    "find . -name '*.py' | xargs wc -l | tail -1",
    "du -sh node_modules/ target/ dist/",
    "lsof -i :8080", "ss -tlnp | grep 8080",
    "journalctl -u myapp --since '1 hour ago'",
)

_GO_PACKAGES = (
    "fmt", "os", "io", "net", "http", "context", "sync", "time",
    "strings", "strconv", "encoding/json", "log", "errors", "math",
    "sort", "bytes", "crypto", "regexp", "path/filepath", "database/sql",
    # popular third-party packages
    "github.com/gin-gonic/gin", "go.uber.org/zap",
    "github.com/spf13/viper", "github.com/spf13/cobra",
    "gorm.io/gorm", "google.golang.org/grpc",
    "github.com/prometheus/client_golang/prometheus",
    "github.com/redis/go-redis/v9", "github.com/nats-io/nats.go",
    "github.com/jackc/pgx/v5",
)

_RUST_CRATES = (
    "std::io", "std::fs", "std::collections", "std::sync", "std::fmt",
    "serde", "serde_json", "tokio", "anyhow", "thiserror", "tracing",
    "clap", "reqwest", "axum", "sqlx", "uuid", "chrono", "regex",
    # additional popular crates
    "tower", "hyper", "diesel", "sea_orm", "tonic", "prost",
    "async_trait", "futures",
)

_TS_IMPORTS = (
    "express", "axios", "lodash", "zod", "prisma", "next",
    "react", "react-dom", "typescript", "jest", "vitest",
    "node:fs", "node:path", "node:http", "node:crypto",
    # additional popular packages
    "@nestjs/common", "typeorm", "drizzle-orm", "bullmq",
    "@trpc/server", "ioredis", "pg", "knex",
)

_DECORATORS = (
    "@staticmethod", "@classmethod", "@property", "@abstractmethod",
    "@override", "@cached_property", "@dataclass", "@lru_cache",
    "@pytest.mark.asyncio", "@pytest.mark.parametrize",
    "@app.route", "@app.get", "@app.post", "@router.get",
    # ML framework decorators
    "@torch.no_grad()", "@torch.inference_mode()", "@torch.compile",
    "@torch.jit.script", "@torch.cuda.amp.autocast",
)

_ML_IMPORTS = (
    "torch", "torch.nn", "torch.optim", "torch.utils.data",
    "torch.cuda", "torch.distributed", "torch.amp",
    "transformers", "datasets", "accelerate", "peft",
    "numpy", "safetensors", "wandb", "tensorboard",
    "deepspeed", "bitsandbytes", "trl", "vllm", "triton",
)

_ML_CLASSES = (
    "Linear", "Conv2d", "MultiheadAttention", "LayerNorm", "Embedding",
    "CrossEntropyLoss", "AdamW", "CosineAnnealingLR", "DataLoader",
    "DistributedDataParallel", "AutoModelForCausalLM", "AutoTokenizer",
    "TrainingArguments", "Trainer", "GenerationConfig",
    "BitsAndBytesConfig", "LoraConfig", "PeftModel",
    "StoppingCriteria", "LogitsProcessor",
)

_ML_METHODS = (
    "forward", "backward", "zero_grad", "step", "state_dict",
    "load_state_dict", "save_pretrained", "from_pretrained",
    "generate", "encode", "decode", "batch_decode",
    "to", "cuda", "cpu",
)

_ML_VARS = (
    "logits", "hidden_states", "attention_mask", "input_ids",
    "labels", "loss", "grad_norm", "learning_rate", "num_epochs",
    "batch_size", "max_length", "temperature", "top_p", "top_k",
    "model_name",
)

_MODEL_NAMES = (
    "meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-70B",
    "mistralai/Mixtral-8x7B-v0.1", "mistralai/Mistral-7B-v0.1",
    "google/gemma-2-9b", "Qwen/Qwen2.5-72B",
    "nvidia/Llama-3.1-Nemotron-70B-Instruct",
    "deepseek-ai/DeepSeek-V3", "microsoft/phi-4",
)

_CUDA_ERRORS = (
    "CUDA out of memory. Tried to allocate 2.00 GiB",
    "RuntimeError: Expected all tensors to be on the same device",
    "torch.cuda.OutOfMemoryError: CUDA out of memory",
    "NCCL error: unhandled system error, NCCL version 2.18.5",
    "RuntimeError: NCCL communicator was aborted on rank 0",
    "RuntimeError: cuDNN error: CUDNN_STATUS_NOT_SUPPORTED",
    "RuntimeError: FlashAttention only supports Ampere GPUs or newer",
    "torch.distributed.DistBackendError: NCCL error",
    "RuntimeError: Deterministic behavior was enabled",
    "CUDA error: device-side assert triggered",
)

_USER_REQUESTS = (
    # simple one-liners (original)
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
    # multi-step tasks
    "Migrate {cls}.{method}() from sync to async — it's called in 3 places across {module} and needs backward compat",
    "Split the {cls} class into two: one for {method} and one for the {var} lifecycle management",
    "We need to add {method}() to {cls}, then wire it into the {module} pipeline and add an integration test",
    "Extract the {method} logic from {cls} into a standalone service, update all callers, and add a deprecation warning to the old path",
    "Rewrite the {module} retry logic: replace the sleep loop with a proper backoff strategy using {cls}",
    # error context prompts
    "Getting {error} after upgrading {module} to the latest version — only happens under load",
    "The {cls}.{method}() call started returning {error} after we merged the {var} migration PR",
    "Users are reporting {error} intermittently — the {module} logs show {var} is sometimes null",
    "After deploying the {method} change, we see {error} on about 5%% of requests to {cls}",
    "The staging environment throws {error} but prod is fine — suspect it's the {var} config difference",
    # file path references
    "Look at {module}/{cls}.{method}() — the {var} parameter is never validated before being passed to the database layer",
    "In the {module} service, the {method}() function at line ~200 has a subtle bug with {var} boundary handling",
    "The {cls} constructor in {module} initializes {var} too early — move it to the {method}() call site",
    # constraint-carrying
    "Add {method}() to {cls} without breaking the existing API contract — we have downstream consumers",
    "Optimize {cls}.{method}() for the case where {var} has over 10K entries, but keep the simple path fast too",
    "Fix the {error} in {module} — but don't change the public interface, we're in a code freeze for other modules",
    "Add telemetry to {cls}.{method}() without adding any new dependencies to the {module} package",
    # multi-sentence with background
    "We profiled the {method} endpoint and {var} is growing unbounded in {cls}. We need to add eviction or cap the size. The 99th percentile latency spiked 3x last week.",
    "The {cls} pool keeps hitting {error} during peak hours. We scaled horizontally but the issue persists. I think {method}() is holding a lock too long.",
    "After the last {module} refactor, {cls}.{method}() no longer returns deterministic results. The old tests still pass but the integration tests are flaky. Might be a race condition on {var}.",
    "We're moving from REST to gRPC for the {module} service. Start by converting {cls}.{method}() — it's the most latency-sensitive endpoint. Keep the REST handler as a thin adapter for backward compat.",
    # review / debugging style
    "Can you review the {cls}.{method}() implementation? I think the error handling around {var} is wrong",
    "Why does {cls} create a new {var} on every call to {method}()? Seems wasteful",
    "Walk me through the {method}() flow in {module} — I need to understand where {var} gets validated",
    "Is there a reason {cls}.{method}() catches Exception instead of the specific {error}?",
    # infra / DevOps
    "Add a Dockerfile for the {module} service that runs {cls} on port 8080 with health checks",
    "The k8s deployment for {module} keeps OOMKilling — add memory limits and check if {cls} leaks during {method}()",
    "Set up a GitHub Action that runs the {module} tests, lints with ruff, and blocks merge on failure",
    "Add Prometheus metrics for {cls}.{method}() — we need p50/p95/p99 latency and error rate by status code",
    # data / schema
    "Add a new {var} column to the {module} table with a default value and backfill script",
    "The {cls} serializer is dropping {var} fields when they're empty lists — should preserve them as []",
    "Normalize the {var} schema in {module}: split the nested object into its own table with a foreign key",
)

# -- Bridge text for multi-turn conversations --

_BRIDGE_ANALYZE = (
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
)

_BRIDGE_FIX = (
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
)

_BRIDGE_TEST = (
    "Let me run the tests to verify.",
    "Now let me check if the tests pass.",
    "Let me verify the fix with the test suite.",
    "Running the tests to confirm the change works.",
    "Let me make sure nothing else broke.",
    "I'll add a test for the new behavior and run the suite.",
    "Let me run just the relevant tests first.",
    "Let me verify with both unit and integration tests.",
)

_BRIDGE_EXPLAIN = (
    "Here's what's happening in this code:",
    "The flow works like this:",
    "This is structured as follows:",
    "The key parts are:",
    "Let me walk through the logic:",
    "The architecture here is layered -- {cls} delegates to {module} for the heavy lifting.",
    "There are two paths through this code depending on whether {var} is set.",
    "The call chain is: {method}() -> {module}.{method}() -> the underlying store.",
)

_BRIDGE_SUMMARY = (
    "The fix adds proper error handling for the {var} case.",
    "I've updated {cls}.{method}() to handle the edge case.",
    "The change ensures {var} is validated before use.",
    "This should resolve the {error} issue. The root cause was missing validation on {var}.",
    "Done. The {method}() call now correctly handles the {var} boundary condition.",
    "Summary: added null check for {var} and updated the return type of {method}().",
    "All tests pass. The change is backward-compatible since {method}() still returns the same type.",
    "Fixed. The {cls} now properly cleans up {var} on both the happy path and the error path.",
    "To summarize: {cls}.{method}() was holding a reference to {var} after the connection closed. "
    "The fix moves the cleanup into a finally block.",
)

_BRIDGE_SECURITY = (
    "This endpoint is vulnerable to SQL injection. The {var} parameter is interpolated directly into the query without sanitization.",
    "The JWT validation is missing the audience claim check. An attacker could use a token issued for a different service.",
    "Let me check the authentication middleware. The RBAC rules should prevent unauthorized access to {method}().",
    "The TLS certificate is using an insecure cipher suite. Let me update the configuration.",
    "I see the issue -- the CORS policy allows wildcard origins, which bypasses the CSRF protection.",
    "The API key is being logged in plaintext. Let me add a secrets filter to the logging configuration.",
    "The password hashing is using MD5. Let me migrate to bcrypt with a proper salt.",
    "Let me verify the OAuth2 authorization code flow. The redirect URI validation looks incomplete.",
)

_BRIDGE_DISTRIBUTED = (
    "The problem is a split-brain scenario. When the network partitions, both nodes think they're the leader.",
    "This needs eventual consistency. Let me add a vector clock to track causal ordering of {var} updates.",
    "The quorum calculation is wrong -- with 5 nodes you need at least 3 for a write quorum, not 2.",
    "Let me add a distributed lock with a TTL to prevent the {method}() race condition across replicas.",
    "The gossip protocol is flooding the network. Let me switch to a pull-based protocol with exponential backoff.",
    "I see the issue -- the Raft log is not being compacted, so leader election takes increasingly long.",
    "The shard rebalancing is not atomic. If it fails midway, some keys become unreachable.",
    "Let me add a read-repair mechanism so stale replicas converge after the partition heals.",
)

_BRIDGE_OBSERVABILITY = (
    "The trace spans are not being propagated across the {module} service boundary. Let me add the OpenTelemetry context injection.",
    "I'll add a histogram metric for {method}() latency with buckets at p50/p90/p99 to track the SLO.",
    "The structured logs are missing the correlation_id field, making it impossible to trace requests across services.",
    "Let me set up a Prometheus alert that fires when the error rate exceeds the SLI threshold for 5 minutes.",
    "The dashboard is missing the {cls} service panel. Let me add a Grafana query for the {method} latency distribution.",
    "I see the problem -- the span context is being dropped at the async boundary. Let me propagate it through the task.",
)

_BRIDGE_DATA_ARCHITECTURE = (
    "The EXPLAIN ANALYZE shows a sequential scan on {var} -- we need a composite index on ({var}, {method}).",
    "This is a classic N+1 query problem. The ORM is issuing a separate SELECT for each {var} in the loop.",
    "Let me batch the {method}() inserts into a single transaction. The current approach holds a lock per row.",
    "The connection pool is exhausted because {cls}.{method}() opens a new connection without releasing it on error.",
    "I'll denormalize the {var} join to avoid the cross-shard query. The read pattern is 100x more frequent than writes.",
    "The transaction isolation level needs to be SERIALIZABLE here to prevent phantom reads on {var}.",
    "Let me add a covering index so the query can be satisfied from the index alone without a table lookup.",
    "The partition key is wrong -- hashing by {var} creates hot spots because the distribution is skewed.",
)

_BRIDGE_ARCHITECTURE_TRADEOFF = (
    "There are two approaches here. Option A: add a caching layer in front of {cls}.{method}() with a TTL-based invalidation. "
    "This gives us sub-millisecond reads but introduces a consistency window where stale {var} can be returned. "
    "Option B: use a write-through cache that invalidates on every {method}() call. This maintains consistency but adds "
    "latency to writes and complexity to the error handling path. Given the read-heavy workload (100:1 ratio), "
    "I'd recommend Option A with a 30-second TTL and a manual invalidation endpoint for critical updates.",

    "The current architecture has {cls} calling {module} synchronously, which blocks the event loop during {method}(). "
    "We could switch to a message queue (Redis Streams or Kafka) to decouple the producer and consumer. "
    "The tradeoff is that we lose the synchronous error feedback -- if {method}() fails, the caller won't know until it "
    "polls for the result. We'd need to add a dead-letter queue and a retry policy with exponential backoff. "
    "For this use case, I think the decoupling is worth it because the {method}() latency varies 10x under load.",

    "Looking at this from a security perspective, the {var} field is user-controlled input that flows through "
    "{cls}.{method}() into a SQL query. The ORM provides parameterized queries, so SQL injection isn't a risk, "
    "but the {var} value is reflected in error messages which could leak internal table names. Additionally, "
    "the rate limiter on this endpoint uses a per-IP strategy, but behind a load balancer all requests share "
    "the same source IP. We should switch to a per-API-key rate limit and sanitize error responses.",

    "This is a classic CAP theorem tradeoff. The {module} service currently prioritizes consistency (CP) -- "
    "if a network partition occurs, the service rejects writes rather than risk divergent state. For the {cls} "
    "use case, availability matters more than strict consistency because {method}() is idempotent and clients "
    "already handle retries. I'd recommend switching to an AP model with conflict resolution via last-write-wins "
    "using the timestamp from the {var} field. We'd need to add a reconciliation job that runs hourly.",
)

_BRIDGE_REFACTOR = (
    "Let me extract this into a separate method for clarity.",
    "I'll restructure {cls} to separate the {method} concern from the lifecycle logic.",
    "The current approach mixes IO with business logic. Let me split them.",
    "I'll move {method}() into its own module since it's used across multiple services.",
    "Let me introduce an interface so we can swap the {module} implementation later.",
    "I'll consolidate the duplicate {method} logic into a shared helper.",
    "The {cls} class is doing too much. Let me split it along the {var}/{method} boundary.",
)

_BRIDGE_PERF = (
    "Let me profile {method}() to see where the time goes.",
    "The bottleneck is likely in the {var} allocation. Let me check.",
    "I'll add some timing instrumentation first.",
    "The issue is that {cls} creates a new {var} on every call. Let me add pooling.",
    "Let me check the query plan to see if we're missing an index.",
    "This is doing N+1 queries. Let me batch the {method}() calls.",
    "The {var} is being serialized on every request. Let me cache it.",
    "I see the problem -- {method}() is called inside the lock, blocking all other workers.",
)

_BRIDGE_DEPLOY = (
    "Let me check the deployment configuration.",
    "I'll look at the Dockerfile and the k8s manifests.",
    "Let me verify the environment variables are set correctly.",
    "I'll check the CI pipeline configuration.",
    "Let me look at the health check endpoint.",
    "I see the issue in the resource limits. Let me update the deployment.",
    "The liveness probe is too aggressive. Let me increase the timeout.",
)

_BRIDGE_WRITE_TEST = (
    "Let me write tests for the new behavior.",
    "I'll add test cases for both the happy path and the error cases.",
    "Let me add a parametrized test to cover all the edge cases.",
    "I'll write an integration test that exercises the full {method}() flow.",
    "Let me add a regression test for this specific bug.",
    "I'll mock the {module} dependency so the test is isolated.",
    "Here's a test that verifies the fix -- it would have caught the original bug.",
)

_FOLLOWUP_QUESTIONS = (
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
)

_LANGUAGES = ("python", "go", "rust", "typescript")

_TEXT_POOL_BLOCKS = 200
_BASELINE_POOL_TOKENS = 10_000_000

# Block counts per generator, weighted to reflect AI inference server workloads.
# ML/AI content (~12%) reflects the primary use case of benchmarking LLM inference
# servers, where MoE models route tokens based on content domain. Real library
# names (torch, numpy, etc.) activate correct expert pathways.
# ~28% code, ~11% ML/AI code, ~20% bash/output+training logs, ~11% JSON,
# ~9% errors, ~3% SQL, ~10% other (tool use, diffs, CI, config, docs, tests),
# ~8% user prompts (natural language coding requests)
_TOOL_POOL_BLOCK_COUNTS: dict[str, int] = {
    # Code (~28%)
    "_gen_python_code": 45,
    "_gen_go_code": 45,
    "_gen_rust_code": 45,
    "_gen_typescript_code": 45,
    # ML/AI code (~11%)
    "_gen_ml_training_code": 30,
    "_gen_ml_inference_code": 25,
    "_gen_ml_config": 15,
    # Bash/output + training logs (~20%)
    "_gen_bash_output": 130,
    "_gen_ml_training_log": 20,
    # JSON (~11%)
    "_gen_json_response": 80,
    # Errors (~9%)
    "_gen_error_traceback": 45,
    "_gen_cuda_error": 20,
    # SQL (~3%)
    "_gen_sql_query": 20,
    # User prompts (~6%)
    "_gen_user_prompt": 35,
    # Tool use / diffs / CI / config / docs / tests (~8%)
    "_gen_tool_use_block": 25,
    # Multi-turn conversations (~10%)
    "_gen_coding_conversation": 90,
    "_gen_git_diff": 15,
    "_gen_cicd_output": 15,
    "_gen_config_file": 15,
    "_gen_markdown_doc": 15,
    "_gen_test_output": 15,
}
# fmt: on


class CodingContentGenerator(BaseGenerator):
    """Generator for structurally plausible coding content.

    Builds two pre-tokenized pools from template-based content:
    - text_pool: natural language coding requests (~100K tokens)
    - tool_pool: mixed technical content — code, errors, diffs, etc. (~500K tokens)

    Supports the PromptGenerator-compatible surface used by synthetic composers
    (``generate``, hash_id blocks, prefix pool, shared-system, user-context)
    so ``prompts.corpus=coding`` is a corpus swap, not a narrower generator type.
    """

    def __init__(
        self,
        config: PromptConfig,
        tokenizer: Tokenizer,
        pool_tokens_target: int | None = None,
        prefix_prompts: PrefixPromptConfig | None = None,
        **kwargs,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.prefix_prompts = prefix_prompts
        self._pool_scale = max(
            1.0, (pool_tokens_target or _BASELINE_POOL_TOKENS) / _BASELINE_POOL_TOKENS
        )

        self._template_rng = rng.derive("dataset.coding_content.template")
        self._corpus_rng = rng.derive("dataset.coding_content.corpus")
        self._length_rng = rng.derive("dataset.coding_content.length")
        self._prefix_rng = rng.derive("dataset.coding_content.prefix")

        # Hash-ID-based RNG for deterministic per-hash_id generation.
        # Required by BaseTraceDatasetLoader for parallel conversion.
        self._hash_id_corpus_rng = HashIdRandomGenerator.from_base_rng(self._corpus_rng)

        super().__init__(config=config, tokenizer=tokenizer, **kwargs)

        self._text_pool: list[int] | None = None
        self._tool_pool: list[int] = []
        self._cache: dict[int, list[int]] = {}
        self._decoded_cache: dict[tuple[tuple[int, ...], int, int], str] = {}
        # No stable terminator probe for the coding corpus; segment synthesis
        # falls back to no terminator (matches the empty-list contract in
        # HashIdsSynthesisMixin.bpe_stable_terminator_tokens).
        self._bpe_stable_terminator_tokens: list[int] = []

        self._prefix_prompts: list[str] = []
        self._shared_system_prompt: str | None = None
        self._user_context_prompts: list[str] = []

        self._build_tool_pool()

        # Alias for BaseTraceDatasetLoader compatibility (parallel_convert reads this)
        self._tokenized_corpus = self._tool_pool
        self._corpus_size = len(self._tool_pool)

        pool_size = (
            self.prefix_prompts.pool_size if self.prefix_prompts is not None else None
        ) or 0
        if pool_size > 0:
            self._create_prefix_prompt_pool()

        shared_system_length = (
            self.prefix_prompts.shared_system_length
            if self.prefix_prompts is not None
            else None
        )
        if shared_system_length is not None:
            self._generate_shared_system_prompt()

    def preseed(self, n: int, generator: object) -> None:
        """No-op — CodingContentGenerator draws on demand, no preseed needed."""

    def generate(
        self,
        mean: int | None = None,
        stddev: int | None = None,
        hash_ids: list[int] | None = None,
        block_size: int | None = None,
    ) -> str:
        if hash_ids:
            if mean is None:
                raise ValueError("mean must be provided when hash_ids is set.")
            bs = block_size or self.config.block_size
            return self._generate_cached_prompt(mean, hash_ids, bs)
        num_tokens = self.calculate_num_tokens(mean, stddev)
        return self.generate_prompt(num_tokens)

    def generate_prompt(self, num_tokens: int) -> str:
        tokens = self._sample_tokens(num_tokens, self._tool_pool)
        return self.tokenizer.decode(tokens)

    def calculate_num_tokens(
        self,
        mean: int | None = None,
        stddev: int | None = None,
    ) -> int:
        return self._length_rng.sample_positive_normal_integer(mean, stddev)

    def _create_prefix_prompt_pool(self) -> None:
        """Generate a pool of prefix prompts sampled from the coding corpus."""
        if self.prefix_prompts is None:
            return
        length = self.prefix_prompts.length or 0
        pool_size = self.prefix_prompts.pool_size or 0
        self._prefix_prompts = [self.generate_prompt(length) for _ in range(pool_size)]
        self.debug(
            lambda: f"Initialized coding prefix prompts pool with {len(self._prefix_prompts)} prompts"
        )

    def get_random_prefix_prompt(self) -> str:
        """Fetch a random prefix prompt from the coding corpus pool."""
        if not self._prefix_prompts:
            raise InvalidStateError(
                "Attempted to sample a prefix prompt but the prefix prompts pool is empty. "
                "Please ensure that the prefix prompts pool is initialized."
            )
        return self._prefix_rng.choice(self._prefix_prompts)

    def _generate_shared_system_prompt(self) -> None:
        """Generate the shared system prompt once from the coding corpus."""
        length = (
            self.prefix_prompts.shared_system_length
            if self.prefix_prompts is not None
            else None
        )
        if length is None:
            return
        self._shared_system_prompt = self.generate_prompt(length)
        self.debug(
            lambda: f"Generated coding shared system prompt with {length} tokens"
        )

    def get_shared_system_prompt(self) -> str:
        """Return the shared system prompt sampled from the coding corpus."""
        if self._shared_system_prompt is None:
            raise InvalidStateError(
                "Shared system prompt is not initialized. "
                "Ensure --shared-system-prompt-length is specified."
            )
        return self._shared_system_prompt

    def generate_user_context_prompt(self, session_index: int) -> str:
        """Generate unique user context for ``session_index`` from the coding corpus."""
        length = (
            self.prefix_prompts.user_context_length
            if self.prefix_prompts is not None
            else None
        )
        if length is None:
            raise InvalidStateError(
                "User context prompt length is not configured. "
                "Ensure --user-context-prompt-length is specified."
            )
        while session_index >= len(self._user_context_prompts):
            self._user_context_prompts.append(self.generate_prompt(length))
            self.debug(
                lambda: f"Generated coding user context prompt "
                f"#{len(self._user_context_prompts) - 1}"
            )
        return self._user_context_prompts[session_index]

    def _ensure_text_pool(self) -> list[int]:
        if self._text_pool is None:
            self._build_text_pool()
        assert self._text_pool is not None
        return self._text_pool

    def _build_text_pool(self) -> None:
        blocks: list[str] = []
        for _ in range(int(_TEXT_POOL_BLOCKS * self._pool_scale)):
            blocks.append(self._gen_user_prompt())
        text = "\n\n".join(blocks)
        self._text_pool = self.tokenizer.encode(text)
        pool = self._text_pool
        self.debug(
            lambda: f"Built text pool with {len(pool)} tokens from {len(blocks)} blocks"
        )

    def _build_tool_pool(self) -> None:
        blocks: list[str] = []
        for gen_name, count in _TOOL_POOL_BLOCK_COUNTS.items():
            gen_fn = getattr(self, gen_name)
            for _ in range(int(count * self._pool_scale)):
                blocks.append(gen_fn())
        self._template_rng.shuffle(blocks)
        text = "\n\n".join(blocks)
        self._tool_pool = self.tokenizer.encode(text)
        self.debug(
            lambda: f"Built tool pool with {len(self._tool_pool)} tokens "
            f"from {len(blocks)} blocks"
        )

    def _sample_tokens(self, num_tokens: int, pool: list[int]) -> list[int]:
        if not pool:
            raise NotInitializedError("Token pool is not initialized.")
        pool_size = len(pool)
        if num_tokens <= 0:
            return []
        start_idx = self._corpus_rng.randrange(pool_size)
        end_idx = start_idx + num_tokens
        tokens = pool[start_idx:end_idx]
        if end_idx > pool_size:
            tokens += pool[: end_idx - pool_size]
        return tokens

    def _generate_cached_prompt(
        self,
        num_tokens: int,
        hash_ids: list[int],
        block_size: int,
    ) -> str:
        cache_key = (tuple(hash_ids), num_tokens, block_size)
        if cache_key in self._decoded_cache:
            return self._decoded_cache[cache_key]

        final_prompt = self._build_token_sequence(num_tokens, hash_ids, block_size)
        decoded = self.tokenizer.decode(final_prompt, skip_special_tokens=False)
        self._decoded_cache[cache_key] = decoded
        return decoded

    def _build_token_sequence(
        self,
        num_tokens: int,
        hash_ids: list[int],
        block_size: int,
    ) -> list[int]:
        """Build a token sequence from the tool pool without decoding.

        Mirrors :meth:`PromptGenerator._build_token_sequence` so the two are
        interchangeable when composers swap this generator into trace loaders.
        Supports the same three layouts: exact tile, last-block-partial
        (synthetic prompts), and prefix-only (real captured traces where
        ``hash_ids`` lists only the cached prefix and the un-hashed tail is
        padded with sampled tokens).
        """
        if num_tokens <= 0 or block_size <= 0:
            raise ConfigurationError(
                f"Input length: {num_tokens}, Hash IDs: {hash_ids}, "
                f"Block size: {block_size} are not compatible. num_tokens "
                f"and block_size must both be greater than 0."
            )

        if not hash_ids:
            return self._sample_tokens(num_tokens, self._tool_pool)

        m = len(hash_ids)
        total_hashed = m * block_size
        final_prompt: list[int] = []

        if total_hashed > num_tokens:
            # Synthetic-prompt path: last hash is a partial block.
            final_block_size = num_tokens - ((m - 1) * block_size)
            if final_block_size <= 0 or final_block_size > block_size:
                raise ConfigurationError(
                    f"Input length: {num_tokens}, Hash IDs: {hash_ids}, Block size: {block_size} "
                    f"are not compatible. The final hash block size: {final_block_size} must be "
                    f"greater than 0 and less than or equal to {block_size}."
                )
        else:
            # Exact-tile or prefix-only path: every hash is a full block.
            final_block_size = block_size

        for index, hash_id in enumerate(hash_ids):
            current_block_size = final_block_size if index == m - 1 else block_size

            cached = self._cache.get(hash_id)
            if cached is None:
                self._hash_id_corpus_rng.reseed_for_hash_id(hash_id)
                cached = sample_tokens_from_corpus(
                    self._tool_pool,
                    current_block_size,
                    self._hash_id_corpus_rng,
                    self.tokenizer.block_separation_token_id,
                )
                self._cache[hash_id] = cached
            elif len(cached) != current_block_size:
                # A hash_id identifies a fixed block of content, so it can only
                # ever have one size. The same id at two sizes means a corrupt
                # trace or a block_size that disagrees with the recorded blocks.
                raise ConfigurationError(
                    f"hash_id {hash_id} requested at {current_block_size} tokens "
                    f"but was already materialized at {len(cached)} tokens. A "
                    f"hash_id must map to a single fixed block size; inconsistent "
                    f"sizes indicate a corrupt trace or a --isl-block-size that "
                    f"disagrees with the recorded blocks."
                )

            final_prompt.extend(cached)

        # Prefix-only: pad the un-hashed tail with sampled (uncached) tokens.
        tail = num_tokens - len(final_prompt)
        if tail > 0:
            final_prompt.extend(self._sample_tokens(tail, self._tool_pool))

        return final_prompt

    def _gen_python_code(self) -> str:
        return self._template_rng.choice(
            [
                self._gen_python_class,
                self._gen_python_functions,
                self._gen_python_test,
                self._gen_python_http_handler,
                self._gen_python_data_model,
            ]
        )()

    def _gen_python_class(self) -> str:
        r = self._template_rng
        cls = r.choice(_CLASSES)
        mod = r.choice(_MODULES)
        m1, m2, m3 = r.sample(_METHODS, 3)
        v1, v2, v3 = r.sample(_VARS, 3)
        t1, t2 = r.sample(_TYPES, 2)
        dec = r.choice(_DECORATORS)
        imp_mod = r.choice(_MODULES)
        imp_cls = r.choice(_CLASSES)
        err = r.choice(_ERROR_MESSAGES)

        return f"""\
import {mod}
from {mod}.{imp_mod} import {imp_cls}


class {cls}:
    \"\"\"Handles {m1} operations for {mod}.\"\"\"

    _default_{v3} = 64

    def __init__(self, {v1}: {t1}, {v2}: {t2} = None):
        self._{v1} = {v1}
        self._{v2} = {v2}
        self._{v3} = self._default_{v3}
        self._initialized = False

    {dec}
    async def {m1}(self, {v1}: {t1}) -> {t2}:
        if not self._initialized:
            raise RuntimeError("{cls} not initialized")
        {v2} = await self._{m2}({v1})
        return {v2}

    async def _{m2}(self, {v1}: {t1}) -> {t2}:
        try:
            {v2} = {mod}.{m2}({v1})
            return {v2}
        except Exception as e:
            raise ValueError("{err}") from e

    def {m3}(self) -> None:
        self._initialized = True
        self._{v3} = 0
"""

    def _gen_python_functions(self) -> str:
        r = self._template_rng
        m1, m2, m3 = r.sample(_METHODS, 3)
        v1, v2, v3 = r.sample(_VARS, 3)
        t1, t2, t3 = r.sample(_TYPES, 3)
        mod = r.choice(_MODULES)
        imp_mod = r.choice(_MODULES)
        cls = r.choice(_CLASSES)
        err = r.choice(_ERROR_MESSAGES)

        return f"""\
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from {mod}.{imp_mod} import {cls}

logger = logging.getLogger(__name__)


async def {m1}({v1}: {t1}, {v2}: {t2} | None = None) -> {t3}:
    async with _acquire_{v3}({v1}) as {v3}:
        {v2} = await {cls}().{m2}({v3})
        return [{v2} for _ in range(10) if {v2} is not None]


@asynccontextmanager
async def _acquire_{v3}({v1}: {t1}) -> AsyncIterator[{t2}]:
    {v3} = {mod}.{m3}({v1})
    try:
        yield {v3}
    finally:
        await {v3}.close()


def {m2}_sync({v1}: {t1}, *, max_retries: int = 3) -> {t2}:
    for attempt in range(max_retries):
        try:
            return {mod}.{m2}({v1})
        except RuntimeError:
            if attempt == max_retries - 1:
                raise
            logger.warning("{err}, attempt %d", attempt + 1)
    raise AssertionError("unreachable")
"""

    def _gen_python_test(self) -> str:
        r = self._template_rng
        cls = r.choice(_CLASSES)
        mod = r.choice(_MODULES)
        m1, m2, m3 = r.sample(_METHODS, 3)
        v1, v2 = r.sample(_VARS, 2)
        err = r.choice(_ERROR_MESSAGES)

        return f"""\
import pytest
from unittest.mock import AsyncMock, patch

from {mod} import {cls}


class Test{cls}:
    @pytest.fixture
    def instance(self):
        return {cls}({v1}="test_value")

    @pytest.mark.asyncio
    async def test_{m1}_returns_expected(self, instance):
        instance._{m2} = AsyncMock(return_value=42)
        result = await instance.{m1}()
        assert result == 42
        instance._{m2}.assert_awaited_once()

    @pytest.mark.parametrize("{v1}", ["alpha", "beta", "gamma"])
    def test_{m2}_with_values(self, instance, {v1}):
        instance._{v1} = {v1}
        result = instance.{m2}()
        assert result is not None

    @pytest.mark.asyncio
    async def test_{m3}_raises_on_{v2}(self, instance):
        with pytest.raises(ValueError, match="{err}"):
            await instance.{m3}(None)

    @pytest.mark.asyncio
    async def test_{m1}_with_mock_dependency(self, instance):
        with patch("{mod}.{m2}") as mock:
            mock.return_value = {{{{"key": "{v2}"}}}}\n            result = await instance.{m1}()
            assert "{v2}" in str(result)
"""

    def _gen_python_http_handler(self) -> str:
        r = self._template_rng
        cls = r.choice(_CLASSES)
        mod = r.choice(_MODULES)
        m1, m2 = r.sample(_METHODS, 2)
        v1, v2, v3 = r.sample(_VARS, 3)
        route = r.choice(_HTTP_ROUTES)
        table = r.choice(_DB_TABLES)
        err = r.choice(_ERROR_MESSAGES)

        return f"""\
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from {mod}.{cls.lower()} import {cls}

router = APIRouter(prefix="{route}", tags=["{mod}"])


class {cls}Request(BaseModel):
    {v1}: str = Field(description="Primary {v1} identifier")
    {v2}: int = Field(default=10, ge=1, le=100, description="Page size")
    {v3}: str | None = Field(default=None, description="Optional filter")


class {cls}Response(BaseModel):
    items: list[dict] = Field(description="Result items from {table}")
    total: int = Field(description="Total count")
    page: int = Field(description="Current page number")


@router.post("/", response_model={cls}Response, status_code=201)
async def {m1}(
    body: {cls}Request,
    svc: {cls} = Depends(),
) -> {cls}Response:
    try:
        items = await svc.{m1}(body.{v1}, page_size=body.{v2})
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {cls}Response(items=items, total=len(items), page=1)


@router.get("/{{{{{v1}}}}}")
async def {m2}({v1}: str, svc: {cls} = Depends()) -> dict:
    result = await svc.{m2}({v1})
    if result is None:
        raise HTTPException(status_code=404, detail="{err}")
    return {{"status": "ok", "data": result}}
"""

    def _gen_python_data_model(self) -> str:
        r = self._template_rng
        cls = r.choice(_CLASSES)
        v1, v2, v3, v4 = r.sample(_VARS, 4)
        m1 = r.choice(_METHODS)
        table = r.choice(_DB_TABLES)

        return f"""\
from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class {cls}Status(StrEnum):
    PENDING = "pending"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DELETED = "deleted"


class {cls}Config(BaseModel):
    {v1}: str = Field(description="{cls} {v1} identifier")
    {v2}: int = Field(default=0, ge=0, description="Current {v2} count")
    {v3}: float = Field(default=1.0, gt=0, description="Rate limit for {m1}")
    status: {cls}Status = Field(default={cls}Status.PENDING, description="Lifecycle status")
    {v4}: dict[str, str] = Field(default_factory=dict, description="Arbitrary {v4}")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    source_table: str = Field(default="{table}", description="Backing store table")

    @field_validator("{v1}")
    @classmethod
    def _validate_{v1}(cls, v: str) -> str:
        if not v or len(v) > 256:
            raise ValueError("{v1} must be 1-256 characters")
        return v.strip()

    @field_validator("{v3}")
    @classmethod
    def _validate_{v3}(cls, v: float) -> float:
        if v > 10_000:
            raise ValueError("{v3} exceeds max rate")
        return v

    def {m1}(self) -> bool:
        return self.status == {cls}Status.ACTIVE and self.{v2} > 0
"""

    def _gen_go_code(self) -> str:
        return self._template_rng.choice(
            [
                self._gen_go_struct,
                self._gen_go_http_handler,
                self._gen_go_errors,
                self._gen_go_test,
            ]
        )()

    def _gen_go_struct(self) -> str:
        r = self._template_rng
        pkg1, pkg2 = r.sample(list(_GO_PACKAGES), 2)
        cls = r.choice(_CLASSES)
        m1, m2 = r.sample(_METHODS, 2)
        v1, v2, v3 = r.sample(_VARS, 3)
        pkg_name = r.choice(_MODULES)
        err = r.choice(_ERROR_MESSAGES)

        return f"""\
package {pkg_name}

import (
    "{pkg1}"
    "{pkg2}"
)

type {cls} struct {{{{
    {v1} string `json:"{v1}"`
    {v2} int    `json:"{v2},omitempty"`
    {v3} bool   `json:"-"`
    mu  sync.RWMutex
}}}}

func New{cls}({v1} string) *{cls} {{{{
    return &{cls}{{{{{v1}: {v1}}}}}
}}}}

func (s *{cls}) {m1.title()}(ctx context.Context) error {{{{
    s.mu.Lock()
    defer s.mu.Unlock()
    if s.{v1} == "" {{{{
        return {pkg1}.Errorf("{err}")
    }}}}
    s.{v2}++
    return nil
}}}}

func (s *{cls}) {m2.title()}() (string, error) {{{{
    s.mu.RLock()
    defer s.mu.RUnlock()
    if !s.{v3} {{{{
        return "", {pkg1}.Errorf("%w: not initialized", Err{cls})
    }}}}
    return {pkg2}.Sprintf("%s:%d", s.{v1}, s.{v2}), nil
}}}}
"""

    def _gen_go_http_handler(self) -> str:
        r = self._template_rng
        cls = r.choice(_CLASSES)
        m1, m2 = r.sample(_METHODS, 2)
        v1, v2 = r.sample(_VARS, 2)
        pkg_name = r.choice(_MODULES)
        table = r.choice(_DB_TABLES)
        err = r.choice(_ERROR_MESSAGES)
        status_code = r.choice(
            ["http.StatusOK", "http.StatusCreated", "http.StatusAccepted"]
        )

        return f"""\
package {pkg_name}

import (
    "encoding/json"
    "net/http"
    "log/slog"
)

type {m1.title()}Request struct {{{{
    {v1.title()} string `json:"{v1}" binding:"required"`
    {v2.title()} int    `json:"{v2}" binding:"gte=0"`
}}}}

type {m1.title()}Response struct {{{{
    Items []map[string]any `json:"items"`
    Total int              `json:"total"`
}}}}

func (h *{cls}) {m1.title()}Handler(w http.ResponseWriter, r *http.Request) {{{{
    var req {m1.title()}Request
    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {{{{
        slog.Error("{err}", "handler", "{m1}")
        http.Error(w, err.Error(), http.StatusBadRequest)
        return
    }}}}

    items, err := h.svc.{m2.title()}(r.Context(), req.{v1.title()})
    if err != nil {{{{
        slog.Error("{err}", "table", "{table}")
        http.Error(w, "{err}", http.StatusInternalServerError)
        return
    }}}}

    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader({status_code})
    json.NewEncoder(w).Encode({m1.title()}Response{{{{Items: items, Total: len(items)}}}})
}}}}
"""

    def _gen_go_errors(self) -> str:
        r = self._template_rng
        cls = r.choice(_CLASSES)
        pkg_name = r.choice(_MODULES)
        e1, e2, e3 = r.sample(_ERROR_MESSAGES, 3)
        m1 = r.choice(_METHODS)
        v1 = r.choice(_VARS)

        return f"""\
package {pkg_name}

import (
    "errors"
    "fmt"
)

var (
    Err{cls}         = errors.New("{e1}")
    ErrNot{m1.title()} = errors.New("{e2}")
    ErrInvalid{v1.title()} = errors.New("{e3}")
)

type {cls}Error struct {{{{
    Op      string
    {v1.title()} string
    Err     error
}}}}

func (e *{cls}Error) Error() string {{{{
    return fmt.Sprintf("%s %s: %v", e.Op, e.{v1.title()}, e.Err)
}}}}

func (e *{cls}Error) Unwrap() error {{{{
    return e.Err
}}}}

func Wrap{cls}Error(op, {v1} string, err error) error {{{{
    return &{cls}Error{{{{Op: op, {v1.title()}: {v1}, Err: err}}}}
}}}}
"""

    def _gen_go_test(self) -> str:
        r = self._template_rng
        cls = r.choice(_CLASSES)
        pkg_name = r.choice(_MODULES)
        m1, m2 = r.sample(_METHODS, 2)
        v1, v2 = r.sample(_VARS, 2)

        return f"""\
package {pkg_name}_test

import (
    "context"
    "testing"
)

func Test{cls}_{m1.title()}(t *testing.T) {{{{
    tests := []struct {{{{
        name    string
        {v1}    string
        want    int
        wantErr bool
    }}}}{{{{
        {{{{"valid {v1}", "test_value", 42, false}}}},
        {{{{"empty {v1}", "", 0, true}}}},
        {{{{"long {v1}", "a]very_long_value_that_exceeds_limit", 0, true}}}},
    }}}}

    for _, tt := range tests {{{{
        t.Run(tt.name, func(t *testing.T) {{{{
            s := New{cls}(tt.{v1})
            got, err := s.{m1.title()}(context.Background())
            if (err != nil) != tt.wantErr {{{{
                t.Errorf("{m1.title()}() error = %v, wantErr %v", err, tt.wantErr)
                return
            }}}}
            if got != tt.want {{{{
                t.Errorf("{m1.title()}() = %v, want %v", got, tt.want)
            }}}}
        }}}})
    }}}}
}}}}

func Test{cls}_{m2.title()}_Concurrent(t *testing.T) {{{{
    s := New{cls}("{v2}")
    ctx := context.Background()
    errs := make(chan error, 10)
    for i := 0; i < 10; i++ {{{{
        go func() {{{{ errs <- s.{m2.title()}(ctx) }}}}()
    }}}}
    for i := 0; i < 10; i++ {{{{
        if err := <-errs; err != nil {{{{
            t.Errorf("concurrent {m2}: %v", err)
        }}}}
    }}}}
}}}}
"""

    def _gen_rust_code(self) -> str:
        return self._template_rng.choice(
            [
                self._gen_rust_struct,
                self._gen_rust_http_handler,
                self._gen_rust_errors,
                self._gen_rust_test,
            ]
        )()

    def _gen_rust_struct(self) -> str:
        r = self._template_rng
        cr1, cr2 = r.sample(list(_RUST_CRATES), 2)
        cls = r.choice(_CLASSES)
        m1, m2 = r.sample(_METHODS, 2)
        v1, v2, v3 = r.sample(_VARS, 3)
        err = r.choice(_ERROR_MESSAGES)

        return f"""\
use {cr1};
use {cr2};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct {cls} {{{{
    {v1}: String,
    {v2}: Vec<u8>,
    #[serde(default)]
    {v3}: Option<u64>,
    initialized: bool,
}}}}

impl {cls} {{{{
    pub fn new({v1}: impl Into<String>) -> Self {{{{
        Self {{{{
            {v1}: {v1}.into(),
            {v2}: Vec::new(),
            {v3}: None,
            initialized: false,
        }}}}
    }}}}

    pub async fn {m1}(&mut self) -> Result<(), anyhow::Error> {{{{
        if !self.initialized {{{{
            anyhow::bail!("{err}");
        }}}}
        self.{m2}().await
    }}}}

    async fn {m2}(&self) -> Result<(), anyhow::Error> {{{{
        let _{v2} = self.{v1}.as_bytes();
        tracing::debug!("{m2} completed for {{}}", self.{v1});
        Ok(())
    }}}}
}}}}
"""

    def _gen_rust_http_handler(self) -> str:
        r = self._template_rng
        cls = r.choice(_CLASSES)
        m1, m2 = r.sample(_METHODS, 2)
        v1, v2 = r.sample(_VARS, 2)
        mod = r.choice(_MODULES)

        return f"""\
use axum::{{extract::{{Path, State}}, http::StatusCode, Json}};
use serde::{{Deserialize, Serialize}};
use std::sync::Arc;

use crate::{mod}::{cls};

#[derive(Debug, Deserialize)]
pub struct {m1.title()}Request {{{{
    {v1}: String,
    {v2}: Option<i64>,
}}}}

#[derive(Debug, Serialize)]
pub struct {m1.title()}Response {{{{
    id: String,
    {v1}: String,
    created: bool,
}}}}

pub async fn {m1}_handler(
    State(svc): State<Arc<{cls}>>,
    Json(body): Json<{m1.title()}Request>,
) -> Result<Json<{m1.title()}Response>, StatusCode> {{{{
    let result = svc
        .{m1}(&body.{v1})
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok(Json({m1.title()}Response {{{{
        id: result.id.to_string(),
        {v1}: body.{v1},
        created: true,
    }}}}))
}}}}

pub async fn {m2}_handler(
    State(svc): State<Arc<{cls}>>,
    Path({v1}): Path<String>,
) -> Result<Json<serde_json::Value>, StatusCode> {{{{
    svc.{m2}(&{v1})
        .await
        .map(|v| Json(serde_json::json!({{{{"status": "ok", "data": v}}}})))
        .map_err(|_| StatusCode::NOT_FOUND)
}}}}
"""

    def _gen_rust_errors(self) -> str:
        r = self._template_rng
        cls = r.choice(_CLASSES)
        e1, e2, e3 = r.sample(_ERROR_MESSAGES, 3)
        v1 = r.choice(_VARS)
        mod = r.choice(_MODULES)

        return f"""\
use thiserror::Error;

#[derive(Debug, Error)]
pub enum {cls}Error {{{{
    #[error("{e1}")]
    NotInitialized,

    #[error("{e2}: {{{{{v1}}}}}")]
    InvalidInput {{{{ {v1}: String }}}},

    #[error("{e3}")]
    Internal(#[from] anyhow::Error),

    #[error("io error in {mod}")]
    Io(#[from] std::io::Error),

    #[error("serialization failed")]
    Serde(#[from] serde_json::Error),
}}}}

impl {cls}Error {{{{
    pub fn is_retryable(&self) -> bool {{{{
        matches!(self, Self::Internal(_) | Self::Io(_))
    }}}}

    pub fn status_code(&self) -> u16 {{{{
        match self {{{{
            Self::NotInitialized => 503,
            Self::InvalidInput {{{{ .. }}}} => 400,
            Self::Internal(_) => 500,
            Self::Io(_) => 502,
            Self::Serde(_) => 422,
        }}}}
    }}}}
}}}}
"""

    def _gen_rust_test(self) -> str:
        r = self._template_rng
        cls = r.choice(_CLASSES)
        m1, m2 = r.sample(_METHODS, 2)
        v1, v2 = r.sample(_VARS, 2)
        err = r.choice(_ERROR_MESSAGES)
        cr = r.choice(_RUST_CRATES)

        return f"""\
use {cr};

#[cfg(test)]
mod tests {{{{
    use super::*;

    fn make_{cls.lower()}() -> {cls} {{{{
        {cls}::new("{v1}_test")
    }}}}

    #[tokio::test]
    async fn test_{m1}_success() {{{{
        let mut svc = make_{cls.lower()}();
        svc.initialized = true;
        let result = svc.{m1}().await;
        assert!(result.is_ok(), "expected Ok, got {{:?}}", result);
    }}}}

    #[tokio::test]
    async fn test_{m1}_not_initialized() {{{{
        let mut svc = make_{cls.lower()}();
        let err = svc.{m1}().await.unwrap_err();
        assert!(err.to_string().contains("{err}"));
    }}}}

    #[test]
    fn test_{m2}_returns_bytes() {{{{
        let svc = make_{cls.lower()}();
        let {v2} = svc.{v1}.as_bytes();
        assert!(!{v2}.is_empty());
    }}}}

    #[tokio::test]
    async fn test_{m1}_concurrent() {{{{
        let svc = std::sync::Arc::new(tokio::sync::Mutex::new(make_{cls.lower()}()));
        let mut handles = vec![];
        for _ in 0..5 {{{{
            let svc = svc.clone();
            handles.push(tokio::spawn(async move {{{{
                svc.lock().await.{m1}().await
            }}}}));
        }}}}
        for h in handles {{{{
            let _ = h.await.unwrap();
        }}}}
    }}}}
}}}}
"""

    def _gen_typescript_code(self) -> str:
        return self._template_rng.choice(
            [
                self._gen_typescript_class,
                self._gen_typescript_http_handler,
                self._gen_typescript_types,
                self._gen_typescript_test,
            ]
        )()

    def _gen_typescript_class(self) -> str:
        r = self._template_rng
        imp = r.choice(_TS_IMPORTS)
        imp_cls = r.choice(_CLASSES)
        cls = r.choice(_CLASSES)
        m1, m2, m3 = r.sample(_METHODS, 3)
        v1, v2, v3 = r.sample(_VARS, 3)
        err = r.choice(_ERROR_MESSAGES)

        return f"""\
import {{{{ {imp_cls} }}}} from '{imp}';

interface {cls}Config {{{{
  {v1}: string;
  {v2}?: number;
  timeout: number;
}}}}

export class {cls} {{{{
  #{v1}: string;
  #{v2}: number;
  readonly {v3}: string;

  constructor(config: {cls}Config) {{{{
    this.#{v1} = config.{v1};
    this.#{v2} = config.{v2} ?? 0;
    this.{v3} = crypto.randomUUID();
  }}}}

  async {m1}({v1}: string): Promise<void> {{{{
    try {{{{
      const {v2} = await this.{m2}({v1});
      console.log(`${{{{this.#{v1}}}}}: ${{{{{v2}}}}}`);
    }}}} catch (err) {{{{
      throw new Error(`{err}`);
    }}}}
  }}}}

  async {m3}(): Promise<boolean> {{{{
    return this.#{v2} > 0;
  }}}}

  private async {m2}({v1}: string): Promise<number> {{{{
    return this.#{v2};
  }}}}
}}}}
"""

    def _gen_typescript_http_handler(self) -> str:
        r = self._template_rng
        cls = r.choice(_CLASSES)
        m1, m2 = r.sample(_METHODS, 2)
        v1, v2 = r.sample(_VARS, 2)
        route = r.choice(_HTTP_ROUTES)
        err = r.choice(_ERROR_MESSAGES)

        return f"""\
import {{ Hono }} from 'hono';
import {{ z }} from 'zod';
import {{ {cls} }} from './{cls.lower()}';

const {m1}Schema = z.object({{{{
  {v1}: z.string().min(1).max(256),
  {v2}: z.number().int().positive().optional(),
}}}});

type {m1.title()}Input = z.infer<typeof {m1}Schema>;

const app = new Hono();

app.post('{route}', async (c) => {{{{
  const body = {m1}Schema.safeParse(await c.req.json());
  if (!body.success) {{{{
    return c.json({{{{ error: body.error.flatten() }}}}, 400);
  }}}}

  const svc = new {cls}();
  try {{{{
    const result = await svc.{m1}(body.data.{v1});
    return c.json({{{{ status: 'ok', data: result }}}}, 201);
  }}}} catch (err) {{{{
    return c.json({{{{ error: '{err}' }}}}, 500);
  }}}}
}}}});

app.get('{route}/:id', async (c) => {{{{
  const id = c.req.param('id');
  const svc = new {cls}();
  const item = await svc.{m2}(id);
  if (!item) return c.json({{{{ error: 'not found' }}}}, 404);
  return c.json({{{{ status: 'ok', data: item }}}});
}}}});

export default app;
"""

    def _gen_typescript_types(self) -> str:
        r = self._template_rng
        cls = r.choice(_CLASSES)
        v1, v2, v3 = r.sample(_VARS, 3)
        m1, m2 = r.sample(_METHODS, 2)
        err = r.choice(_ERROR_MESSAGES)

        return f"""\
export type {cls}Status = 'pending' | 'active' | 'failed' | 'completed';

export interface {cls}Event {{{{
  kind: '{m1}' | '{m2}' | 'error';
  {v1}: string;
  timestamp: number;
}}}}

export type {m1.title()}Event = Extract<{cls}Event, {{{{ kind: '{m1}' }}}}>;
export type ErrorEvent = Extract<{cls}Event, {{{{ kind: 'error' }}}}>;

export interface {cls}Config {{{{
  readonly {v1}: string;
  readonly {v2}: number;
  readonly {v3}?: Record<string, unknown>;
}}}}

export type Partial{cls} = Partial<{cls}Config> & Pick<{cls}Config, '{v1}'>;

export function is{cls}Event(e: unknown): e is {cls}Event {{{{
  return (
    typeof e === 'object' &&
    e !== null &&
    'kind' in e &&
    typeof (e as {cls}Event).{v1} === 'string'
  );
}}}}

export function assert{cls}Status(s: string): asserts s is {cls}Status {{{{
  const valid: {cls}Status[] = ['pending', 'active', 'failed', 'completed'];
  if (!valid.includes(s as {cls}Status)) {{{{
    throw new Error(`{err}: ${{{{s}}}}`);
  }}}}
}}}}
"""

    def _gen_typescript_test(self) -> str:
        r = self._template_rng
        cls = r.choice(_CLASSES)
        m1, m2, m3 = r.sample(_METHODS, 3)
        v1, v2 = r.sample(_VARS, 2)
        err = r.choice(_ERROR_MESSAGES)
        mod = r.choice(_MODULES)

        return f"""\
import {{ describe, it, expect, beforeEach, vi }} from 'vitest';
import {{ {cls} }} from '../{mod}';

describe('{cls}', () => {{{{
  let instance: {cls};

  beforeEach(() => {{{{
    instance = new {cls}({{{{ {v1}: 'test', timeout: 5000 }}}});
    vi.clearAllMocks();
  }}}});

  describe('{m1}', () => {{{{
    it('should return expected value', async () => {{{{
      const result = await instance.{m1}('{v2}');
      expect(result).toBeDefined();
      expect(typeof result).toBe('object');
    }}}});

    it('should throw on invalid input', async () => {{{{
      await expect(instance.{m1}('')).rejects.toThrow('{err}');
    }}}});
  }}}});

  describe('{m2}', () => {{{{
    it('should call dependency', async () => {{{{
      const spy = vi.spyOn(instance as any, '{m3}');
      await instance.{m2}('{v1}');
      expect(spy).toHaveBeenCalledOnce();
    }}}});
  }}}});

  it('should handle concurrent calls', async () => {{{{
    const promises = Array.from({{{{ length: 5 }}}}, () => instance.{m1}('{v1}'));
    const results = await Promise.all(promises);
    expect(results).toHaveLength(5);
  }}}});
}}}});
"""

    def _file_pool(self, language: str | None) -> tuple[str, ...]:
        if language:
            return _LANG_FILE_PATHS.get(language, _FILE_PATHS)
        return _FILE_PATHS

    def _gen_tool_use_block(self, language: str | None = None) -> str:
        r = self._template_rng
        return r.choice(
            [
                lambda: self._gen_tool_read(language=language),
                lambda: self._gen_tool_edit(language=language),
                lambda: self._gen_tool_search(language=language),
                lambda: self._gen_tool_bash(language=language),
            ]
        )()

    def _gen_tool_read(self, language: str | None = None) -> str:
        r = self._template_rng
        file_pool = self._file_pool(language)
        f = r.choice(file_pool)
        start_line = r.randint(1, 200)
        cls = r.choice(_CLASSES)
        m1, m2 = r.sample(_METHODS, 2)
        v1, v2 = r.sample(_VARS, 2)
        mod = r.choice(_MODULES)
        err = r.choice(_ERROR_MESSAGES)

        lang_lines: dict[str | None, list[str]] = {
            "python": [
                f"def {m1}(self, {v1}):",
                f"self._{v1} = {v1}",
                f"{v2} = {mod}.{m2}({v1})",
                f"if {v1} is None:",
                f'    raise ValueError("{err}")',
                f"return {v2}",
                f'logger.debug(f"{cls}.{m1}: {{{{{v1}}}}}")',
                "",
            ],
            "go": [
                f"func (s *{cls}) {m1.title()}(ctx context.Context) error {{",
                f"s.{v1} = {v1}",
                f"{v2}, err := s.{m2.title()}(ctx)",
                "if err != nil {",
                f'return fmt.Errorf("{err}: %w", err)',
                "}",
                "return nil",
                "",
            ],
            "rust": [
                f"pub async fn {m1}(&mut self) -> Result<()> {{",
                f"let {v1} = self.{v2}.clone();",
                f"let {v2} = self.{m2}(&{v1}).await?;",
                f"if {v2}.is_empty() {{",
                f'anyhow::bail!("{err}");',
                "}",
                "Ok(())",
                "",
            ],
            "typescript": [
                f"async {m1}({v1}: string): Promise<void> {{",
                f"this.{v1} = {v1};",
                f"const {v2} = await this.{m2}({v1});",
                f"if (!{v2}) {{",
                f"  throw new Error('{err}');",
                "}",
                f"console.log(`{cls}.{m1}: ${{{{{v2}}}}}`);",
                "",
            ],
        }
        code_lines = lang_lines.get(language, lang_lines["python"])

        lines = []
        for i in range(start_line, start_line + r.randint(15, 30)):
            indent = "    " if r.random() > 0.3 else "        "
            line_content = r.choice(code_lines)
            lines.append(f"{i:>6}\t{indent}{line_content}")

        content = "\n".join(lines)
        return f"""\
<tool_name>read</tool_name>
<parameter name="file_path">{f}</parameter>
<result>
{content}
</result>
"""

    def _gen_tool_edit(self, language: str | None = None) -> str:
        r = self._template_rng
        file_pool = self._file_pool(language)
        f = r.choice(file_pool)
        m1, m2 = r.sample(_METHODS, 2)
        v1, v2 = r.sample(_VARS, 2)
        cls = r.choice(_CLASSES)
        err = r.choice(_ERROR_MESSAGES)

        edits: dict[str | None, tuple[str, str]] = {
            "python": (
                f"    def {m1}(self, {v1}):\n        return self._{m2}({v1})",
                f"    async def {m1}(self, {v1}: str) -> dict:\n"
                f"        try:\n"
                f"            {v2} = await self._{m2}({v1})\n"
                f"            if {v2} is None:\n"
                f'                raise ValueError("{err}")\n'
                f'            return {{{{"status": "ok", "data": {v2}}}}}\n'
                f"        except Exception as exc:\n"
                f'            logger.error("{cls}.{m1} failed: %s", exc)\n'
                f"            raise",
            ),
            "go": (
                f"func (s *{cls}) {m1.title()}() error {{{{\n    return nil\n}}}}",
                f"func (s *{cls}) {m1.title()}(ctx context.Context) error {{{{\n"
                f"    {v2}, err := s.{m2.title()}(ctx)\n"
                f"    if err != nil {{{{\n"
                f'        return fmt.Errorf("{err}: %w", err)\n'
                f"    }}}}\n"
                f"    s.{v1} = {v2}\n"
                f"    return nil\n"
                f"}}}}",
            ),
            "rust": (
                f"fn {m1}(&self) -> Result<()> {{{{\n    Ok(())\n}}}}",
                f"async fn {m1}(&mut self) -> Result<()> {{{{\n"
                f"    let {v2} = self.{m2}().await?;\n"
                f'    anyhow::ensure!(!{v2}.is_empty(), "{err}");\n'
                f"    self.{v1} = {v2};\n"
                f"    Ok(())\n"
                f"}}}}",
            ),
            "typescript": (
                f"{m1}({v1}: string) {{{{\n    return this.{m2}({v1});\n}}}}",
                f"async {m1}({v1}: string): Promise<Record<string, unknown>> {{{{\n"
                f"    const {v2} = await this.{m2}({v1});\n"
                f"    if (!{v2}) throw new Error('{err}');\n"
                f"    return {{ status: 'ok', data: {v2} }};\n"
                f"}}}}",
            ),
        }
        old_str, new_str = edits.get(language, edits["python"])

        return f"""\
<tool_name>edit</tool_name>
<parameter name="file_path">{f}</parameter>
<parameter name="old_string">{old_str}</parameter>
<parameter name="new_string">{new_str}</parameter>
<result>
The file {f} has been updated successfully.
</result>
"""

    def _gen_tool_search(self, language: str | None = None) -> str:
        r = self._template_rng
        file_pool = self._file_pool(language)

        lang_patterns: dict[str | None, list[str]] = {
            "python": [
                f"class {r.choice(_CLASSES)}",
                f"def {r.choice(_METHODS)}",
                f"import {r.choice(_MODULES)}",
                f"async def {r.choice(_METHODS)}",
            ],
            "go": [
                f"func {r.choice(_METHODS).title()}",
                f"type {r.choice(_CLASSES)} struct",
                f'"{r.choice(list(_GO_PACKAGES))}"',
                f"func New{r.choice(_CLASSES)}",
            ],
            "rust": [
                f"fn {r.choice(_METHODS)}",
                f"pub struct {r.choice(_CLASSES)}",
                f"use {r.choice(list(_RUST_CRATES))}",
                f"impl {r.choice(_CLASSES)}",
            ],
            "typescript": [
                f"class {r.choice(_CLASSES)}",
                f"export function {r.choice(_METHODS)}",
                f"import {{ {r.choice(_CLASSES)} }}",
                f"interface {r.choice(_CLASSES)}",
            ],
        }
        patterns = lang_patterns.get(language, lang_patterns["python"])
        pattern = r.choice([*patterns, r.choice(_ERROR_MESSAGES)])

        files = r.sample(list(file_pool), min(r.randint(3, 6), len(file_pool)))
        matches = []
        for f in files:
            line_num = r.randint(1, 400)
            ctx = r.choice(_VARS)
            matches.append(f"{f}:{line_num}:    {pattern}({ctx})")

        content = "\n".join(matches)
        return f"""\
<tool_name>search</tool_name>
<parameter name="pattern">{pattern}</parameter>
<result>
{content}
</result>
"""

    def _gen_tool_bash(self, language: str | None = None) -> str:
        r = self._template_rng
        mod = r.choice(_MODULES)
        cls = r.choice(_CLASSES)
        methods = r.sample(list(_METHODS), 4)
        n_pass = r.randint(10, 80)
        n_fail = r.randint(0, 3)
        dur = r.uniform(0.5, 30.0)

        lang_cmds: dict[str | None, str] = {
            "python": "pytest -xvs tests/",
            "go": "go test -v ./...",
            "rust": "cargo test",
            "typescript": "npx vitest run",
        }
        cmd = lang_cmds.get(language, r.choice(_CLI_COMMANDS))

        test_lines = []
        for m in methods:
            passed = r.random() > 0.2
            if language == "go":
                status = "ok" if passed else "FAIL"
                test_lines.append(
                    f"--- {status}: Test{m.title()} ({r.uniform(0.001, 2.0):.3f}s)"
                )
            elif language == "rust":
                status = "ok" if passed else "FAILED"
                test_lines.append(f"test {mod}::{cls.lower()}::test_{m} ... {status}")
            elif language == "typescript":
                mark = "\u2713" if passed else "\u2717"
                test_lines.append(f"  {mark} {cls} > {m} ({r.randint(1, 500)} ms)")
            else:
                status = "PASSED" if passed else "FAILED"
                test_lines.append(f"tests/test_{mod}.py::Test{cls}::test_{m} {status}")
        test_output = "\n".join(test_lines)

        return f"""\
<tool_name>bash</tool_name>
<parameter name="command">{cmd}</parameter>
<result>
{test_output}

{n_pass} passed, {n_fail} failed in {dur:.2f}s
</result>
"""

    def _gen_bash_output(self, language: str | None = None) -> str:
        r = self._template_rng
        return r.choice(
            [
                lambda: self._gen_bash_file_explore(language=language),
                lambda: self._gen_bash_build_test(language=language),
                lambda: self._gen_bash_git_workflow(language=language),
            ]
        )()

    def _gen_bash_file_explore(self, language: str | None = None) -> str:
        r = self._template_rng
        file_pool = self._file_pool(language)
        ext_cmds: dict[str | None, tuple[str, str]] = {
            "python": ("find . -name '*.py'", "src/**/*.py"),
            "go": ("find . -name '*.go'", "**/*.go"),
            "rust": ("find . -name '*.rs'", "src/**/*.rs"),
            "typescript": ("find . -name '*.ts'", "src/**/*.ts"),
        }
        find_cmd, glob_pat = ext_cmds.get(language, ext_cmds["python"])
        cmd = r.choice(("ls -la", find_cmd, "tree src/", "wc -l"))
        files = r.sample(list(file_pool), min(r.randint(4, 8), len(file_pool)))
        file_listing = "\n".join(
            f"  {f:<42} {r.randint(1, 500):>4} lines  {r.randint(1, 50):>3}K"
            for f in files
        )
        total_lines = r.randint(500, 15000)

        return f"""\
$ {cmd}
{file_listing}
$ wc -l {glob_pat} | tail -1
  {total_lines} total
$ du -sh .
  {r.randint(1, 500)}M\t.
"""

    def _gen_bash_build_test(self, language: str | None = None) -> str:
        r = self._template_rng
        mod = r.choice(_MODULES)
        n_pkgs = r.randint(10, 200)
        build_time = r.uniform(0.5, 30.0)
        n_pass = r.randint(20, 150)
        n_fail = r.randint(0, 5)
        test_time = r.uniform(1.0, 60.0)

        lang_build: dict[str | None, tuple[str, str]] = {
            "python": (
                "pip install -e '.[dev]'",
                f"pytest tests/ -x\n  {n_pass} passed, {n_fail} failed in {test_time:.1f}s",
            ),
            "go": (
                f"go build ./cmd/{mod}\n  Compiled {n_pkgs} packages in {build_time:.1f}s",
                f"go test -v -race ./...\n  {n_pass} passed, {n_fail} failed in {test_time:.1f}s",
            ),
            "rust": (
                f"cargo build --release\n  Compiling {n_pkgs} crates\n  Finished in {build_time:.1f}s",
                f"cargo test\n  {n_pass} passed, {n_fail} failed in {test_time:.1f}s",
            ),
            "typescript": (
                f"npm ci && npm run build\n  Resolved {n_pkgs} packages in {build_time:.1f}s",
                f"npx vitest run\n  {n_pass} passed, {n_fail} failed in {test_time:.1f}s",
            ),
        }
        build_cmd, test_cmd = lang_build.get(language, lang_build["python"])

        return f"""\
$ {build_cmd}
$ {test_cmd}
$ echo $?
{"0" if n_fail == 0 else "1"}
"""

    def _gen_bash_git_workflow(self, language: str | None = None) -> str:
        r = self._template_rng
        file_pool = self._file_pool(language)
        branch = f"{r.choice(_MODULES)}/{r.choice(_METHODS)}-{r.choice(_VARS)}"
        mod = r.choice(_MODULES)
        files = r.sample(list(file_pool), min(3, len(file_pool)))
        changed = "\n".join(f"  M {f}" for f in files)
        hash1 = f"{r.randint(1000000, 9999999):07x}"
        hash2 = f"{r.randint(1000000, 9999999):07x}"

        return f"""\
$ git checkout -b {branch}
Switched to a new branch '{branch}'
$ git status
On branch {branch}
Changes not staged for commit:
{changed}
$ git add -A && git commit -m "feat: {r.choice(_METHODS)} {r.choice(_VARS)} in {mod}"
[{branch} {hash1}] feat: {r.choice(_METHODS)} {r.choice(_VARS)} in {mod}
 {len(files)} files changed, {r.randint(10, 200)} insertions(+), {r.randint(1, 50)} deletions(-)
$ git log --oneline -3
{hash1} feat: {r.choice(_METHODS)} {r.choice(_VARS)} in {mod}
{hash2} fix: {r.choice(_ERROR_MESSAGES)}
"""

    def _gen_json_response(self, language: str | None = None) -> str:
        return self._template_rng.choice(
            [
                self._gen_json_object,
                self._gen_json_paginated,
                self._gen_json_error,
            ]
        )()

    def _gen_json_object(self) -> str:
        r = self._template_rng
        m1, m2 = r.sample(_METHODS, 2)
        v1, v2, v3 = r.sample(_VARS, 3)
        cls = r.choice(_CLASSES)
        id_suffix = r.randint(1000, 9999)
        num_val = r.randint(0, 1000)
        float_val = r.uniform(0, 1)
        ts = r.randint(1700000000, 1800000000)
        items = [
            f'      {{{{"id": {r.randint(1, 999)}, "name": "{r.choice(_VARS)}"}}}}'
            for _ in range(3)
        ]
        items_str = ",\n".join(items)

        return f"""\
{{{{
  "status": "ok",
  "data": {{{{
    "{v1}": "{cls.lower()}_{id_suffix}",
    "{v2}": {num_val},
    "{v3}": {float_val:.4f},
    "metadata": {{{{
      "action": "{m1}",
      "source": "{m2}",
      "timestamp": "{ts}"
    }}}},
    "items": [
{items_str}
    ]
  }}}}
}}}}
"""

    def _gen_json_paginated(self) -> str:
        r = self._template_rng
        v1, v2 = r.sample(_VARS, 2)
        cls = r.choice(_CLASSES)
        total = r.randint(50, 5000)
        page = r.randint(1, 20)
        per_page = r.choice([10, 20, 50, 100])
        items = [
            f'    {{{{"id": "{cls.lower()}_{r.randint(1000, 9999)}", "{v1}": "{r.choice(_MODULES)}", "{v2}": {r.randint(0, 100)}}}}}'
            for _ in range(min(per_page, 5))
        ]
        items_str = ",\n".join(items)

        return f"""\
{{{{
  "data": [
{items_str}
  ],
  "pagination": {{{{
    "page": {page},
    "per_page": {per_page},
    "total": {total},
    "total_pages": {(total + per_page - 1) // per_page},
    "has_next": {str(page * per_page < total).lower()},
    "has_prev": {str(page > 1).lower()}
  }}}}
}}}}
"""

    def _gen_json_error(self) -> str:
        r = self._template_rng
        err = r.choice(_ERROR_MESSAGES)
        status = r.choice(_STATUS_CODES)
        code = status.split()[0]
        trace_id = f"{r.randint(100000, 999999):06x}-{r.randint(100000, 999999):06x}"
        v1 = r.choice(_VARS)
        cls = r.choice(_CLASSES)

        return f"""\
{{{{
  "error": {{{{
    "code": {code},
    "status": "{status}",
    "message": "{err}",
    "details": [
      {{{{
        "field": "{v1}",
        "reason": "{err}",
        "type": "{cls}"
      }}}}
    ],
    "trace_id": "{trace_id}",
    "documentation_url": "https://docs.example.com/errors/{code}"
  }}}}
}}}}
"""

    def _gen_error_traceback(self, language: str | None = None) -> str:
        r = self._template_rng
        err = r.choice(_ERROR_MESSAGES)
        cls = r.choice(_CLASSES)
        m1, m2, m3, m4 = r.sample(_METHODS, 4)

        lang_to_kind = {
            "python": "python",
            "go": "go",
            "rust": "rust",
            "typescript": "node",
        }
        kind = (
            lang_to_kind[language]
            if language in lang_to_kind
            else r.choice(["python", "go", "rust", "node"])
        )
        file_pool = self._file_pool(language)
        f1, f2, f3, f4 = r.sample(list(file_pool), 4)
        if kind == "python":
            v = r.choice(_VARS)
            mod = r.choice(_MODULES)
            err2 = r.choice(_ERROR_MESSAGES)
            cls2 = r.choice(_CLASSES)
            return f"""\
Traceback (most recent call last):
  File "{f1}", line {r.randint(10, 500)}, in {m1}
    result = self.{m2}(data)
  File "{f2}", line {r.randint(10, 300)}, in {m2}
    {v} = await self._{m3}()
  File "{f3}", line {r.randint(10, 200)}, in _{m3}
    return {mod}.{m4}({v})
  File "{f4}", line {r.randint(1, 200)}, in {m4}
    raise ValueError("{err}")
ValueError: {err}

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "{f1}", line {r.randint(10, 500)}, in {m1}
    self._{v} = {mod}.{m1}()
  File "{f2}", line {r.randint(10, 300)}, in __init__
    raise RuntimeError("{err2}")
RuntimeError: {cls}.{m1}() failed: {err2}

The above exception was the direct cause of the following exception:

{cls2}Error: {cls}.{m1}() aborted after {err}: {err2}
"""
        elif kind == "go":
            g1 = r.randint(1, 100)
            g2 = r.randint(101, 200)
            cls2 = r.choice(_CLASSES)
            return f"""\
goroutine {g1} [running]:
runtime/debug.Stack()
    /usr/local/go/src/runtime/debug/stack.go:{r.randint(10, 50)}
main.{cls}.{m1.title()}(...)
    {f1}:{r.randint(10, 300)}
main.{cls}.{m2.title()}(0xc000{r.randint(10000, 99999):05x})
    {f2}:{r.randint(10, 300)}
main.{cls}.{m3.title()}(0xc000{r.randint(10000, 99999):05x}, 0x{r.randint(100, 999):x})
    {f3}:{r.randint(10, 300)}
panic: {err}

goroutine {g2} [select]:
main.{cls2}.{m4.title()}(0xc000{r.randint(10000, 99999):05x})
    {f4}:{r.randint(10, 300)} +0x{r.randint(100, 999):x}
created by main.New{cls2}
    {f4}:{r.randint(10, 100)}
"""
        elif kind == "rust":
            mod1, mod2, mod3 = r.sample(list(_MODULES), 3)
            return f"""\
thread 'main' panicked at '{err}', {f1}:{r.randint(10, 300)}
stack backtrace:
   0: std::panicking::begin_panic
   1: {mod1}::{cls}::{m1}
             at {f1}:{r.randint(10, 300)}
   2: {mod2}::{cls}::{m2}
             at {f2}:{r.randint(10, 300)}
   3: {mod3}::{cls}::{m3}
             at {f3}:{r.randint(10, 300)}
   4: {mod1}::main
             at {f4}:{r.randint(10, 300)}
   5: std::rt::lang_start::{{{{closure}}}}
             at /rustc/src/rt.rs:{r.randint(50, 200)}
   6: std::rt::lang_start
             at /rustc/src/rt.rs:{r.randint(50, 200)}
note: run with `RUST_BACKTRACE=1` for a full backtrace
"""
        else:
            async_cls = r.choice(_CLASSES)
            async_method = r.choice(_METHODS)
            cls2 = r.choice(_CLASSES)
            return f"""\
Error: {err}
    at {cls}.{m1} ({f1}:{r.randint(10, 300)}:{r.randint(1, 40)})
    at {cls}.{m2} ({f2}:{r.randint(10, 300)}:{r.randint(1, 40)})
    at {cls2}.{m3} ({f3}:{r.randint(10, 300)}:{r.randint(1, 40)})
    at processTicksAndRejections (node:internal/process/task_queues:{r.randint(50, 100)})
    at async {async_cls}.{async_method} ({f4}:{r.randint(10, 300)})
Caused by: {r.choice(_ERROR_MESSAGES)}
    at {cls2}.{m4} ({f3}:{r.randint(10, 300)}:{r.randint(1, 40)})
"""

    def _gen_git_diff(self, language: str | None = None) -> str:
        r = self._template_rng
        file_pool = self._file_pool(language)
        f1, f2, f3 = r.sample(list(file_pool), 3)
        m1, m2, m3 = r.sample(_METHODS, 3)
        v1, v2, v3 = r.sample(_VARS, 3)
        cls = r.choice(_CLASSES)
        ln = r.randint(10, 200)
        ln2 = r.randint(50, 300)
        err = r.choice(_ERROR_MESSAGES)
        mod = r.choice(_MODULES)
        idx = lambda: f"{r.randint(1000000, 9999999):07x}"  # noqa: E731
        hunk_old, hunk_new = r.randint(1, 50), r.randint(1, 50)
        commit_hash = f"{r.randint(1000000, 9999999):07x}"

        lang_hunks: dict[str | None, tuple[str, str, str]] = {
            "python": (
                f"""\
@@ -{ln},8 +{ln},14 @@ class {cls}:
     def {m1}(self):
-        {v1} = self._{m2}()
-        return {v1}
+        try:
+            {v1} = await self._{m2}()
+            if {v1} is None:
+                raise ValueError("{err}")
+            return {v1}
+        except Exception as e:
+            logger.error(f"{cls}.{m1} failed: {{{{e}}}}")
+            raise""",
                f"""\
@@ -{ln2},5 +{ln2},9 @@ def {m2}({v1}):
     {v2} = {mod}.{m3}({v1})
-    return {v2}
+    if not {v2}:
+        raise RuntimeError("{err}")
+    logger.info("{m2} completed: %s", {v2})
+    return {{{{"{v1}": {v2}, "status": "ok"}}}}""",
                f"""\
@@ -{hunk_old},3 +{hunk_new},7 @@
+import logging
+from {mod} import {cls}
+
+logger = logging.getLogger(__name__)""",
            ),
            "go": (
                f"""\
@@ -{ln},6 +{ln},12 @@ func (s *{cls}) {m1.title()}() error {{{{
-    return nil
+    {v1}, err := s.{m2.title()}(ctx)
+    if err != nil {{{{
+        return fmt.Errorf("{err}: %w", err)
+    }}}}
+    s.{v2} = {v1}
+    return nil""",
                f"""\
@@ -{ln2},4 +{ln2},8 @@ func (s *{cls}) {m2.title()}() (string, error) {{{{
     s.mu.RLock()
     defer s.mu.RUnlock()
-    return s.{v1}, nil
+    if s.{v1} == "" {{{{
+        return "", fmt.Errorf("{err}")
+    }}}}
+    return fmt.Sprintf("%s:%d", s.{v1}, s.{v2}), nil""",
                f"""\
@@ -{hunk_old},3 +{hunk_new},7 @@
+import (
+    "fmt"
+    "log/slog"
+)""",
            ),
            "rust": (
                f"""\
@@ -{ln},5 +{ln},11 @@ impl {cls} {{{{
     pub fn {m1}(&self) -> Result<()> {{{{
-        Ok(())
+        let {v1} = self.{m2}()?;
+        if {v1}.is_empty() {{{{
+            anyhow::bail!("{err}");
+        }}}}
+        tracing::info!("{m1} completed: {{}}", {v1});
+        Ok(())""",
                f"""\
@@ -{ln2},4 +{ln2},7 @@ impl {cls} {{{{
     fn {m2}(&self) -> Result<String> {{{{
-        Ok(self.{v1}.clone())
+        let {v2} = &self.{v1};
+        anyhow::ensure!(!{v2}.is_empty(), "{err}");
+        Ok({v2}.clone())""",
                f"""\
@@ -{hunk_old},3 +{hunk_new},6 @@
+use anyhow::Result;
+use tracing;
+use {mod}::{cls};""",
            ),
            "typescript": (
                f"""\
@@ -{ln},6 +{ln},12 @@ export class {cls} {{{{
   {m1}({v1}: string) {{{{
-    return this.{m2}({v1});
+    try {{{{
+      const {v2} = await this.{m2}({v1});
+      if (!{v2}) throw new Error('{err}');
+      return {{ status: 'ok', data: {v2} }};
+    }}}} catch (err) {{{{
+      console.error(`{cls}.{m1} failed: ${{{{err}}}}`);
+      throw err;
+    }}}}""",
                f"""\
@@ -{ln2},4 +{ln2},7 @@ export class {cls} {{{{
   private {m2}({v1}: string): {v2} {{{{
-    return this.#{v1};
+    if (!this.#{v1}) {{{{
+      throw new Error('{err}');
+    }}}}
+    return this.#{v1};""",
                f"""\
@@ -{hunk_old},3 +{hunk_new},6 @@
+import {{ {cls} }} from './{mod}';
+import type {{ {v3.title()} }} from './types';
+""",
            ),
        }
        hunk1, hunk2, hunk3 = lang_hunks.get(language, lang_hunks["python"])

        return f"""\
commit {commit_hash}
Author: dev <dev@example.com>
Date:   Mon Jan 15 14:32:00 2025 +0000

    feat({mod}): add async {m1} with error handling

diff --git a/{f1} b/{f1}
index {idx()}..{idx()} 100644
--- a/{f1}
+++ b/{f1}
{hunk1}
diff --git a/{f2} b/{f2}
index {idx()}..{idx()} 100644
--- a/{f2}
+++ b/{f2}
{hunk2}
diff --git a/{f3} b/{f3}
index {idx()}..{idx()} 100644
--- a/{f3}
+++ b/{f3}
{hunk3}
"""

    def _gen_cicd_output(self, language: str | None = None) -> str:
        r = self._template_rng
        mod = r.choice(_MODULES)
        n_pass = r.randint(20, 200)
        n_fail = r.randint(0, 5)
        n_skip = r.randint(0, 10)
        n_pkgs = r.randint(50, 300)
        install_time = r.uniform(0.5, 10)
        n_lint_files = r.randint(10, 100)
        n_type_mods = r.randint(100, 500)
        coverage = r.uniform(70, 99)
        ver = f"{r.randint(1, 9)}.{r.randint(0, 99)}.{r.randint(0, 99)}"
        artifact_size = r.uniform(0.1, 50)
        status = "PASSED" if n_fail == 0 else "FAILED"
        elapsed = r.randint(30, 600)

        lang_toolchain = {
            "python": {
                "install": f"pip install -r requirements.txt\n  Resolved {n_pkgs} packages in {install_time:.1f}s",
                "lint": f"ruff check . && ruff format --check .\n  All checks passed ({n_lint_files} files)",
                "typecheck": f"mypy src/\n  Success: {n_type_mods} modules checked",
                "test": f"pytest tests/ -v\n  {n_pass} passed, {n_fail} failed, {n_skip} skipped\n  Coverage: {coverage:.1f}%",
                "build": f"python -m build\n  Built {mod}-{ver}.tar.gz ({artifact_size:.1f} MB)",
            },
            "go": {
                "install": f"go mod download\n  Resolved {n_pkgs} packages in {install_time:.1f}s",
                "lint": f"golangci-lint run ./...\n  All checks passed ({n_lint_files} files)",
                "typecheck": f"go vet ./...\n  Success: {n_type_mods} packages checked",
                "test": f"go test -v -race -coverprofile=coverage.out ./...\n  {n_pass} passed, {n_fail} failed, {n_skip} skipped\n  Coverage: {coverage:.1f}%",
                "build": f"go build -o bin/{mod} ./cmd/{mod}\n  Built bin/{mod} ({artifact_size:.1f} MB)",
            },
            "rust": {
                "install": f"cargo fetch\n  Resolved {n_pkgs} crates in {install_time:.1f}s",
                "lint": f"cargo clippy -- -D warnings\n  All checks passed ({n_lint_files} files)",
                "typecheck": f"cargo check\n  Checked {n_type_mods} crates",
                "test": f"cargo test\n  {n_pass} passed, {n_fail} failed, {n_skip} ignored\n  Coverage: {coverage:.1f}%",
                "build": f"cargo build --release\n  Built target/release/{mod} ({artifact_size:.1f} MB)",
            },
            "typescript": {
                "install": f"npm ci\n  Resolved {n_pkgs} packages in {install_time:.1f}s",
                "lint": f"eslint src/ && prettier --check src/\n  All checks passed ({n_lint_files} files)",
                "typecheck": f"tsc --noEmit\n  Success: {n_type_mods} modules checked",
                "test": f"vitest run\n  {n_pass} passed, {n_fail} failed, {n_skip} skipped\n  Coverage: {coverage:.1f}%",
                "build": f"npm run build\n  Built dist/{mod}-{ver}.tgz ({artifact_size:.1f} MB)",
            },
        }
        toolchain = lang_toolchain.get(
            language, r.choice(list(lang_toolchain.values()))
        )

        return f"""\
=== CI Pipeline: {mod} ===
Step 1/5: Installing dependencies...
  {toolchain["install"]}
Step 2/5: Linting...
  {toolchain["lint"]}
Step 3/5: Type checking...
  {toolchain["typecheck"]}
Step 4/5: Running tests...
  {toolchain["test"]}
Step 5/5: Building artifacts...
  {toolchain["build"]}
Pipeline {status} in {elapsed}s
"""

    def _gen_config_file(self, language: str | None = None) -> str:
        r = self._template_rng
        mod = r.choice(_MODULES)
        v1, v2, v3 = r.sample(_VARS, 3)

        lang_to_kinds: dict[str, list[str]] = {
            "python": ["yaml", "toml", "dockerfile"],
            "go": ["yaml", "makefile"],
            "rust": ["toml"],
            "typescript": ["yaml", "dockerfile"],
        }
        choices = (
            lang_to_kinds.get(language, ["yaml", "toml", "dockerfile", "makefile"])
            if language
            else ["yaml", "toml", "dockerfile", "makefile"]
        )
        kind = r.choice(choices)
        if kind == "yaml":
            port = r.randint(3000, 9999)
            workers = r.randint(1, 16)
            v2_val = r.randint(1, 1000)
            v3_val = r.choice(_MODULES)
            db_port = r.choice([5432, 3306, 27017, 6379])
            pool = r.randint(5, 50)
            return f"""\
# {mod} configuration
service:
  name: {mod}
  port: {port}
  workers: {workers}
  {v1}:
    enabled: true
    {v2}: {v2_val}
    {v3}: "{v3_val}"
  logging:
    level: info
    format: json
  database:
    host: localhost
    port: {db_port}
    pool_size: {pool}
"""
        elif kind == "toml":
            ver = f"{r.randint(0, 9)}.{r.randint(0, 99)}.{r.randint(0, 99)}"
            desc_cls = r.choice(_CLASSES)
            desc_method = r.choice(_METHODS)
            dep1, dep2 = r.choice(_MODULES), r.choice(_MODULES)
            dep1_ver = f"{r.randint(1, 5)}.{r.randint(0, 20)}"
            dep2_ver = f"{r.randint(0, 3)}.{r.randint(0, 40)}"
            tool_mod = r.choice(_MODULES)
            v1_val = r.randint(1, 100)
            v2_val = r.choice(_MODULES)
            return f"""\
[project]
name = "{mod}"
version = "{ver}"
description = "{desc_cls} {desc_method} service"

[dependencies]
{dep1} = "{dep1_ver}"
{dep2} = "{dep2_ver}"

[tool.{tool_mod}]
{v1} = {v1_val}
{v2} = "{v2_val}"
{v3} = true
"""
        elif kind == "dockerfile":
            env1_val = r.randint(1, 100)
            env2_val = r.choice(_MODULES)
            port = r.randint(3000, 9999)
            docker_lang = language or "python"
            if docker_lang == "python":
                py_ver = r.randint(10, 13)
                base_image = f"python:3.{py_ver}-slim"
                install_cmd = "COPY requirements.txt .\nRUN pip install --no-cache-dir -r requirements.txt"
                run_cmd = f'CMD ["python", "-m", "{mod}"]'
            elif docker_lang == "go":
                go_ver = f"1.{r.randint(21, 23)}"
                base_image = f"golang:{go_ver}-alpine"
                install_cmd = "COPY go.mod go.sum ./\nRUN go mod download"
                run_cmd = f'CMD ["./bin/{mod}"]'
            elif docker_lang == "rust":
                base_image = "rust:1-slim"
                install_cmd = "COPY Cargo.toml Cargo.lock ./\nRUN cargo fetch"
                run_cmd = f'CMD ["./target/release/{mod}"]'
            else:
                node_ver = r.randint(18, 22)
                base_image = f"node:{node_ver}-alpine"
                install_cmd = "COPY package.json package-lock.json ./\nRUN npm ci"
                run_cmd = f'CMD ["node", "dist/{mod}/index.js"]'
            return f"""\
FROM {base_image}

WORKDIR /app

{install_cmd}

COPY src/ ./src/

ENV {v1.upper()}={env1_val}
ENV {v2.upper()}={env2_val}

EXPOSE {port}

{run_cmd}
"""
        else:
            return f"""\
.PHONY: build test lint clean

build:
\t@echo "Building {mod}..."
\tgo build -o bin/{mod} ./cmd/{mod}

test:
\t@echo "Testing {mod}..."
\tgo test -v -race ./...

lint:
\tgolangci-lint run ./...

clean:
\trm -rf bin/ dist/ *.egg-info
"""

    def _gen_markdown_doc(self, language: str | None = None) -> str:
        r = self._template_rng
        cls = r.choice(_CLASSES)
        m1, m2 = r.sample(_METHODS, 2)
        mod = r.choice(_MODULES)
        v1 = r.choice(_VARS)
        err = r.choice(_ERROR_MESSAGES)

        lang_examples = {
            "python": {
                "fence": "python",
                "code": f'from {mod} import {cls}\n\ninstance = {cls}({v1}="value")\nresult = await instance.{m1}()',
                "param_type": r.choice(
                    ("str", "int", "float", "bool", "dict", "list", "Any", "Optional")
                ),
                "return_type": r.choice(
                    ("str", "int", "bool", "dict", "list", "None", "Any")
                ),
            },
            "go": {
                "fence": "go",
                "code": f'import "{mod}"\n\nc := {mod}.New{cls}("{v1}")\nerr := c.{m1.title()}(ctx)',
                "param_type": r.choice(
                    (
                        "string",
                        "int",
                        "int64",
                        "bool",
                        "[]byte",
                        "error",
                        "context.Context",
                    )
                ),
                "return_type": r.choice(("string", "int", "bool", "error", f"*{cls}")),
            },
            "rust": {
                "fence": "rust",
                "code": f'use {mod}::{cls};\n\nlet mut c = {cls}::new("{v1}");\nc.{m1}().await?;',
                "param_type": r.choice(
                    (
                        "&str",
                        "String",
                        "i64",
                        "bool",
                        "Vec<u8>",
                        "&[u8]",
                        "Option<String>",
                    )
                ),
                "return_type": r.choice(
                    ("Result<()>", "Result<String>", "bool", "Option<String>", "&str")
                ),
            },
            "typescript": {
                "fence": "typescript",
                "code": f"import {{ {cls} }} from './{mod}';\n\nconst c = new {cls}({{ {v1}: 'value' }});\nawait c.{m1}();",
                "param_type": r.choice(
                    (
                        "string",
                        "number",
                        "boolean",
                        "Record<string, unknown>",
                        "unknown[]",
                    )
                ),
                "return_type": r.choice(
                    ("string", "number", "boolean", "void", "Promise<void>")
                ),
            },
        }
        example = lang_examples.get(language, r.choice(list(lang_examples.values())))

        v2, v3 = r.sample(_VARS, 2)
        err2 = r.choice(_ERROR_MESSAGES)
        env_var = f"AIPERF_{mod.upper()}_{v1.upper()}"

        return f"""\
# {cls}

## Overview

The `{cls}` class provides {m1} and {m2} operations for the `{mod}` module.

## Usage

```{example["fence"]}
{example["code"]}
```

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `{v1}` | {example["param_type"]} | required | Primary {v1} identifier |
| `{v2}` | {example["param_type"]} | `None` | Optional {v2} override |
| `{v3}` | int | `10` | Maximum {v3} per batch |
| `timeout` | float | `30.0` | Operation timeout in seconds |

Environment variable override: `{env_var}`

## API Reference

### `{m1}({v1})`

Performs the {m1} operation.

**Parameters:**
- `{v1}` ({example["param_type"]}): The input {v1}.

**Returns:** {example["return_type"]}

### `{m2}()`

Performs the {m2} operation.

**Raises:** `ValueError` if {err}.

## Errors

| Error | Condition | Recovery |
|-------|-----------|----------|
| `ValueError` | {err} | Check {v1} parameter |
| `RuntimeError` | {err2} | Retry with backoff |
| `TimeoutError` | Operation exceeds timeout | Increase timeout or reduce {v3} |
"""

    def _gen_test_output(self, language: str | None = None) -> str:
        r = self._template_rng
        mod = r.choice(_MODULES)
        cls = r.choice(_CLASSES)
        methods = r.sample(list(_METHODS), 5)

        lang_to_kind = {
            "python": "pytest",
            "go": "go",
            "rust": "cargo",
            "typescript": "jest",
        }
        kind = (
            lang_to_kind[language]
            if language in lang_to_kind
            else r.choice(["pytest", "go", "cargo"])
        )
        if kind == "pytest":
            lines = [
                "============================= test session starts ============================="
            ]
            lines.append(f"collected {r.randint(10, 100)} items\n")
            for m in methods:
                status = r.choice(["PASSED", "PASSED", "PASSED", "FAILED"])
                lines.append(f"tests/test_{mod}.py::Test{cls}::test_{m} {status}")
            n_pass = sum(1 for line in lines if "PASSED" in line)
            n_fail = len(methods) - n_pass
            dur = r.uniform(0.5, 30.0)
            lines.append(f"\n{'=' * 70}")
            lines.append(f"{n_pass} passed, {n_fail} failed in {dur:.2f}s")
            return "\n".join(lines) + "\n"
        elif kind == "jest":
            runner = r.choice(["JEST", "VITEST"])
            lines = [
                f" {runner}  v{r.randint(28, 30)}.{r.randint(0, 9)}.{r.randint(0, 9)}"
            ]
            lines.append("")
            results: list[str] = []
            for m in methods:
                passed = r.choice([True, True, True, False])
                mark = "\u2713" if passed else "\u2717"
                dur_ms = r.randint(1, 500)
                results.append(f"  {mark} {cls} > {m} ({dur_ms} ms)")
                lines.append(results[-1])
            n_pass = sum(1 for res in results if "\u2713" in res)
            n_fail = len(methods) - n_pass
            dur = r.uniform(0.5, 15.0)
            lines.append("")
            lines.append(
                f"Tests:       {n_fail} failed, {n_pass} passed, {len(methods)} total"
            )
            lines.append(f"Time:        {dur:.3f} s")
            lines.append(f"Ran all test suites matching /src/{mod}.test.ts/i.")
            return "\n".join(lines) + "\n"
        elif kind == "go":
            lines = []
            for m in methods:
                status = r.choice(["ok", "ok", "ok", "FAIL"])
                dur = r.uniform(0.001, 2.0)
                lines.append(f"--- {status}: Test{m.title()} ({dur:.3f}s)")
            lines.append(
                f"{status}  \t{mod}/{r.choice(_MODULES)}\t{r.uniform(0.1, 5.0):.3f}s"
            )
            return "\n".join(lines) + "\n"
        else:
            lines = [f"   Compiling {mod} v0.{r.randint(1, 99)}.{r.randint(0, 9)}"]
            lines.append(f"    Finished test target(s) in {r.uniform(1, 30):.2f}s")
            lines.append("     Running unittests src/lib.rs\n")
            for m in methods:
                status = r.choice(["ok", "ok", "ok", "FAILED"])
                lines.append(f"test {mod}::{cls.lower()}::test_{m} ... {status}")
            n_pass = sum(1 for line in lines if "... ok" in line)
            n_fail = len(methods) - n_pass
            lines.append(
                f"\ntest result: {'ok' if n_fail == 0 else 'FAILED'}. "
                f"{n_pass} passed; {n_fail} failed; 0 ignored"
            )
            return "\n".join(lines) + "\n"

    def _gen_ml_training_code(self) -> str:
        r = self._template_rng
        model = r.choice(_MODEL_NAMES)
        imp1, imp2, imp3 = r.sample(list(_ML_IMPORTS), 3)
        cls1, cls2 = r.sample(list(_ML_CLASSES), 2)
        m1, m2 = r.sample(list(_ML_METHODS), 2)
        v1, v2, v3, v4 = r.sample(list(_ML_VARS), 4)
        lr = r.choice([1e-5, 2e-5, 5e-5, 1e-4, 3e-4])
        epochs = r.randint(1, 10)
        bs = r.choice([1, 2, 4, 8, 16, 32])
        grad_accum = r.choice([1, 2, 4, 8])

        return f"""\
import {imp1}
import {imp2}
from {imp3} import {cls1}, {cls2}

model_name = "{model}"
tokenizer = {cls2}.from_pretrained(model_name)
model = {cls1}.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

train_dataset = datasets.load_dataset("json", data_files="train.jsonl", split="train")

training_args = TrainingArguments(
    output_dir="./checkpoints",
    num_train_epochs={epochs},
    per_device_train_batch_size={bs},
    gradient_accumulation_steps={grad_accum},
    learning_rate={lr},
    max_grad_norm=1.0,
    warmup_ratio=0.1,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    report_to="wandb",
)

optimizer = torch.optim.AdamW(model.parameters(), lr={lr}, weight_decay=0.01)

for epoch in range({epochs}):
    model.train()
    for step, batch in enumerate(train_loader):
        {v1} = batch["{v1}"].to("cuda")
        {v2} = batch["{v2}"].to("cuda")
        outputs = model({m1}={v1}, {v2}={v2})
        {v3} = outputs.{v3}
        {v3}.{m2}()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        if step % 10 == 0:
            print(f"Epoch {{epoch}} Step {{step}} {v3}: {{{{{v3}.item():.4f}}}}")

model.save_pretrained("./final_model")
tokenizer.save_pretrained("./final_model")
"""

    def _gen_ml_inference_code(self) -> str:
        r = self._template_rng
        model = r.choice(_MODEL_NAMES)
        cls1 = r.choice(("AutoModelForCausalLM", "AutoModelForSeq2SeqLM"))
        v1, v2, v3 = r.sample(list(_ML_VARS), 3)
        temp = r.choice([0.1, 0.3, 0.7, 1.0])
        top_p = r.choice([0.9, 0.95, 1.0])
        max_new = r.choice([128, 256, 512, 1024, 2048])

        return f"""\
import torch
from transformers import {cls1}, AutoTokenizer, GenerationConfig

model_name = "{model}"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = {cls1}.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="flash_attention_2",
)

generation_config = GenerationConfig(
    max_new_tokens={max_new},
    temperature={temp},
    top_p={top_p},
    do_sample={"True" if temp > 0 else "False"},
    repetition_penalty=1.1,
)

prompt = "Explain the architecture of a transformer model."
{v1} = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.inference_mode():
    {v2} = model.generate(
        **{v1},
        generation_config=generation_config,
        pad_token_id=tokenizer.eos_token_id,
    )

{v3} = tokenizer.batch_decode({v2}[:, {v1}["{v1}"].shape[-1]:], skip_special_tokens=True)
print({v3}[0])
"""

    def _gen_ml_config(self) -> str:
        r = self._template_rng
        model = r.choice(_MODEL_NAMES)
        lr = r.choice([1e-5, 2e-5, 5e-5, 1e-4, 3e-4])
        epochs = r.randint(1, 10)
        bs = r.choice([1, 2, 4, 8, 16, 32])
        grad_accum = r.choice([1, 2, 4, 8])
        max_len = r.choice([512, 1024, 2048, 4096])
        warmup = r.choice([0.03, 0.05, 0.1])
        lora_r = r.choice([8, 16, 32, 64])
        lora_alpha = lora_r * 2
        quant_bits = r.choice([4, 8])

        return f"""\
{{{{
  "model_name_or_path": "{model}",
  "torch_dtype": "bfloat16",
  "attn_implementation": "flash_attention_2",
  "max_seq_length": {max_len},
  "training": {{{{
    "num_train_epochs": {epochs},
    "per_device_train_batch_size": {bs},
    "gradient_accumulation_steps": {grad_accum},
    "learning_rate": {lr},
    "weight_decay": 0.01,
    "warmup_ratio": {warmup},
    "lr_scheduler_type": "cosine",
    "max_grad_norm": 1.0,
    "bf16": true,
    "gradient_checkpointing": true,
    "optim": "adamw_torch_fused"
  }}}},
  "lora": {{{{
    "r": {lora_r},
    "lora_alpha": {lora_alpha},
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "task_type": "CAUSAL_LM"
  }}}},
  "quantization": {{{{
    "load_in_{quant_bits}bit": true,
    "bnb_{quant_bits}bit_compute_dtype": "bfloat16",
    "bnb_{quant_bits}bit_quant_type": "nf4",
    "bnb_{quant_bits}bit_use_double_quant": true
  }}}},
  "data": {{{{
    "dataset_name": "train.jsonl",
    "max_length": {max_len},
    "packing": true,
    "num_proc": 8
  }}}}
}}}}
"""

    def _gen_ml_training_log(self) -> str:
        r = self._template_rng
        model = r.choice(_MODEL_NAMES).split("/")[-1]
        total_steps = r.randint(500, 10000)
        epoch = r.randint(0, 5)
        lines = []

        for _ in range(r.randint(8, 15)):
            step = r.randint(1, total_steps)
            loss = r.uniform(0.3, 4.0)
            lr_val = r.uniform(1e-6, 5e-4)
            grad = r.uniform(0.1, 10.0)
            tokens_per_sec = r.randint(1000, 50000)
            lines.append(
                f"{{{{'step': {step}, 'epoch': {epoch + step / total_steps:.2f}, "
                f"'loss': {loss:.4f}, 'lr': {lr_val:.2e}, "
                f"'grad_norm': {grad:.3f}, 'tokens_per_sec': {tokens_per_sec}}}}}"
            )

        gpu_mem = r.uniform(10, 80)
        gpu_util = r.randint(80, 100)
        eval_loss = r.uniform(0.5, 3.0)
        eval_ppl = r.uniform(2.0, 20.0)

        lines.append(
            f"\n[Eval] epoch={epoch + 1} loss={eval_loss:.4f} perplexity={eval_ppl:.2f}"
        )
        lines.append(f"[GPU] memory_allocated={gpu_mem:.1f}GB utilization={gpu_util}%")
        lines.append(
            f"[GPU] peak_memory={gpu_mem + r.uniform(1, 10):.1f}GB "
            f"reserved={gpu_mem + r.uniform(5, 20):.1f}GB"
        )
        lines.append(
            f"[Checkpoint] Saved model checkpoint to ./checkpoints/{model}/step-{total_steps}"
        )

        return "\n".join(lines) + "\n"

    def _gen_cuda_error(self) -> str:
        r = self._template_rng
        err = r.choice(_CUDA_ERRORS)
        model = r.choice(_MODEL_NAMES).split("/")[-1]
        rank = r.randint(0, 7)
        gpu_id = r.randint(0, 7)
        alloc_gb = r.uniform(0.5, 16.0)
        total_gb = r.choice([24.0, 40.0, 48.0, 80.0])
        free_gb = r.uniform(0.01, 2.0)
        cls1, cls2 = r.sample(list(_ML_CLASSES), 2)
        m1, m2 = r.sample(list(_ML_METHODS), 2)

        return f"""\
Traceback (most recent call last):
  File "train.py", line {r.randint(50, 300)}, in main
    outputs = model.{m1}(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
  File "torch/nn/modules/module.py", line {r.randint(1400, 1600)}, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "transformers/models/llama/modeling_llama.py", line {r.randint(800, 1200)}, in {m1}
    hidden_states = self.model(input_ids, attention_mask=attention_mask)
  File "torch/nn/modules/module.py", line {r.randint(1400, 1600)}, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "transformers/models/llama/modeling_llama.py", line {r.randint(400, 800)}, in {m2}
    layer_outputs = decoder_layer(hidden_states, attention_mask=attention_mask)
{err}

|===========================================================================|
|                  PyTorch CUDA memory summary, device: {gpu_id}                |
|---------------------------------------------------------------------------|
|            CUDA OOMs: {r.randint(1, 5):>10}                                          |
|---------------------------------------------------------------------------|
|        Metric        |  Cur Usage  |  Peak Usage  |  Total Alloc  |
|---------------------------------------------------------------------------|
| Allocated memory     | {alloc_gb:>8.2f} GB | {total_gb - free_gb:>8.2f} GB  | {total_gb * r.randint(2, 10):>9.2f} GB  |
| Reserved memory      | {total_gb - free_gb + 1:>8.2f} GB | {total_gb:>8.2f} GB  | {total_gb * r.randint(2, 10):>9.2f} GB  |
| Free memory          | {free_gb:>8.2f} GB |              |               |
|===========================================================================|

Model: {model} | Rank: {rank} | GPU: {gpu_id} (NVIDIA A100-SXM4-{int(total_gb)}GB)
"""

    def _gen_sql_query(self) -> str:
        r = self._template_rng
        t1, t2, t3 = r.sample(list(_DB_TABLES), 3)
        v1, v2, v3 = r.sample(list(_VARS), 3)
        kind = r.choice(["select_join", "insert", "create", "alter"])

        if kind == "select_join":
            limit = r.randint(10, 1000)
            offset = r.randint(0, 500)
            return f"""\
SELECT
    t1.id,
    t1.{v1},
    t1.created_at,
    t2.{v2},
    t2.{v3},
    COUNT(t3.id) AS {v3}_count
FROM {t1} t1
INNER JOIN {t2} t2 ON t2.{t1}_id = t1.id
LEFT JOIN {t3} t3 ON t3.{t2}_id = t2.id
WHERE t1.status = 'active'
  AND t1.created_at >= NOW() - INTERVAL '30 days'
  AND t2.{v2} IS NOT NULL
GROUP BY t1.id, t1.{v1}, t1.created_at, t2.{v2}, t2.{v3}
HAVING COUNT(t3.id) > 0
ORDER BY t1.created_at DESC
LIMIT {limit} OFFSET {offset};
"""
        elif kind == "insert":
            n_rows = r.randint(1, 5)
            rows = []
            for _ in range(n_rows):
                rows.append(
                    f"    ('{r.choice(_MODULES)}', {r.randint(1, 1000)}, "
                    f"'{r.choice(_STATUS_CODES).split()[0]}', NOW())"
                )
            rows_str = ",\n".join(rows)
            return f"""\
INSERT INTO {t1} ({v1}, {v2}, status, created_at)
VALUES
{rows_str}
ON CONFLICT ({v1})
DO UPDATE SET
    {v2} = EXCLUDED.{v2},
    status = EXCLUDED.status,
    updated_at = NOW()
RETURNING id, {v1}, {v2};
"""
        elif kind == "create":
            return f"""\
CREATE TABLE IF NOT EXISTS {t1} (
    id BIGSERIAL PRIMARY KEY,
    {v1} VARCHAR(256) NOT NULL,
    {v2} INTEGER DEFAULT 0,
    {v3} JSONB DEFAULT '{{}}'::jsonb,
    status VARCHAR(32) DEFAULT 'pending',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT {t1}_{v1}_unique UNIQUE ({v1})
);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_{t1}_{v1}
    ON {t1} ({v1});
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_{t1}_status_created
    ON {t1} (status, created_at DESC);
"""
        else:
            col_type = r.choice(
                ["VARCHAR(256)", "INTEGER", "BOOLEAN", "JSONB", "TIMESTAMPTZ"]
            )
            return f"""\
BEGIN;

ALTER TABLE {t1}
    ADD COLUMN IF NOT EXISTS {v1} {col_type},
    ADD COLUMN IF NOT EXISTS {v2} INTEGER DEFAULT 0;

UPDATE {t1}
SET {v1} = (
    SELECT {v2} FROM {t2}
    WHERE {t2}.{t1}_id = {t1}.id
    LIMIT 1
)
WHERE {v1} IS NULL;

ALTER TABLE {t1}
    ALTER COLUMN {v1} SET NOT NULL;

COMMIT;
"""

    def _gen_user_prompt(self) -> str:
        r = self._template_rng
        template = r.choice(_USER_REQUESTS)
        base = template.format(
            module=r.choice(_MODULES),
            cls=r.choice(_CLASSES),
            method=r.choice(_METHODS),
            var=r.choice(_VARS),
            error=r.choice(_ERROR_MESSAGES),
            type=r.choice(_TYPES),
        )

        if r.random() < 0.3:
            base += "\n\n" + self._gen_prompt_context()
        return base

    def _gen_prompt_context(self) -> str:
        r = self._template_rng
        kind = r.choice(["snippet", "error_output", "constraint"])
        if kind == "snippet":
            cls = r.choice(_CLASSES)
            m1 = r.choice(_METHODS)
            v1, v2 = r.sample(_VARS, 2)
            f = r.choice(_FILE_PATHS)
            return (
                f"Here's the relevant code from `{f}`:\n\n"
                f"```\n"
                f"class {cls}:\n"
                f"    def {m1}(self, {v1}):\n"
                f"        {v2} = self._{v1}\n"
                f"        return {v2}\n"
                f"```"
            )
        elif kind == "error_output":
            err = r.choice(_ERROR_MESSAGES)
            cls = r.choice(_CLASSES)
            m1 = r.choice(_METHODS)
            f = r.choice(_FILE_PATHS)
            return (
                f"Error output:\n\n"
                f"```\n"
                f'  File "{f}", line {r.randint(10, 300)}, in {m1}\n'
                f'    raise RuntimeError("{err}")\n'
                f"RuntimeError: {err}\n"
                f"```"
            )
        else:
            return r.choice(
                (
                    "Constraint: no new dependencies allowed in this PR.",
                    "This is on the hot path — keep allocations minimal.",
                    "Must remain backward-compatible with the v1 API.",
                    f"The {r.choice(_MODULES)} service is frozen — only touch {r.choice(_MODULES)}.",
                    f"Target is under {r.randint(5, 50)}ms p99 latency.",
                    "We need this for the release on Friday — keep it simple.",
                )
            )

    def _gen_coding_conversation(self) -> str:
        r = self._template_rng
        return r.choice(
            [
                self._gen_conv_bugfix,
                self._gen_conv_review,
                self._gen_conv_feature,
                self._gen_conv_debug,
                self._gen_conv_qa,
                self._gen_conv_refactor,
                self._gen_conv_perf,
                self._gen_conv_cicd,
                self._gen_conv_ml_debug,
                self._gen_conv_test_write,
                self._gen_conv_migration,
                self._gen_conv_deploy,
                self._gen_conv_security,
                self._gen_conv_distributed,
                self._gen_conv_observability,
                self._gen_conv_db_optimize,
                self._gen_conv_architecture_review,
                self._gen_conv_incident_response,
            ]
        )()

    def _conv_ids(self) -> dict[str, str]:
        r = self._template_rng
        return {
            "cls": r.choice(_CLASSES),
            "module": r.choice(_MODULES),
            "method": r.choice(_METHODS),
            "var": r.choice(_VARS),
            "error": r.choice(_ERROR_MESSAGES),
        }

    def _conv_bridge(self, pool: tuple[str, ...], ids: dict[str, str]) -> str:
        r = self._template_rng
        return r.choice(pool).format_map(_SafeFormatMap(ids))

    def _conv_user_msg(self, ids: dict[str, str]) -> str:
        r = self._template_rng
        template = r.choice(_USER_REQUESTS)
        return template.format_map(_SafeFormatMap(ids))

    def _gen_tool_read_long(self, language: str | None = None) -> str:
        """Like _gen_tool_read but with 40-80 lines for realistic large file reads."""
        r = self._template_rng
        file_pool = self._file_pool(language)
        f = r.choice(file_pool)
        start_line = r.randint(1, 200)
        cls = r.choice(_CLASSES)
        m1, m2, m3 = r.sample(_METHODS, 3)
        v1, v2, v3 = r.sample(_VARS, 3)
        mod = r.choice(_MODULES)
        err = r.choice(_ERROR_MESSAGES)
        t1, t2 = r.sample(_TYPES, 2)

        blocks: dict[str | None, list[str]] = {
            "python": [
                f"class {cls}:",
                f'    """{cls} handles {m1} operations for {mod}."""',
                "",
                f"    _default_{v3} = 64",
                "",
                f"    def __init__(self, {v1}: {t1}, {v2}: {t2} = None):",
                f"        self._{v1} = {v1}",
                f"        self._{v2} = {v2}",
                f"        self._{v3} = self._default_{v3}",
                "        self._initialized = False",
                "        self._lock = asyncio.Lock()",
                "",
                f"    async def {m1}(self, {v1}: {t1}) -> {t2}:",
                "        if not self._initialized:",
                f'            raise RuntimeError("{cls} not initialized")',
                "        async with self._lock:",
                f"            {v2} = await self._{m2}({v1})",
                f"            if {v2} is None:",
                f'                raise ValueError("{err}")',
                f"            return {v2}",
                "",
                f"    async def _{m2}(self, {v1}: {t1}) -> {t2}:",
                "        try:",
                f"            {v2} = await {mod}.{m2}({v1})",
                f'            logger.debug(f"{cls}.{m2}: {{{{{v1}}}}}")',
                f"            return {v2}",
                "        except Exception as e:",
                f'            logger.error("{err}: %s", e)',
                f'            raise ValueError("{err}") from e',
                "",
                f"    async def {m3}(self, {v1}: {t1}, {v2}: {t2}) -> None:",
                f"        if {v1} is None:",
                "            return",
                f"        existing = await self._{m2}({v1})",
                "        if existing is not None:",
                f"            existing.{v3} = {v2}",
                "            await existing.save()",
                "        else:",
                f"            await {mod}.{m3}({v1}, {v2})",
                "",
                f"    def {m1}_sync(self) -> None:",
                "        self._initialized = True",
                f"        self._{v3} = 0",
            ],
            "go": [
                f"type {cls} struct {{",
                f"\t{v1} {t1}",
                f"\t{v2} {t2}",
                "\tmu   sync.RWMutex",
                "\tlog  *zap.Logger",
                "}",
                "",
                f"func New{cls}({v1} {t1}, log *zap.Logger) *{cls} {{",
                f"\treturn &{cls}{{",
                f"\t\t{v1}: {v1},",
                "\t\tlog: log,",
                "\t}",
                "}",
                "",
                f"func (s *{cls}) {m1.title()}(ctx context.Context) error {{",
                "\ts.mu.Lock()",
                "\tdefer s.mu.Unlock()",
                "",
                f"\t{v2}, err := s.{m2.title()}(ctx)",
                "\tif err != nil {",
                f'\t\treturn fmt.Errorf("{err}: %w", err)',
                "\t}",
                f"\ts.{v1} = {v2}",
                "\treturn nil",
                "}",
                "",
                f"func (s *{cls}) {m2.title()}(ctx context.Context) ({t2}, error) {{",
                f'\ts.log.Debug("{cls}.{m2.title()}", zap.String("{v1}", s.{v1}))',
                f"\tresult, err := {mod}.{m2.title()}(ctx, s.{v1})",
                "\tif err != nil {",
                f'\t\treturn "", fmt.Errorf("{err}: %w", err)',
                "\t}",
                "\treturn result, nil",
                "}",
            ],
            "rust": [
                f"pub struct {cls} {{",
                f"    {v1}: {t1},",
                f"    {v2}: Option<{t2}>,",
                "    initialized: bool,",
                "}",
                "",
                f"impl {cls} {{",
                f"    pub fn new({v1}: {t1}) -> Self {{",
                f"        Self {{ {v1}, {v2}: None, initialized: false }}",
                "    }",
                "",
                f"    pub async fn {m1}(&mut self) -> Result<{t2}> {{",
                f'        anyhow::ensure!(self.initialized, "{cls} not initialized");',
                f"        let {v2} = self.{m2}().await?;",
                f"        if {v2}.is_empty() {{",
                f'            anyhow::bail!("{err}");',
                "        }",
                f"        Ok({v2})",
                "    }",
                "",
                f"    async fn {m2}(&self) -> Result<{t2}> {{",
                f"        let {v2} = {mod}::{m2}(&self.{v1}).await?;",
                f'        tracing::debug!("{cls}.{m2}: {{}}", self.{v1});',
                f"        Ok({v2})",
                "    }",
                "",
                f"    pub async fn {m3}(&mut self, {v1}: {t1}) -> Result<()> {{",
                f"        let existing = self.{m2}().await.ok();",
                "        match existing {",
                "            Some(val) if !val.is_empty() => {",
                f"                self.{v2} = Some(val);",
                "            }",
                "            _ => {",
                f"                {mod}::{m3}(&{v1}).await?;",
                "            }",
                "        }",
                "        Ok(())",
                "    }",
                "}",
            ],
            "typescript": [
                f"export class {cls} {{",
                f"  private {v1}: {t1};",
                f"  private {v2}: {t2} | null = null;",
                "  private initialized = false;",
                "",
                f"  constructor({v1}: {t1}) {{",
                f"    this.{v1} = {v1};",
                "  }",
                "",
                f"  async {m1}({v1}: {t1}): Promise<{t2}> {{",
                "    if (!this.initialized) {",
                f"      throw new Error('{cls} not initialized');",
                "    }",
                f"    const {v2} = await this.{m2}({v1});",
                f"    if (!{v2}) {{",
                f"      throw new Error('{err}');",
                "    }",
                f"    return {v2};",
                "  }",
                "",
                f"  private async {m2}({v1}: {t1}): Promise<{t2} | null> {{",
                "    try {",
                f"      const {v2} = await {mod}.{m2}({v1});",
                f"      console.debug(`{cls}.{m2}: ${{{{{v1}}}}}`);",
                f"      return {v2};",
                "    } catch (err) {",
                f"      console.error('{err}:', err);",
                "      throw err;",
                "    }",
                "  }",
                "",
                f"  async {m3}({v1}: {t1}, {v2}: {t2}): Promise<void> {{",
                f"    const existing = await this.{m2}({v1}).catch(() => null);",
                "    if (existing) {",
                f"      Object.assign(existing, {{ {v3}: {v2} }});",
                "      await existing.save();",
                "    } else {",
                f"      await {mod}.{m3}({v1}, {v2});",
                "    }",
                "  }",
                "}",
            ],
        }
        code_lines = blocks.get(language, blocks["python"])

        lines = []
        for i, content in enumerate(code_lines, start=start_line):
            lines.append(f"{i:>6}\t{content}")

        content = "\n".join(lines)
        return f"""\
<tool_name>read</tool_name>
<parameter name="file_path">{f}</parameter>
<result>
{content}
</result>
"""

    def _gen_tool_bash_verbose(self, language: str | None = None) -> str:
        """Like _gen_tool_bash but with longer, more realistic test output."""
        r = self._template_rng
        mod = r.choice(_MODULES)
        cls = r.choice(_CLASSES)
        methods = r.sample(list(_METHODS), r.randint(8, 15))
        n_pass = r.randint(30, 150)
        n_fail = r.randint(0, 3)
        dur = r.uniform(2.0, 45.0)

        lang_cmds: dict[str | None, str] = {
            "python": "pytest -xvs tests/",
            "go": "go test -v ./...",
            "rust": "cargo test",
            "typescript": "npx vitest run",
        }
        cmd = lang_cmds.get(language, r.choice(_CLI_COMMANDS))

        test_lines = []
        for m in methods:
            passed = r.random() > 0.15
            t = r.uniform(0.001, 3.0)
            if language == "go":
                status = "ok" if passed else "FAIL"
                test_lines.append(f"--- {status}: Test{m.title()} ({t:.3f}s)")
                if not passed:
                    v = r.choice(_VARS)
                    test_lines.append(
                        f"        {mod}_test.go:{r.randint(20, 300)}: "
                        f"expected {v} to be non-nil"
                    )
            elif language == "rust":
                status = "ok" if passed else "FAILED"
                test_lines.append(f"test {mod}::{cls.lower()}::test_{m} ... {status}")
                if not passed:
                    err = r.choice(_ERROR_MESSAGES)
                    test_lines.append(f"  thread '{m}' panicked at '{err}'")
            elif language == "typescript":
                mark = "\u2713" if passed else "\u2717"
                test_lines.append(f"  {mark} {cls} > {m} ({r.randint(1, 800)} ms)")
                if not passed:
                    test_lines.append("    Expected: true\n    Received: false")
            else:
                status = "PASSED" if passed else "FAILED"
                test_lines.append(f"tests/test_{mod}.py::Test{cls}::test_{m} {status}")
                if not passed:
                    err = r.choice(_ERROR_MESSAGES)
                    v = r.choice(_VARS)
                    test_lines.extend(
                        [
                            f"    FAILED tests/test_{mod}.py::Test{cls}::test_{m}",
                            f"    AssertionError: assert {v} == expected",
                            f"      where {v} = {cls}().{m}()",
                            f"    {err}",
                        ]
                    )
        test_output = "\n".join(test_lines)

        warnings = ""
        if r.random() < 0.4:
            w_count = r.randint(1, 5)
            warnings = f"\n\n{w_count} warning(s)"

        return f"""\
<tool_name>bash</tool_name>
<parameter name="command">{cmd}</parameter>
<result>
{test_output}
{warnings}
========================= {n_pass} passed, {n_fail} failed in {dur:.2f}s =========================
</result>
"""

    def _gen_tool_search_verbose(self, language: str | None = None) -> str:
        """Like _gen_tool_search but returns many matches across multiple files."""
        r = self._template_rng
        file_pool = self._file_pool(language)
        pattern = r.choice(_CLASSES)

        files = r.sample(list(file_pool), min(r.randint(6, 12), len(file_pool)))
        matches = []
        for f in files:
            n_hits = r.randint(1, 4)
            for _ in range(n_hits):
                line_num = r.randint(1, 500)
                v = r.choice(_VARS)
                m = r.choice(_METHODS)
                ctx = r.choice(
                    [
                        f"class {pattern}({r.choice(_CLASSES)}):",
                        f"    {m} = {pattern}({v})",
                        f"from {r.choice(_MODULES)} import {pattern}",
                        f"    self._{v} = {pattern}.{m}()",
                        f"    result: {pattern} = await svc.{m}({v})",
                        f"# TODO: refactor {pattern} to use async",
                    ]
                )
                matches.append(f"{f}:{line_num}:{ctx}")

        content = "\n".join(matches)
        return f"""\
<tool_name>search</tool_name>
<parameter name="pattern">{pattern}</parameter>
<result>
Found {len(matches)} matches in {len(files)} files:

{content}
</result>
"""

    def _gen_conv_bugfix(self) -> str:
        r = self._template_rng
        lang = r.choice(_LANGUAGES)
        ids = self._conv_ids()

        turns = [
            f"[User]\n{self._conv_user_msg(ids)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_ANALYZE, ids)}\n\n"
            f"{self._gen_tool_read_long(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_FIX, ids)}\n\n"
            f"{self._gen_tool_edit(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_TEST, ids)}\n\n"
            f"{self._gen_tool_bash(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_SUMMARY, ids)}",
        ]
        return "\n\n".join(turns)

    def _gen_conv_review(self) -> str:
        r = self._template_rng
        lang = r.choice(_LANGUAGES)
        ids = self._conv_ids()

        turns = [
            f"[User]\n{self._conv_user_msg(ids)}\n\n"
            f"{self._gen_git_diff(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_ANALYZE, ids)}\n\n"
            f"{self._gen_tool_read_long(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_FIX, ids)}\n\n"
            f"{self._gen_tool_edit(language=lang)}",
            f"[User]\n{self._conv_bridge(_FOLLOWUP_QUESTIONS, ids)}",
        ]
        return "\n\n".join(turns)

    def _gen_conv_feature(self) -> str:
        r = self._template_rng
        lang = r.choice(_LANGUAGES)
        ids = self._conv_ids()

        turns = [
            f"[User]\n{self._conv_user_msg(ids)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_ANALYZE, ids)}\n\n"
            f"{self._gen_tool_search_verbose(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_ANALYZE, ids)}\n\n"
            f"{self._gen_tool_read_long(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_FIX, ids)}\n\n"
            f"{self._gen_tool_edit(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_WRITE_TEST, ids)}\n\n"
            f"{self._gen_tool_edit(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_TEST, ids)}\n\n"
            f"{self._gen_tool_bash_verbose(language=lang)}",
        ]
        return "\n\n".join(turns)

    def _gen_conv_debug(self) -> str:
        r = self._template_rng
        lang = r.choice(_LANGUAGES)
        ids = self._conv_ids()
        error_block = r.choice(
            [
                lambda: self._gen_error_traceback(language=lang),
                self._gen_cuda_error,
            ]
        )()

        turns = [
            f"[User]\n{self._conv_user_msg(ids)}\n\n{error_block}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_ANALYZE, ids)}\n\n"
            f"{self._gen_tool_read_long(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_ANALYZE, ids)}\n\n"
            f"{self._gen_tool_search_verbose(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_FIX, ids)}\n\n"
            f"{self._gen_tool_edit(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_SUMMARY, ids)}",
        ]
        return "\n\n".join(turns)

    def _gen_conv_qa(self) -> str:
        r = self._template_rng
        lang = r.choice(_LANGUAGES)
        ids = self._conv_ids()

        turns = [
            f"[User]\n{self._conv_user_msg(ids)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_ANALYZE, ids)}\n\n"
            f"{self._gen_tool_read_long(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_EXPLAIN, ids)}",
            f"[User]\n{self._conv_bridge(_FOLLOWUP_QUESTIONS, ids)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_FIX, ids)}\n\n"
            f"{self._gen_tool_edit(language=lang)}",
        ]
        return "\n\n".join(turns)

    def _gen_conv_refactor(self) -> str:
        """Multi-file refactoring: search callers, read multiple files, edit each."""
        r = self._template_rng
        lang = r.choice(_LANGUAGES)
        ids = self._conv_ids()

        turns = [
            f"[User]\n{self._conv_user_msg(ids)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_ANALYZE, ids)}\n\n"
            f"{self._gen_tool_search_verbose(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_ANALYZE, ids)}\n\n"
            f"{self._gen_tool_read_long(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_ANALYZE, ids)}\n\n"
            f"{self._gen_tool_read(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_REFACTOR, ids)}\n\n"
            f"{self._gen_tool_edit(language=lang)}",
            f"[Assistant]\nNow let me update the callers.\n\n"
            f"{self._gen_tool_edit(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_REFACTOR, ids)}\n\n"
            f"{self._gen_tool_edit(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_TEST, ids)}\n\n"
            f"{self._gen_tool_bash_verbose(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_SUMMARY, ids)}",
        ]
        return "\n\n".join(turns)

    def _gen_conv_perf(self) -> str:
        """Performance investigation: profile, read hot path, optimize, benchmark."""
        r = self._template_rng
        lang = r.choice(_LANGUAGES)
        ids = self._conv_ids()

        turns = [
            f"[User]\n{self._conv_user_msg(ids)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_PERF, ids)}\n\n"
            f"{self._gen_tool_bash(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_ANALYZE, ids)}\n\n"
            f"{self._gen_tool_read_long(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_PERF, ids)}\n\n"
            f"{self._gen_tool_search(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_FIX, ids)}\n\n"
            f"{self._gen_tool_edit(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_TEST, ids)}\n\n"
            f"{self._gen_tool_bash_verbose(language=lang)}",
            f"[User]\n{self._conv_bridge(_FOLLOWUP_QUESTIONS, ids)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_EXPLAIN, ids)}\n\n"
            f"{self._conv_bridge(_BRIDGE_SUMMARY, ids)}",
        ]
        return "\n\n".join(turns)

    def _gen_conv_cicd(self) -> str:
        """CI/CD debugging: failing pipeline, read logs, fix config, re-run."""
        r = self._template_rng
        lang = r.choice(_LANGUAGES)
        ids = self._conv_ids()

        ci_output = self._gen_cicd_output(language=lang)

        turns = [
            f"[User]\nThe CI pipeline is failing on the {ids['module']} service. "
            f"Can you take a look?\n\n{ci_output}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_ANALYZE, ids)}\n\n"
            f"{self._gen_tool_read(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_ANALYZE, ids)}\n\n"
            f"{self._gen_tool_read(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_FIX, ids)}\n\n"
            f"{self._gen_tool_edit(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_TEST, ids)}\n\n"
            f"{self._gen_tool_bash_verbose(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_SUMMARY, ids)}",
            f"[User]\n{self._conv_bridge(_FOLLOWUP_QUESTIONS, ids)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_EXPLAIN, ids)}",
        ]
        return "\n\n".join(turns)

    def _gen_conv_ml_debug(self) -> str:
        """ML/GPU debugging: CUDA error, read training code, fix, re-run."""
        ids = self._conv_ids()

        cuda_err = self._gen_cuda_error()
        training_code = self._gen_ml_training_code()
        training_log = self._gen_ml_training_log()
        inference_code = self._gen_ml_inference_code()

        turns = [
            f"[User]\nI'm getting a CUDA error during training. "
            f"Here's the error:\n\n{cuda_err}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_ANALYZE, ids)}\n\n"
            f"<tool_name>read</tool_name>\n"
            f'<parameter name="file_path">train.py</parameter>\n'
            f"<result>\n{training_code}\n</result>",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_ANALYZE, ids)}\n\n"
            f"<tool_name>read</tool_name>\n"
            f'<parameter name="file_path">inference.py</parameter>\n'
            f"<result>\n{inference_code}\n</result>",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_FIX, ids)}\n\n"
            f"{self._gen_tool_edit(language='python')}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_TEST, ids)}\n\n"
            f"<tool_name>bash</tool_name>\n"
            f'<parameter name="command">python train.py --max-steps 10</parameter>\n'
            f"<result>\n{training_log}\n</result>",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_SUMMARY, ids)}",
            "[User]\nCan you also check if the inference script has the same issue?",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_ANALYZE, ids)}\n\n"
            f"{self._conv_bridge(_BRIDGE_EXPLAIN, ids)}\n\n"
            f"{self._conv_bridge(_BRIDGE_SUMMARY, ids)}",
        ]
        return "\n\n".join(turns)

    def _gen_conv_test_write(self) -> str:
        """Test writing session: read code, write tests, iterate on failures."""
        r = self._template_rng
        lang = r.choice(_LANGUAGES)
        ids = self._conv_ids()

        turns = [
            f"[User]\nWrite comprehensive tests for {ids['cls']}.{ids['method']}(). "
            f"Cover the happy path, edge cases, and error handling.",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_ANALYZE, ids)}\n\n"
            f"{self._gen_tool_read_long(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_ANALYZE, ids)}\n\n"
            f"{self._gen_tool_search(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_WRITE_TEST, ids)}\n\n"
            f"{self._gen_tool_edit(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_TEST, ids)}\n\n"
            f"{self._gen_tool_bash_verbose(language=lang)}",
            f"[User]\n{self._conv_bridge(_FOLLOWUP_QUESTIONS, ids)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_WRITE_TEST, ids)}\n\n"
            f"{self._gen_tool_edit(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_TEST, ids)}\n\n"
            f"{self._gen_tool_bash(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_SUMMARY, ids)}",
        ]
        return "\n\n".join(turns)

    def _gen_conv_migration(self) -> str:
        """Multi-file migration: search all usages, update each file, run tests."""
        r = self._template_rng
        lang = r.choice(_LANGUAGES)
        ids = self._conv_ids()

        turns = [
            f"[User]\nMigrate {ids['cls']}.{ids['method']}() from "
            f"sync to async. It's called across multiple files in {ids['module']}. "
            f"Update all callers and add backward compat.",
            f"[Assistant]\nLet me find all the callers first.\n\n"
            f"{self._gen_tool_search_verbose(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_ANALYZE, ids)}\n\n"
            f"{self._gen_tool_read_long(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_ANALYZE, ids)}\n\n"
            f"{self._gen_tool_read(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_ANALYZE, ids)}\n\n"
            f"{self._gen_tool_read(language=lang)}",
            f"[Assistant]\nI'll start with the core change to {ids['cls']}, "
            f"then update each caller.\n\n"
            f"{self._gen_tool_edit(language=lang)}",
            f"[Assistant]\nNow updating the first caller.\n\n"
            f"{self._gen_tool_edit(language=lang)}",
            f"[Assistant]\nUpdating the second caller.\n\n"
            f"{self._gen_tool_edit(language=lang)}",
            f"[Assistant]\nUpdating the third caller and adding the "
            f"backward-compat wrapper.\n\n"
            f"{self._gen_tool_edit(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_TEST, ids)}\n\n"
            f"{self._gen_tool_bash_verbose(language=lang)}",
            f"[User]\n{self._conv_bridge(_FOLLOWUP_QUESTIONS, ids)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_EXPLAIN, ids)}\n\n"
            f"{self._conv_bridge(_BRIDGE_SUMMARY, ids)}",
        ]
        return "\n\n".join(turns)

    def _gen_conv_deploy(self) -> str:
        """Deployment troubleshooting: check config, logs, fix, verify."""
        r = self._template_rng
        lang = r.choice(_LANGUAGES)
        ids = self._conv_ids()

        config_block = self._gen_config_file(language=lang)
        json_resp = self._gen_json_response(language=lang)

        turns = [
            f"[User]\nThe {ids['module']} service keeps crashing after deploy. "
            f"The health check is failing and pods are in CrashLoopBackOff.",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_DEPLOY, ids)}\n\n"
            f"<tool_name>bash</tool_name>\n"
            f'<parameter name="command">kubectl describe pod {ids["module"]}-'
            f"{r.randint(1000, 9999)}-{r.choice('abcdef')}"
            f"{r.choice('abcdef')}{r.choice('0123456789')}"
            f"{r.choice('abcdef')}{r.choice('0123456789')}</parameter>\n"
            f"<result>\n"
            f"Name:         {ids['module']}-deployment-{r.randint(1000, 9999)}\n"
            f"Namespace:    default\n"
            f"Status:       Running\n"
            f"Containers:\n"
            f"  {ids['module']}:\n"
            f"    Image:          registry.internal/{ids['module']}:latest\n"
            f"    State:          Waiting (CrashLoopBackOff)\n"
            f"    Last State:     Terminated (Error, exit code 1)\n"
            f"    Ready:          False\n"
            f"    Restart Count:  7\n"
            f"    Limits:\n"
            f"      cpu:     2\n"
            f"      memory:  512Mi\n"
            f"    Requests:\n"
            f"      cpu:     500m\n"
            f"      memory:  256Mi\n"
            f"    Liveness:   http-get http://:8080/health delay=10s timeout=3s period=5s\n"
            f"    Readiness:  http-get http://:8080/ready delay=5s timeout=3s period=5s\n"
            f"Events:\n"
            f"  Warning  BackOff  2m (x7 over 10m)  kubelet  "
            f"Back-off restarting failed container\n"
            f"</result>",
            f"[Assistant]\nThe memory limit looks too low. Let me check the config.\n\n"
            f"<tool_name>read</tool_name>\n"
            f'<parameter name="file_path">kubernetes/deployment.yaml</parameter>\n'
            f"<result>\n{config_block}\n</result>",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_DEPLOY, ids)}\n\n"
            f"<tool_name>bash</tool_name>\n"
            f'<parameter name="command">kubectl logs deploy/{ids["module"]} '
            f"--tail=30</parameter>\n"
            f"<result>\n"
            f"{self._gen_error_traceback(language=lang)}\n"
            f"</result>",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_FIX, ids)}\n\n"
            f"{self._gen_tool_edit(language=lang)}",
            f"[Assistant]\nLet me also increase the memory limits.\n\n"
            f"{self._gen_tool_edit(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_TEST, ids)}\n\n"
            f"<tool_name>bash</tool_name>\n"
            f'<parameter name="command">kubectl apply -f kubernetes/deployment.yaml '
            f"&& kubectl rollout status deploy/{ids['module']} --timeout=120s</parameter>\n"
            f"<result>\n"
            f"deployment.apps/{ids['module']} configured\n"
            f'Waiting for deployment "{ids["module"]}" rollout to finish: '
            f"1 old replicas are pending termination...\n"
            f'deployment "{ids["module"]}" successfully rolled out\n'
            f"</result>",
            f"[Assistant]\nLet me verify the health check is passing now.\n\n"
            f"<tool_name>bash</tool_name>\n"
            f'<parameter name="command">curl -s http://localhost:8080/health '
            f"| python -m json.tool</parameter>\n"
            f"<result>\n{json_resp}\n</result>",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_SUMMARY, ids)}",
        ]
        return "\n\n".join(turns)

    def _gen_conv_security(self) -> str:
        """Security vulnerability investigation: find vuln, analyze attack vectors, fix, test."""
        r = self._template_rng
        lang = r.choice(_LANGUAGES)
        ids = self._conv_ids()

        turns = [
            f"[User]\nI think there's a security vulnerability in the {ids['module']} "
            f"service. The {ids['method']}() endpoint accepts user input for {ids['var']} "
            f"without proper validation.",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_SECURITY, ids)}\n\n"
            f"{self._gen_tool_read_long(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_ANALYZE, ids)}\n\n"
            f"{self._gen_tool_search_verbose(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_ARCHITECTURE_TRADEOFF, ids)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_SECURITY, ids)}\n\n"
            f"{self._gen_tool_edit(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_WRITE_TEST, ids)}\n\n"
            f"{self._gen_tool_edit(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_TEST, ids)}\n\n"
            f"{self._gen_tool_bash_verbose(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_SUMMARY, ids)}",
        ]
        return "\n\n".join(turns)

    def _gen_conv_distributed(self) -> str:
        """Distributed systems debugging: inconsistency, analyze replication, fix consensus."""
        r = self._template_rng
        lang = r.choice(_LANGUAGES)
        ids = self._conv_ids()

        config_block = self._gen_config_file(language=lang)

        turns = [
            f"[User]\nThere are inconsistent reads across replicas in the "
            f"{ids['module']} service. After writing to {ids['var']} via "
            f"{ids['cls']}.{ids['method']}(), some replicas return stale data.",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_DISTRIBUTED, ids)}\n\n"
            f"<tool_name>read</tool_name>\n"
            f'<parameter name="file_path">config/replication.yaml</parameter>\n'
            f"<result>\n{config_block}\n</result>",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_ANALYZE, ids)}\n\n"
            f"{self._gen_tool_search_verbose(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_ARCHITECTURE_TRADEOFF, ids)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_DISTRIBUTED, ids)}\n\n"
            f"{self._gen_tool_edit(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_TEST, ids)}\n\n"
            f"{self._gen_tool_bash_verbose(language=lang)}",
            f"[User]\n{self._conv_bridge(_FOLLOWUP_QUESTIONS, ids)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_DISTRIBUTED, ids)}\n\n"
            f"{self._conv_bridge(_BRIDGE_SUMMARY, ids)}",
        ]
        return "\n\n".join(turns)

    def _gen_conv_observability(self) -> str:
        """Observability gap: add tracing, metrics, structured logging."""
        r = self._template_rng
        lang = r.choice(_LANGUAGES)
        ids = self._conv_ids()

        config_block = self._gen_config_file(language=lang)

        turns = [
            f"[User]\nCan't debug a production latency spike in {ids['module']}. "
            f"There's no tracing or metrics on {ids['cls']}.{ids['method']}().",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_ANALYZE, ids)}\n\n"
            f"{self._gen_tool_read_long(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_OBSERVABILITY, ids)}\n\n"
            f"{self._gen_tool_search_verbose(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_OBSERVABILITY, ids)}\n\n"
            f"{self._gen_tool_edit(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_OBSERVABILITY, ids)}\n\n"
            f"{self._gen_tool_edit(language=lang)}",
            f"[Assistant]\nLet me also add the telemetry configuration.\n\n"
            f"<tool_name>read</tool_name>\n"
            f'<parameter name="file_path">config/telemetry.yaml</parameter>\n'
            f"<result>\n{config_block}\n</result>",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_TEST, ids)}\n\n"
            f"<tool_name>bash</tool_name>\n"
            f'<parameter name="command">curl -s http://localhost:8080/metrics '
            f"| head -20</parameter>\n"
            f"<result>\n{self._gen_json_response(language=lang)}\n</result>",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_SUMMARY, ids)}",
        ]
        return "\n\n".join(turns)

    def _gen_conv_db_optimize(self) -> str:
        """Database optimization: EXPLAIN, read ORM code, add index, benchmark."""
        r = self._template_rng
        lang = r.choice(_LANGUAGES)
        ids = self._conv_ids()

        table = r.choice(_DB_TABLES)
        sql_block = self._gen_sql_query()

        turns = [
            f"[User]\nThe {ids['method']}() query on the {table} table is taking "
            f"over 5 seconds in production. Can you optimize it?",
            f"[Assistant]\nLet me run EXPLAIN ANALYZE to see the query plan.\n\n"
            f"<tool_name>bash</tool_name>\n"
            f'<parameter name="command">psql -d mydb -c "EXPLAIN ANALYZE '
            f"SELECT * FROM {table} WHERE {ids['var']} = 'test'\"</parameter>\n"
            f"<result>\n{sql_block}\n</result>",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_DATA_ARCHITECTURE, ids)}\n\n"
            f"{self._gen_tool_read_long(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_ARCHITECTURE_TRADEOFF, ids)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_DATA_ARCHITECTURE, ids)}\n\n"
            f"{self._gen_tool_edit(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_TEST, ids)}\n\n"
            f"{self._gen_tool_bash_verbose(language=lang)}",
            f"[User]\nShould we also partition the {table} table?",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_ARCHITECTURE_TRADEOFF, ids)}\n\n"
            f"{self._conv_bridge(_BRIDGE_SUMMARY, ids)}",
        ]
        return "\n\n".join(turns)

    def _gen_conv_architecture_review(self) -> str:
        """Architecture review: read multiple files, deep multi-paragraph analysis, refactor."""
        r = self._template_rng
        lang = r.choice(_LANGUAGES)
        ids = self._conv_ids()

        turns = [
            f"[User]\nCan you do an architecture review of the {ids['module']} "
            f"service? I'm concerned about coupling and scalability.",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_ANALYZE, ids)}\n\n"
            f"{self._gen_tool_read_long(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_ANALYZE, ids)}\n\n"
            f"{self._gen_tool_read_long(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_ANALYZE, ids)}\n\n"
            f"{self._gen_tool_search_verbose(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_ARCHITECTURE_TRADEOFF, ids)}\n\n"
            f"{self._conv_bridge(_BRIDGE_ARCHITECTURE_TRADEOFF, ids)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_REFACTOR, ids)}\n\n"
            f"{self._gen_tool_edit(language=lang)}",
            f"[User]\nWhat about the scalability of {ids['cls']}? Will this "
            f"approach hold up under 10x traffic?",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_ARCHITECTURE_TRADEOFF, ids)}\n\n"
            f"{self._conv_bridge(_BRIDGE_PERF, ids)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_FIX, ids)}\n\n"
            f"{self._gen_tool_edit(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_TEST, ids)}\n\n"
            f"{self._gen_tool_bash_verbose(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_SUMMARY, ids)}",
        ]
        return "\n\n".join(turns)

    def _gen_conv_incident_response(self) -> str:
        """Production incident: cascading failure, diagnose, fix, add monitoring, post-mortem."""
        r = self._template_rng
        lang = r.choice(_LANGUAGES)
        ids = self._conv_ids()

        config_block = self._gen_config_file(language=lang)
        error_block = self._gen_error_traceback(language=lang)

        turns = [
            f"[User]\nProduction incident: the {ids['module']} service is down "
            f"and it's causing cascading failures in downstream services.",
            "[Assistant]\nLet me check the service health immediately.\n\n"
            "<tool_name>bash</tool_name>\n"
            '<parameter name="command">curl -s http://localhost:8080/health '
            "|| echo 'Connection refused'</parameter>\n"
            "<result>\nConnection refused\n</result>",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_DEPLOY, ids)}\n\n"
            f"<tool_name>read</tool_name>\n"
            f'<parameter name="file_path">kubernetes/deployment.yaml</parameter>\n'
            f"<result>\n{config_block}\n</result>",
            f"[Assistant]\nLet me check the logs for the root cause.\n\n"
            f"<tool_name>bash</tool_name>\n"
            f'<parameter name="command">kubectl logs deploy/{ids["module"]} '
            f"--tail=50</parameter>\n"
            f"<result>\n{error_block}\n</result>",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_ARCHITECTURE_TRADEOFF, ids)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_FIX, ids)}\n\n"
            f"{self._gen_tool_edit(language=lang)}",
            f"[Assistant]\nNow let me add a circuit breaker to prevent cascading failures.\n\n"
            f"{self._gen_tool_edit(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_OBSERVABILITY, ids)}\n\n"
            f"{self._gen_tool_edit(language=lang)}",
            f"[Assistant]\n{self._conv_bridge(_BRIDGE_TEST, ids)}\n\n"
            f"{self._gen_tool_bash_verbose(language=lang)}",
            f"[Assistant]\nPost-mortem summary: The {ids['module']} service experienced "
            f"a cascading failure triggered by {ids['error']}. The root cause was "
            f"{ids['cls']}.{ids['method']}() not handling the error gracefully, which "
            f"caused the health check to fail and pods to restart in a loop. "
            f"Fixes applied: error handling in {ids['method']}(), circuit breaker "
            f"pattern for downstream calls, and Prometheus alerts for early detection.",
        ]
        return "\n\n".join(turns)


class _SafeFormatMap(dict):
    """Dict subclass that returns '{key}' for missing keys in str.format_map."""

    def __missing__(self, key: str) -> str:
        return f"{{{key}}}"
