# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Higher-level conversation generators: migration, deploy, security, distributed, etc.

Extracted from ``coding_content.py`` to keep that module under the
ergonomics file-size cap. Methods read ``self._template_rng`` and the
shared vocabulary tuples; behavior is unchanged.
"""

from __future__ import annotations

from aiperf.dataset.generator._coding_text import (
    _BRIDGE_ANALYZE,
    _BRIDGE_ARCHITECTURE_TRADEOFF,
    _BRIDGE_DATA_ARCHITECTURE,
    _BRIDGE_DEPLOY,
    _BRIDGE_DISTRIBUTED,
    _BRIDGE_EXPLAIN,
    _BRIDGE_FIX,
    _BRIDGE_OBSERVABILITY,
    _BRIDGE_PERF,
    _BRIDGE_REFACTOR,
    _BRIDGE_SECURITY,
    _BRIDGE_SUMMARY,
    _BRIDGE_TEST,
    _BRIDGE_WRITE_TEST,
    _FOLLOWUP_QUESTIONS,
    _LANGUAGES,
)
from aiperf.dataset.generator._coding_vocab import (
    _DB_TABLES,
)


class _ConversationsAdvancedMixin:
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
