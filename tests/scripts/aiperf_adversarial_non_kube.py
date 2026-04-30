#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Literal

BASE = Path(__file__).resolve().parents[2]
URL = os.environ.get('AIPERF_ADVERSARIAL_URL', 'http://127.0.0.1:36037')
ROOT = Path('/tmp/aiperf-non-kube-adversarial') / time.strftime('%Y%m%d-%H%M%S')
LOGS = ROOT / 'logs'; ART = ROOT / 'artifacts'; FIX = ROOT / 'fixtures'
for p in (LOGS, ART, FIX):
    p.mkdir(parents=True, exist_ok=True)
ENV = os.environ.copy()
ENV.update({'PYTHONUNBUFFERED': '1', 'NO_PROXY': '127.0.0.1,localhost', 'no_proxy': '127.0.0.1,localhost'})

single = FIX / 'single_turn.jsonl'
single.write_text(json.dumps({'text': 'hello', 'output_length': 4}) + '\n')
mooncake = FIX / 'mooncake.jsonl'
mooncake.write_text(json.dumps({'timestamp': 0, 'text_input': 'hello', 'output_length': 4}) + '\n')
invalid_jsonl = FIX / 'invalid.jsonl'
invalid_jsonl.write_text('{not json}\n')
random_pool = FIX / 'random_pool'
random_pool.mkdir()
(random_pool / 'prompt.txt').write_text('hello random pool\n')

base = ['uv','run','aiperf','profile','--model','mock-model','--url',URL,'--tokenizer','builtin','--ui','none','--request-timeout-seconds','10','--ready-check-timeout','0','--workers-max','1','--no-gpu-telemetry']

def prof(name: str, *args: str, timeout: int = 90) -> tuple[str, list[str], int, Literal['pass','fail']]:
    return name, base + ['--artifact-dir', str(ART / name)] + list(args), timeout, 'fail'

def prof_pass(name: str, *args: str, timeout: int = 120) -> tuple[str, list[str], int, Literal['pass','fail']]:
    return name, base + ['--artifact-dir', str(ART / name)] + list(args), timeout, 'pass'

cases: list[tuple[str, list[str], int, Literal['pass','fail']]] = [
    # Mutually exclusive / duplicate CLI forms should fail at parse or config validation.
    prof('fixed-auto-and-start-offset', '--endpoint-type','chat','--input-file',str(mooncake),'--custom-dataset-type','mooncake-trace','--fixed-schedule','--fixed-schedule-auto-offset','--fixed-schedule-start-offset','0'),
    prof('fixed-end-before-start', '--endpoint-type','chat','--input-file',str(mooncake),'--custom-dataset-type','mooncake-trace','--fixed-schedule','--fixed-schedule-start-offset','100','--fixed-schedule-end-offset','0'),
    prof('public-and-custom-dataset', '--endpoint-type','chat','--public-dataset','sharegpt','--input-file',str(single),'--custom-dataset-type','single-turn'),
    prof('custom-type-without-input-file', '--endpoint-type','chat','--custom-dataset-type','single-turn'),
    prof_pass('input-file-without-custom-type', '--endpoint-type','chat','--input-file',str(single)),
    prof('bad-jsonl-single-turn', '--endpoint-type','chat','--input-file',str(invalid_jsonl),'--custom-dataset-type','single-turn'),
    prof('nontext-tokenizer-on-image-retrieval', '--endpoint-type','image-retrieval','--tokenizer','builtin','--image-batch-size','1'),
    prof_pass('streaming-embeddings', '--endpoint-type','embeddings','--streaming','--batch-size','2'),
    prof_pass('streaming-rankings', '--endpoint-type','nim-rankings','--streaming','--rankings-passages-mean','2'),
    prof_pass('server-token-count-embeddings', '--endpoint-type','embeddings','--use-server-token-count'),
    prof('negative-request-count', '--endpoint-type','chat','--request-count','-1','--concurrency','1'),
    prof('zero-concurrency', '--endpoint-type','chat','--request-count','1','--concurrency','0'),
    prof('negative-concurrency', '--endpoint-type','chat','--request-count','1','--concurrency','-1'),
    prof('zero-request-rate', '--endpoint-type','chat','--request-count','1','--request-rate','0'),
    prof('negative-request-rate', '--endpoint-type','chat','--request-count','1','--request-rate','-1'),
    prof_pass('both-concurrency-and-request-rate', '--endpoint-type','chat','--request-count','2','--concurrency','1','--request-rate','1'),
    prof_pass('duration-and-request-count', '--endpoint-type','chat','--request-count','2','--benchmark-duration','1'),
    prof_pass('bad-arrival-pattern-for-concurrency', '--endpoint-type','chat','--request-count','1','--concurrency','1','--arrival-pattern','gamma'),
    prof('gamma-without-request-rate', '--endpoint-type','chat','--request-count','1','--arrival-pattern','gamma','--arrival-smoothness','0.5'),
    prof('bad-arrival-smoothness-negative', '--endpoint-type','chat','--request-count','1','--request-rate','1','--arrival-pattern','gamma','--arrival-smoothness','-0.1'),
    prof_pass('bad-arrival-smoothness-large', '--endpoint-type','chat','--request-count','1','--request-rate','1','--arrival-pattern','gamma','--arrival-smoothness','999'),
    prof('bad-seq-dist-syntax', '--endpoint-type','chat','--request-count','1','--concurrency','1','--seq-dist','not-a-dist'),
    prof('bad-extra-inputs', '--endpoint-type','chat','--request-count','1','--concurrency','1','--extra-inputs','notkv'),
    prof('bad-header', '--endpoint-type','chat','--request-count','1','--concurrency','1','--header','NoColonHeader'),
    prof_pass('http2-transport-local-mock', '--endpoint-type','chat','--request-count','1','--concurrency','1','--transport','http2'),
    prof('invalid-url-scheme', '--endpoint-type','chat','--request-count','1','--concurrency','1','--url','ftp://127.0.0.1:36037'),
    prof('custom-endpoint-missing-leading-slash', '--endpoint-type','chat','--request-count','1','--concurrency','1','--custom-endpoint','v1/chat/completions'),
    prof('template-endpoint-without-template', '--endpoint-type','template','--request-count','1','--concurrency','1','--custom-endpoint','/v1/custom-multimodal'),
    prof_pass('gpu-telemetry-and-no-gpu-telemetry', '--endpoint-type','chat','--request-count','1','--concurrency','1','--gpu-telemetry',f'{URL}/dcgm1/metrics'),
    prof_pass('server-metrics-and-no-server-metrics', '--endpoint-type','chat','--request-count','1','--concurrency','1','--server-metrics',f'{URL}/metrics','--no-server-metrics'),
    prof('bad-server-metrics-format', '--endpoint-type','chat','--request-count','1','--concurrency','1','--server-metrics',f'{URL}/metrics','--server-metrics-formats','bogus'),
    prof('bad-export-level', '--endpoint-type','chat','--request-count','1','--concurrency','1','--export-level','everything'),
    prof('negative-slice-duration', '--endpoint-type','chat','--request-count','1','--concurrency','1','--slice-duration','-1'),
    prof('bad-model-selection-weighted-no-weights', '--endpoint-type','chat','--model','m1,m2','--model-selection-strategy','weighted','--request-count','1','--concurrency','1'),
    prof('bad-url-strategy', '--endpoint-type','chat','--request-count','1','--concurrency','1','--url-strategy','random'),
    prof('bad-connection-reuse-strategy', '--endpoint-type','chat','--request-count','1','--concurrency','1','--connection-reuse-strategy','forever'),
    prof('warmup-negative', '--endpoint-type','chat','--request-count','1','--concurrency','1','--warmup-request-count','-1'),
    prof('cancellation-rate-over-100', '--endpoint-type','chat','--request-count','1','--concurrency','1','--streaming','--request-cancellation-rate','101','--request-cancellation-delay','0'),
    prof('cancellation-negative-delay', '--endpoint-type','chat','--request-count','1','--concurrency','1','--streaming','--request-cancellation-rate','50','--request-cancellation-delay','-1'),
    prof('prefix-mutual-exclusion', '--endpoint-type','chat','--request-count','1','--concurrency','1','--num-prefix-prompts','2','--prefix-prompt-length','8','--shared-system-prompt-length','4'),
    prof('random-pool-with-batch-size', '--endpoint-type','chat','--request-count','1','--concurrency','1','--input-file',str(random_pool),'--custom-dataset-type','random-pool','--batch-size','2'),
    # Mock-server edge cases should either pass or fail gracefully without traceback markers.
    prof_pass('streaming-unsupported-chat-embeddings-downshift', '--endpoint-type','chat-embeddings','--custom-endpoint','/v1/chat/embeddings','--request-count','1','--concurrency','1','--streaming','--batch-size','2'),
    prof_pass('multiple-models-round-robin', '--endpoint-type','chat','--model','m1,m2','--request-count','2','--concurrency','1','--model-selection-strategy','round-robin'),
    prof_pass('multi-url-round-robin', '--endpoint-type','chat','--url',URL,'--request-count','2','--concurrency','1','--url-strategy','round_robin'),
]

crash_markers = ['Traceback (most recent call last)', 'TypeError:', 'AttributeError:', 'KeyError:', 'IndexError:', 'AssertionError:', 'RuntimeError:']
results = []
print(f'ADVERSARIAL_ROOT={ROOT}', flush=True)
for idx, (name, cmd, timeout, expected) in enumerate(cases, 1):
    log = LOGS / f'{idx:03d}-{name}.log'
    print(f'[{idx:03d}/{len(cases):03d}] RUN {name} expect={expected}: {shlex.join(cmd)}', flush=True)
    started = time.monotonic()
    with log.open('w') as out:
        out.write(f'$ {shlex.join(cmd)}\n\n')
        try:
            proc = subprocess.run(cmd, cwd=BASE, env=ENV, stdout=out, stderr=subprocess.STDOUT, timeout=timeout, text=True)
            rc = proc.returncode
            status = 'PASS' if rc == 0 else 'FAIL'
        except subprocess.TimeoutExpired:
            rc = 124; status = 'TIMEOUT'; out.write(f'\nTIMEOUT after {timeout}s\n')
    text = log.read_text(errors='replace')
    crashed = any(marker in text for marker in crash_markers)
    if expected == 'fail':
        verdict = 'OK_EXPECTED_FAILURE' if rc != 0 and not crashed else ('BUG_CRASH' if crashed else 'BUG_UNEXPECTED_PASS')
    else:
        verdict = 'OK_EXPECTED_PASS' if rc == 0 and not crashed else ('BUG_CRASH' if crashed else 'BUG_UNEXPECTED_FAIL')
    elapsed = time.monotonic() - started
    print(f'[{idx:03d}/{len(cases):03d}] {verdict} rc={rc} elapsed={elapsed:.1f}s log={log}', flush=True)
    results.append({'name': name, 'cmd': cmd, 'expected': expected, 'rc': rc, 'status': status, 'verdict': verdict, 'crashed': crashed, 'elapsed_seconds': elapsed, 'log': str(log)})
summary = ROOT / 'summary.json'
summary.write_text(json.dumps({'root': str(ROOT), 'url': URL, 'results': results}, indent=2))
bugs = [r for r in results if r['verdict'].startswith('BUG')]
print(f'SUMMARY={summary}', flush=True)
print(f'OK={len(results)-len(bugs)} BUGS={len(bugs)} TOTAL={len(results)}', flush=True)
if bugs:
    print('BUG_NAMES=' + ','.join(r['name'] + ':' + r['verdict'] for r in bugs), flush=True)
    raise SystemExit(1)
