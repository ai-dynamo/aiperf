# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TypeScript code-template generators (mixin for CodingContentGenerator).

Extracted from ``coding_content.py`` to keep that module under the
ergonomics file-size cap. Methods read ``self._template_rng`` and the
shared vocabulary tuples; behavior is unchanged.
"""

from __future__ import annotations

from aiperf.dataset.generator._coding_vocab import (
    _CLASSES,
    _ERROR_MESSAGES,
    _HTTP_ROUTES,
    _METHODS,
    _MODULES,
    _TS_IMPORTS,
    _VARS,
)


class _TypeScriptMixin:
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
