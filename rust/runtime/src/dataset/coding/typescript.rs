// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! TypeScript code-template renderers.

use super::CodingCorpusError;
use super::templates::TemplateRenderer;
use super::vocab::*;

/// Dispatch across the TypeScript structural variants.
pub(super) fn render(r: &mut TemplateRenderer) -> Result<String, CodingCorpusError> {
    match r.index(4)? {
        0 => typescript_class(r),
        1 => typescript_http_handler(r),
        2 => typescript_types(r),
        _ => typescript_test(r),
    }
}

fn typescript_class(r: &mut TemplateRenderer) -> Result<String, CodingCorpusError> {
    let imp = r.pick(TS_IMPORTS)?;
    let imp_cls = r.pick(CLASSES)?;
    let cls = r.pick(CLASSES)?;
    let m = r.sample(METHODS, 3)?;
    let (m1, m2, m3) = (m[0], m[1], m[2]);
    let v = r.sample(VARS, 3)?;
    let (v1, v2, v3) = (v[0], v[1], v[2]);
    let err = r.pick(ERRORS)?;

    Ok(format!(
        r##"import {{{{ {imp_cls} }}}} from '{imp}';

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
"##
    ))
}

fn typescript_http_handler(r: &mut TemplateRenderer) -> Result<String, CodingCorpusError> {
    let cls = r.pick(CLASSES)?;
    let m = r.sample(METHODS, 2)?;
    let (m1, m2) = (m[0], m[1]);
    let v = r.sample(VARS, 2)?;
    let (v1, v2) = (v[0], v[1]);
    let route = r.pick(ROUTES)?;
    let err = r.pick(ERRORS)?;
    let cls_lower = cls.to_lowercase();
    let m1_title = TemplateRenderer::title_case(m1);

    Ok(format!(
        r##"import {{ Hono }} from 'hono';
import {{ z }} from 'zod';
import {{ {cls} }} from './{cls_lower}';

const {m1}Schema = z.object({{{{
  {v1}: z.string().min(1).max(256),
  {v2}: z.number().int().positive().optional(),
}}}});

type {m1_title}Input = z.infer<typeof {m1}Schema>;

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
"##
    ))
}

fn typescript_types(r: &mut TemplateRenderer) -> Result<String, CodingCorpusError> {
    let cls = r.pick(CLASSES)?;
    let v = r.sample(VARS, 3)?;
    let (v1, v2, v3) = (v[0], v[1], v[2]);
    let m = r.sample(METHODS, 2)?;
    let (m1, m2) = (m[0], m[1]);
    let err = r.pick(ERRORS)?;
    let m1_title = TemplateRenderer::title_case(m1);

    Ok(format!(
        r##"export type {cls}Status = 'pending' | 'active' | 'failed' | 'completed';

export interface {cls}Event {{{{
  kind: '{m1}' | '{m2}' | 'error';
  {v1}: string;
  timestamp: number;
}}}}

export type {m1_title}Event = Extract<{cls}Event, {{{{ kind: '{m1}' }}}}>;
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
"##
    ))
}

fn typescript_test(r: &mut TemplateRenderer) -> Result<String, CodingCorpusError> {
    let cls = r.pick(CLASSES)?;
    let m = r.sample(METHODS, 3)?;
    let (m1, m2, m3) = (m[0], m[1], m[2]);
    let v = r.sample(VARS, 2)?;
    let (v1, v2) = (v[0], v[1]);
    let err = r.pick(ERRORS)?;
    let mod_ = r.pick(MODULES)?;

    Ok(format!(
        r##"import {{ describe, it, expect, beforeEach, vi }} from 'vitest';
import {{ {cls} }} from '../{mod_}';

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
"##
    ))
}
