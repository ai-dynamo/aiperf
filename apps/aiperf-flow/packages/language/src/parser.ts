// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
  Diagnostic,
  Result,
  SourcePosition,
  SourceRange,
} from "@aiperf/flow-schema";
import {
  EmbeddedActionsParser,
  type ILexingError,
  type IRecognitionException,
  type IToken,
} from "chevrotain";

import type {
  CameraAst,
  CameraKeyframeAst,
  ComparisonOperator,
  ComponentInvocationAst,
  ConnectorAst,
  DocumentAst,
  ExplainerAst,
  FallbackAst,
  ForLoopAst,
  IdentifierReferenceAst,
  ImportDeclarationAst,
  InteractionActionAst,
  InteractionAst,
  InteractionEventAst,
  LanguageDeclarationAst,
  LiteralAst,
  NarrationAst,
  ObjectLiteralAst,
  ObjectPropertyAst,
  ParamDeclarationAst,
  PropAssignmentAst,
  RectAst,
  ReadingOrderAst,
  ReferenceListAst,
  RenderDeclarationAst,
  RequirementAst,
  ResponsiveAst,
  ResponsiveConditionAst,
  ResponsiveOverrideAst,
  SceneAst,
  ScenePrimitiveAst,
  ScenePrimitiveKeyword,
  SlideAst,
  SlotBlockAst,
  SummaryAst,
  SymbolDefinitionAst,
  SymbolBodyStatementAst,
  ThemeAssignmentAst,
  ThemeDeclarationAst,
  ThemeFontLiteralAst,
  ThemeRoleReferenceAst,
  TimelineAction,
  TimelineAst,
  TimelineCueAst,
  TimelineCueEasing,
  TokenDeclarationAst,
  TokenReferenceAst,
  TypeRefAst,
  UseThemeAst,
  ValueAst,
} from "./ast.js";
import { SCENE_PRIMITIVE_CAPABILITIES } from "./ast.js";
import {
  parseExplainerBlock,
  type ExplainerAstCompat,
  type TokenStream,
} from "./grammar/explainer.js";
import {
  allTokens,
  Arrow,
  As,
  At,
  Axis,
  Bracket,
  Callout,
  Camera,
  Caption,
  Center,
  Children,
  Circle,
  Colon,
  ColorKind,
  Cols,
  Comma,
  ComponentIdentifier,
  Connector,
  Description,
  Detail,
  Direction,
  Do,
  Dot,
  Duration,
  DurationLiteral,
  EaseIn,
  EaseInOut,
  EaseOut,
  Easing,
  Elbow,
  Ellipse,
  EnterChildren,
  EqualEqual,
  EnumKind,
  Equals,
  Exit,
  Explainer,
  Extends,
  Fade,
  False,
  Fallback,
  Fill,
  FontKind,
  Flow,
  For,
  Frame,
  From,
  Gap,
  Greater,
  GreaterEqual,
  Grid,
  Header,
  Height,
  Identifier,
  Import,
  In,
  Inspect,
  Interaction,
  Label,
  Language,
  LBrace,
  LBracket,
  Less,
  LessEqual,
  Linear,
  LParen,
  Narrate,
  NotEqual,
  NumberKind,
  NumberLiteral,
  On,
  Pad,
  Panel,
  QuotedString,
  ReadingOrder,
  Rect,
  Require,
  Responsive,
  Reveal,
  RBrace,
  RBracket,
  Role,
  RParen,
  Scene,
  Select,
  Set,
  Signal,
  Stack,
  Stagger,
  Step,
  Stroke,
  Summary,
  SymbolKeyword,
  Targets,
  Theme,
  Timeline,
  Title,
  To,
  Token,
  Trace,
  True,
  Use,
  Via,
  When,
  Width,
  X,
  Y,
  Zoom,
  flowLexer,
} from "./tokens.js";
function position(
  offset: number | undefined,
  line: number | undefined,
  column: number | undefined,
): SourcePosition {
  return {
    offset: Number.isFinite(offset) ? (offset ?? 0) : 0,
    line: Number.isFinite(line) ? (line ?? 1) : 1,
    column: Number.isFinite(column) ? (column ?? 1) : 1,
  };
}

function tokenRange(sourceName: string, token: IToken): SourceRange {
  const start = position(token.startOffset, token.startLine, token.startColumn);
  return {
    source: sourceName,
    start,
    end: position(
      (token.endOffset ?? start.offset - 1) + 1,
      token.endLine,
      (token.endColumn ?? start.column - 1) + 1,
    ),
  };
}

function rangeBetween(
  sourceName: string,
  startToken: IToken,
  endToken: IToken,
): SourceRange {
  return {
    source: sourceName,
    start: tokenRange(sourceName, startToken).start,
    end: tokenRange(sourceName, endToken).end,
  };
}

function stringValue(token: IToken): string {
  return JSON.parse(token.image) as string;
}

function numberValue(token: IToken): number {
  return Number(token.image);
}

function literal(
  sourceName: string,
  token: IToken,
  value: string | number | boolean,
): LiteralAst {
  return { kind: "literal", value, sourceMap: tokenRange(sourceName, token) };
}

class FlowParser extends EmbeddedActionsParser {
  public constructor(private readonly sourceName: string) {
    super(allTokens, { recoveryEnabled: true, maxLookahead: 3 });
    this.performSelfAnalysis();
  }

  public readonly document = this.RULE("document", (): DocumentAst => {
    const start = this.CONSUME(Flow);
    const titleToken = this.CONSUME(QuotedString);
    this.CONSUME(As);
    const idToken = this.CONSUME(Identifier);
    this.CONSUME(LBrace);
    const language = this.SUBRULE(this.languageDeclaration);
    const imports: ImportDeclarationAst[] = [];
    const requirements: RequirementAst[] = [];
    const tokens: TokenDeclarationAst[] = [];
    const themes: ThemeDeclarationAst[] = [];
    let useTheme: UseThemeAst | undefined;
    const symbols: SymbolDefinitionAst[] = [];
    const scenes: SceneAst[] = [];
    this.MANY(() =>
      this.OR([
        {
          ALT: () => {
            const declaration = this.SUBRULE(this.importDeclaration);
            this.ACTION(() => imports.push(declaration));
          },
        },
        {
          ALT: () => {
            const requirement = this.SUBRULE(this.requirement);
            this.ACTION(() => requirements.push(requirement));
          },
        },
        {
          ALT: () => {
            const token = this.SUBRULE(this.tokenDeclaration);
            this.ACTION(() => tokens.push(token));
          },
        },
        {
          ALT: () => {
            const theme = this.SUBRULE(this.themeDeclaration);
            this.ACTION(() => themes.push(theme));
          },
        },
        {
          ALT: () => {
            const selection = this.SUBRULE(this.useThemeDeclaration);
            this.ACTION(() => {
              useTheme = selection;
            });
          },
        },
        {
          ALT: () => {
            const symbol = this.SUBRULE(this.symbolDefinition);
            this.ACTION(() => symbols.push(symbol));
          },
        },
        {
          ALT: () => {
            const scene = this.SUBRULE(this.scene);
            this.ACTION(() => scenes.push(scene));
          },
        },
      ]),
    );
    const end = this.CONSUME(RBrace);
    return this.ACTION(() => ({
      kind: "document",
      title: stringValue(titleToken),
      id: idToken.image,
      language,
      imports,
      requirements,
      tokens,
      themes,
      ...(useTheme === undefined ? {} : { useTheme }),
      symbols,
      scenes,
      sourceMap: rangeBetween(this.sourceName, start, end),
    }));
  });

  private readonly languageDeclaration = this.RULE(
    "languageDeclaration",
    (): LanguageDeclarationAst => {
      const start = this.CONSUME(Language);
      const version = this.CONSUME(NumberLiteral);
      return this.ACTION(() => ({
        kind: "language",
        version: numberValue(version),
        sourceMap: rangeBetween(this.sourceName, start, version),
      }));
    },
  );

  private readonly importDeclaration = this.RULE(
    "importDeclaration",
    (): ImportDeclarationAst => {
      const start = this.CONSUME(Import);
      const path = this.CONSUME(QuotedString);
      this.CONSUME(As);
      const alias = this.CONSUME(Identifier);
      return this.ACTION(() => ({
        kind: "import",
        path: stringValue(path),
        alias: alias.image,
        sourceMap: rangeBetween(this.sourceName, start, alias),
      }));
    },
  );

  private readonly requirement = this.RULE(
    "requirement",
    (): RequirementAst => {
      const start = this.CONSUME(Require);
      const capability = this.SUBRULE(this.qualifiedName);
      const versionRange = this.CONSUME(QuotedString);
      return this.ACTION(() => ({
        kind: "requirement",
        capability,
        versionRange: stringValue(versionRange),
        sourceMap: rangeBetween(this.sourceName, start, versionRange),
      }));
    },
  );

  private readonly qualifiedName = this.RULE("qualifiedName", (): string => {
    const parts: string[] = [];
    const first = this.CONSUME(Identifier);
    this.ACTION(() => parts.push(first.image));
    this.AT_LEAST_ONE(() => {
      this.CONSUME(Dot);
      const part = this.CONSUME2(Identifier);
      this.ACTION(() => parts.push(part.image));
    });
    return this.ACTION(() => parts.join("."));
  });

  private readonly tokenDeclaration = this.RULE(
    "tokenDeclaration",
    (): TokenDeclarationAst => {
      const start = this.CONSUME(Token);
      const id = this.CONSUME(Identifier);
      this.CONSUME(Equals);
      const valueToken = this.CONSUME(QuotedString);
      return this.ACTION(() => ({
        kind: "token",
        id: id.image,
        value: literal(
          this.sourceName,
          valueToken,
          stringValue(valueToken),
        ),
        sourceMap: rangeBetween(this.sourceName, start, valueToken),
      }));
    },
  );

  private readonly themeDeclaration = this.RULE(
    "themeDeclaration",
    (): ThemeDeclarationAst => {
      const start = this.CONSUME(Theme);
      const id = this.CONSUME(Identifier);
      this.CONSUME(Extends);
      const parent = this.CONSUME2(Identifier);
      this.CONSUME(LBrace);
      const assignments: ThemeAssignmentAst[] = [];
      this.MANY(() => {
        const assignment = this.SUBRULE(this.themeAssignment);
        this.ACTION(() => assignments.push(assignment));
      });
      const end = this.CONSUME(RBrace);
      return this.ACTION(() => ({
        kind: "theme-declaration",
        id: id.image,
        extends: parent.image,
        assignments,
        sourceMap: rangeBetween(this.sourceName, start, end),
      }));
    },
  );

  private readonly themeAssignment = this.RULE(
    "themeAssignment",
    (): ThemeAssignmentAst =>
      this.OR([
        {
          ALT: () => {
            const start = this.CONSUME(ColorKind);
            const role = this.SUBRULE(this.qualifiedName);
            this.CONSUME(Equals);
            const value = this.CONSUME(QuotedString);
            return this.ACTION(() => ({
              kind: "theme-assignment",
              valueKind: "color" as const,
              role,
              value: literal(this.sourceName, value, stringValue(value)),
              sourceMap: rangeBetween(this.sourceName, start, value),
            }));
          },
        },
        {
          ALT: () => {
            const start = this.CONSUME(NumberKind);
            const role = this.SUBRULE2(this.qualifiedName);
            this.CONSUME2(Equals);
            const value = this.CONSUME(NumberLiteral);
            return this.ACTION(() => ({
              kind: "theme-assignment",
              valueKind: "number" as const,
              role,
              value: literal(this.sourceName, value, numberValue(value)),
              sourceMap: rangeBetween(this.sourceName, start, value),
            }));
          },
        },
        {
          ALT: () => {
            const start = this.CONSUME(Duration);
            const role = this.SUBRULE3(this.qualifiedName);
            this.CONSUME3(Equals);
            const value = this.CONSUME(DurationLiteral);
            return this.ACTION(() => ({
              kind: "theme-assignment",
              valueKind: "duration" as const,
              role,
              value: literal(
                this.sourceName,
                value,
                Number(value.image.slice(0, -2)),
              ),
              sourceMap: rangeBetween(this.sourceName, start, value),
            }));
          },
        },
        {
          ALT: () => {
            const start = this.CONSUME(FontKind);
            const role = this.SUBRULE4(this.qualifiedName);
            this.CONSUME4(Equals);
            const value = this.SUBRULE(this.themeFontLiteral);
            return this.ACTION(() => ({
              kind: "theme-assignment",
              valueKind: "font" as const,
              role,
              value,
              sourceMap: {
                source: this.sourceName,
                start: tokenRange(this.sourceName, start).start,
                end: value.sourceMap.end,
              },
            }));
          },
        },
        {
          ALT: () => {
            const start = this.CONSUME(EnumKind);
            const role = this.SUBRULE5(this.qualifiedName);
            this.CONSUME5(Equals);
            const value = this.CONSUME2(QuotedString);
            return this.ACTION(() => ({
              kind: "theme-assignment",
              valueKind: "enum" as const,
              role,
              value: literal(this.sourceName, value, stringValue(value)),
              sourceMap: rangeBetween(this.sourceName, start, value),
            }));
          },
        },
      ]),
  );

  private readonly themeFontLiteral = this.RULE(
    "themeFontLiteral",
    (): ThemeFontLiteralAst => {
      const start = this.CONSUME(LBracket);
      const families: string[] = [];
      this.AT_LEAST_ONE_SEP({
        SEP: Comma,
        DEF: () => {
          const family = this.CONSUME(QuotedString);
          this.ACTION(() => families.push(stringValue(family)));
        },
      });
      const end = this.CONSUME(RBracket);
      return this.ACTION(() => ({
        kind: "theme-font-literal",
        families,
        sourceMap: rangeBetween(this.sourceName, start, end),
      }));
    },
  );

  private readonly useThemeDeclaration = this.RULE(
    "useThemeDeclaration",
    (): UseThemeAst => {
      const start = this.CONSUME(Use);
      this.CONSUME(Theme);
      const themeId = this.CONSUME(Identifier);
      return this.ACTION(() => ({
        kind: "use-theme",
        themeId: themeId.image,
        sourceMap: rangeBetween(this.sourceName, start, themeId),
      }));
    },
  );

  private readonly scene = this.RULE("scene", (): SceneAst => {
    const start = this.CONSUME(Scene);
    const title = this.CONSUME(QuotedString);
    this.CONSUME(As);
    const id = this.CONSUME(Identifier);
    this.CONSUME(LBrace);
    let summary: SummaryAst | undefined;
    const renderDeclarations: RenderDeclarationAst[] = [];
    const cameras: CameraAst[] = [];
    const timelines: TimelineAst[] = [];
    const interactions: InteractionAst[] = [];
    const responsiveVariants: ResponsiveAst[] = [];
    let narration: NarrationAst | undefined;
    let readingOrder: ReadingOrderAst | undefined;
    let fallback: FallbackAst | undefined;
    this.MANY(() =>
      this.OR([
        {
          ALT: () => {
            const node = this.SUBRULE(this.summary);
            this.ACTION(() => {
              summary = node;
            });
          },
        },
        {
          ALT: () => {
            const node = this.SUBRULE(this.rect);
            this.ACTION(() => renderDeclarations.push(node));
          },
        },
        {
          ALT: () => {
            const node = this.SUBRULE(this.connector);
            this.ACTION(() => renderDeclarations.push(node));
          },
        },
        {
          ALT: () => {
            const node = this.SUBRULE(this.scenePrimitive);
            this.ACTION(() => renderDeclarations.push(node));
          },
        },
        {
          ALT: () => {
            const node = this.SUBRULE(this.componentInvocation);
            this.ACTION(() => renderDeclarations.push(node));
          },
        },
        {
          ALT: () => {
            const node = this.SUBRULE(this.camera);
            this.ACTION(() => cameras.push(node));
          },
        },
        {
          ALT: () => {
            const node = this.SUBRULE(this.timeline);
            this.ACTION(() => timelines.push(node));
          },
        },
        {
          ALT: () => {
            const node = this.SUBRULE(this.interaction);
            this.ACTION(() => interactions.push(node));
          },
        },
        {
          ALT: () => {
            const node = this.SUBRULE(this.responsive);
            this.ACTION(() => responsiveVariants.push(node));
          },
        },
        {
          ALT: () => {
            const node = this.SUBRULE(this.narration);
            this.ACTION(() => {
              narration = node;
            });
          },
        },
        {
          ALT: () => {
            const start = this.CONSUME(ReadingOrder);
            const references = this.SUBRULE(this.referenceList);
            this.ACTION(() => {
              readingOrder = {
                kind: "reading-order",
                references: references.references,
                sourceMap: {
                  source: this.sourceName,
                  start: tokenRange(this.sourceName, start).start,
                  end: references.sourceMap.end,
                },
              };
            });
          },
        },
        {
          ALT: () => {
            const node = this.SUBRULE(this.fallback);
            this.ACTION(() => {
              fallback = node;
            });
          },
        },
      ]),
    );
    const end = this.CONSUME(RBrace);
    return this.ACTION(() => ({
      kind: "scene",
      title: stringValue(title),
      id: id.image,
      ...(summary === undefined ? {} : { summary }),
      renderDeclarations,
      cameras,
      timelines,
      interactions,
      responsiveVariants,
      ...(narration === undefined ? {} : { narration }),
      ...(readingOrder === undefined ? {} : { readingOrder }),
      ...(fallback === undefined ? {} : { fallback }),
      sourceMap: rangeBetween(this.sourceName, start, end),
    }));
  });

  private readonly summary = this.RULE("summary", (): SummaryAst => {
    const start = this.CONSUME(Summary);
    const text = this.CONSUME(QuotedString);
    return this.ACTION(() => ({
      kind: "summary",
      text: stringValue(text),
      sourceMap: rangeBetween(this.sourceName, start, text),
    }));
  });

  private readonly rect = this.RULE("rect", (): RectAst => {
    const start = this.CONSUME(Rect);
    const id = this.CONSUME(Identifier);
    this.CONSUME(LBrace);
    let x = 0;
    let y = 0;
    let width = 0;
    let height = 0;
    let fill: ValueAst | undefined;
    let stroke: ValueAst | undefined;
    let label = "";
    let role = "";
    let description = "";
    let fallback: FallbackAst | undefined;
    this.MANY(() =>
      this.OR([
        {
          ALT: () => {
            this.CONSUME(X);
            const value = this.CONSUME(NumberLiteral);
            this.ACTION(() => {
              x = numberValue(value);
            });
          },
        },
        {
          ALT: () => {
            this.CONSUME(Y);
            const value = this.CONSUME2(NumberLiteral);
            this.ACTION(() => {
              y = numberValue(value);
            });
          },
        },
        {
          ALT: () => {
            this.CONSUME(Width);
            const value = this.CONSUME3(NumberLiteral);
            this.ACTION(() => {
              width = numberValue(value);
            });
          },
        },
        {
          ALT: () => {
            this.CONSUME(Height);
            const value = this.CONSUME4(NumberLiteral);
            this.ACTION(() => {
              height = numberValue(value);
            });
          },
        },
        {
          ALT: () => {
            this.CONSUME(Fill);
            const value = this.SUBRULE(this.value);
            this.ACTION(() => {
              fill = value;
            });
          },
        },
        {
          ALT: () => {
            this.CONSUME(Stroke);
            const value = this.SUBRULE2(this.value);
            this.ACTION(() => {
              stroke = value;
            });
          },
        },
        {
          ALT: () => {
            this.CONSUME(Label);
            const value = this.CONSUME(QuotedString);
            this.ACTION(() => {
              label = stringValue(value);
            });
          },
        },
        {
          ALT: () => {
            this.CONSUME(Role);
            const value = this.CONSUME2(QuotedString);
            this.ACTION(() => {
              role = stringValue(value);
            });
          },
        },
        {
          ALT: () => {
            this.CONSUME(Description);
            const value = this.CONSUME3(QuotedString);
            this.ACTION(() => {
              description = stringValue(value);
            });
          },
        },
        {
          ALT: () => {
            const value = this.SUBRULE(this.fallback);
            this.ACTION(() => {
              fallback = value;
            });
          },
        },
      ]),
    );
    const end = this.CONSUME(RBrace);
    return this.ACTION(() => ({
      kind: "rect",
      id: id.image,
      x,
      y,
      width,
      height,
      fill: fill ?? literal(this.sourceName, id, ""),
      ...(stroke === undefined ? {} : { stroke }),
      label,
      role,
      description,
      fallback:
        fallback ?? {
          kind: "fallback",
          text: "",
          sourceMap: tokenRange(this.sourceName, id),
        },
      sourceMap: rangeBetween(this.sourceName, start, end),
    }));
  });

  private readonly connector = this.RULE("connector", (): ConnectorAst => {
    const start = this.CONSUME(Connector);
    const id = this.CONSUME(Identifier);
    this.CONSUME(LBrace);
    let from = "";
    let to = "";
    let label = "";
    let stroke: ValueAst | undefined;
    let fallback: FallbackAst | undefined;
    this.MANY(() =>
      this.OR([
        {
          ALT: () => {
            this.CONSUME(From);
            const value = this.CONSUME2(Identifier);
            this.ACTION(() => {
              from = value.image;
            });
          },
        },
        {
          ALT: () => {
            this.CONSUME(To);
            const value = this.CONSUME3(Identifier);
            this.ACTION(() => {
              to = value.image;
            });
          },
        },
        {
          ALT: () => {
            this.CONSUME(Label);
            const value = this.CONSUME(QuotedString);
            this.ACTION(() => {
              label = stringValue(value);
            });
          },
        },
        {
          ALT: () => {
            this.CONSUME(Stroke);
            const value = this.SUBRULE(this.value);
            this.ACTION(() => {
              stroke = value;
            });
          },
        },
        {
          ALT: () => {
            const value = this.SUBRULE(this.fallback);
            this.ACTION(() => {
              fallback = value;
            });
          },
        },
      ]),
    );
    const end = this.CONSUME(RBrace);
    return this.ACTION(() => ({
      kind: "connector",
      id: id.image,
      from,
      to,
      label,
      stroke: stroke ?? literal(this.sourceName, id, ""),
      fallback:
        fallback ?? {
          kind: "fallback",
          text: "",
          sourceMap: tokenRange(this.sourceName, id),
        },
      sourceMap: rangeBetween(this.sourceName, start, end),
    }));
  });

  private readonly scenePrimitive = this.RULE(
    "scenePrimitive",
    (): ScenePrimitiveAst => {
      const start = this.OR([
        { ALT: () => this.CONSUME(Panel) },
        { ALT: () => this.CONSUME(Header) },
        { ALT: () => this.CONSUME(Circle) },
        { ALT: () => this.CONSUME(Ellipse) },
        { ALT: () => this.CONSUME(Arrow) },
        { ALT: () => this.CONSUME(Elbow) },
        { ALT: () => this.CONSUME(Bracket) },
        { ALT: () => this.CONSUME(Callout) },
        { ALT: () => this.CONSUME(Stack) },
        { ALT: () => this.CONSUME(Grid) },
        { ALT: () => this.CONSUME(Pad) },
        { ALT: () => this.CONSUME(Signal) },
      ]);
      const id = this.CONSUME(Identifier);
      this.CONSUME(LBrace);
      const props: PropAssignmentAst[] = [];
      let children: RenderDeclarationAst[] | undefined;
      let fallback: FallbackAst | undefined;

      const pushProp = (name: string, value: PropAssignmentAst["value"], rangeToken: IToken) => {
        props.push({
          kind: "prop-assignment",
          name,
          value,
          sourceMap: {
            source: this.sourceName,
            start: tokenRange(this.sourceName, rangeToken).start,
            end: value.sourceMap.end,
          },
        });
      };

      this.MANY(() =>
        this.OR2([
          {
            ALT: () => {
              const key = this.CONSUME(X);
              const value = this.CONSUME(NumberLiteral);
              this.ACTION(() =>
                pushProp(
                  "x",
                  literal(this.sourceName, value, numberValue(value)),
                  key,
                ),
              );
            },
          },
          {
            ALT: () => {
              const key = this.CONSUME(Y);
              const value = this.CONSUME2(NumberLiteral);
              this.ACTION(() =>
                pushProp(
                  "y",
                  literal(this.sourceName, value, numberValue(value)),
                  key,
                ),
              );
            },
          },
          {
            ALT: () => {
              const key = this.CONSUME(Width);
              const value = this.CONSUME3(NumberLiteral);
              this.ACTION(() =>
                pushProp(
                  "width",
                  literal(this.sourceName, value, numberValue(value)),
                  key,
                ),
              );
            },
          },
          {
            ALT: () => {
              const key = this.CONSUME(Height);
              const value = this.CONSUME4(NumberLiteral);
              this.ACTION(() =>
                pushProp(
                  "height",
                  literal(this.sourceName, value, numberValue(value)),
                  key,
                ),
              );
            },
          },
          {
            ALT: () => {
              const key = this.CONSUME(Cols);
              const value = this.CONSUME5(NumberLiteral);
              this.ACTION(() =>
                pushProp(
                  "cols",
                  literal(this.sourceName, value, numberValue(value)),
                  key,
                ),
              );
            },
          },
          {
            ALT: () => {
              const key = this.CONSUME(Gap);
              const value = this.CONSUME6(NumberLiteral);
              this.ACTION(() =>
                pushProp(
                  "gap",
                  literal(this.sourceName, value, numberValue(value)),
                  key,
                ),
              );
            },
          },
          {
            // Circle radius (`r`) — bare Identifier to avoid lexing `/r/` keywords.
            GATE: () =>
              this.LA(1).tokenType === Identifier && this.LA(1).image === "r",
            ALT: () => {
              const key = this.CONSUME4(Identifier);
              const value = this.CONSUME7(NumberLiteral);
              this.ACTION(() =>
                pushProp(
                  "r",
                  literal(this.sourceName, value, numberValue(value)),
                  key,
                ),
              );
            },
          },
          {
            ALT: () => {
              const key = this.CONSUME(Title);
              const value = this.CONSUME(QuotedString);
              this.ACTION(() =>
                pushProp(
                  "title",
                  literal(this.sourceName, value, stringValue(value)),
                  key,
                ),
              );
            },
          },
          {
            ALT: () => {
              const key = this.CONSUME(Detail);
              const value = this.CONSUME2(QuotedString);
              this.ACTION(() =>
                pushProp(
                  "detail",
                  literal(this.sourceName, value, stringValue(value)),
                  key,
                ),
              );
            },
          },
          {
            ALT: () => {
              const key = this.CONSUME(Caption);
              const value = this.CONSUME3(QuotedString);
              this.ACTION(() =>
                pushProp(
                  "caption",
                  literal(this.sourceName, value, stringValue(value)),
                  key,
                ),
              );
            },
          },
          {
            ALT: () => {
              const key = this.CONSUME(Label);
              const value = this.CONSUME4(QuotedString);
              this.ACTION(() =>
                pushProp(
                  "label",
                  literal(this.sourceName, value, stringValue(value)),
                  key,
                ),
              );
            },
          },
          {
            ALT: () => {
              const key = this.CONSUME(Role);
              const value = this.CONSUME5(QuotedString);
              this.ACTION(() =>
                pushProp(
                  "role",
                  literal(this.sourceName, value, stringValue(value)),
                  key,
                ),
              );
            },
          },
          {
            ALT: () => {
              const key = this.CONSUME(Description);
              const value = this.CONSUME6(QuotedString);
              this.ACTION(() =>
                pushProp(
                  "description",
                  literal(this.sourceName, value, stringValue(value)),
                  key,
                ),
              );
            },
          },
          {
            ALT: () => {
              const key = this.CONSUME(Direction);
              const value = this.OR3([
                {
                  ALT: () => this.CONSUME7(QuotedString),
                },
                {
                  ALT: () => this.CONSUME2(Identifier),
                },
              ]);
              this.ACTION(() => {
                const propValue =
                  value.tokenType === QuotedString
                    ? literal(this.sourceName, value, stringValue(value))
                    : ({
                        kind: "identifier-reference",
                        name: value.image,
                        sourceMap: tokenRange(this.sourceName, value),
                      } satisfies IdentifierReferenceAst);
                pushProp("direction", propValue, key);
              });
            },
          },
          {
            // SVG path data (`d`) — bare Identifier to avoid lexing `/d/` keywords.
            GATE: () =>
              this.LA(1).tokenType === Identifier && this.LA(1).image === "d",
            ALT: () => {
              const key = this.CONSUME5(Identifier);
              const value = this.CONSUME8(QuotedString);
              this.ACTION(() =>
                pushProp(
                  "d",
                  literal(this.sourceName, value, stringValue(value)),
                  key,
                ),
              );
            },
          },
          {
            ALT: () => {
              const key = this.CONSUME(Fill);
              const value = this.SUBRULE(this.value);
              this.ACTION(() => pushProp("fill", value, key));
            },
          },
          {
            ALT: () => {
              const key = this.CONSUME(Stroke);
              const value = this.SUBRULE2(this.value);
              this.ACTION(() => pushProp("stroke", value, key));
            },
          },
          {
            ALT: () => {
              const key = this.CONSUME(From);
              const value = this.SUBRULE(this.endpointValue);
              this.ACTION(() => pushProp("from", value, key));
            },
          },
          {
            ALT: () => {
              const key = this.CONSUME(To);
              const value = this.SUBRULE2(this.endpointValue);
              this.ACTION(() => pushProp("to", value, key));
            },
          },
          {
            ALT: () => {
              const key = this.CONSUME(Via);
              const value = this.SUBRULE(this.pointValue);
              this.ACTION(() => pushProp("via", value, key));
            },
          },
          {
            ALT: () => {
              const key = this.CONSUME(Center);
              const value = this.SUBRULE2(this.pointValue);
              this.ACTION(() => pushProp("center", value, key));
            },
          },
          {
            ALT: () => {
              const key = this.CONSUME(Axis);
              const value = this.OR4([
                { ALT: () => this.CONSUME2(X) },
                { ALT: () => this.CONSUME2(Y) },
              ]);
              this.ACTION(() =>
                pushProp(
                  "axis",
                  {
                    kind: "identifier-reference",
                    name: value.image,
                    sourceMap: tokenRange(this.sourceName, value),
                  },
                  key,
                ),
              );
            },
          },
          {
            ALT: () => {
              this.CONSUME(Children);
              this.CONSUME2(LBrace);
              const nested: RenderDeclarationAst[] = [];
              this.MANY2(() => {
                const child = this.SUBRULE(this.renderDeclaration);
                this.ACTION(() => nested.push(child));
              });
              this.CONSUME2(RBrace);
              this.ACTION(() => {
                children = nested;
              });
            },
          },
          {
            ALT: () => {
              const value = this.SUBRULE(this.fallback);
              this.ACTION(() => {
                fallback = value;
              });
            },
          },
        ]),
      );
      const end = this.CONSUME(RBrace);
      return this.ACTION(() => {
        const primitive = start.image as ScenePrimitiveKeyword;
        return {
          kind: "scene-primitive",
          id: id.image,
          primitive,
          capability: SCENE_PRIMITIVE_CAPABILITIES[primitive],
          props,
          ...(children === undefined ? {} : { children }),
          ...(fallback === undefined ? {} : { fallback }),
          sourceMap: rangeBetween(this.sourceName, start, end),
        } satisfies ScenePrimitiveAst;
      });
    },
  );

  private readonly renderDeclaration = this.RULE(
    "renderDeclaration",
    (): RenderDeclarationAst =>
      this.OR([
        { ALT: () => this.SUBRULE(this.rect) },
        { ALT: () => this.SUBRULE(this.connector) },
        { ALT: () => this.SUBRULE(this.scenePrimitive) },
        { ALT: () => this.SUBRULE(this.componentInvocation) },
      ]),
  );

  private readonly endpointValue = this.RULE(
    "endpointValue",
    (): PropAssignmentAst["value"] =>
      this.OR([
        { ALT: () => this.SUBRULE(this.objectLiteral) },
        {
          ALT: () => {
            const name = this.CONSUME(Identifier);
            return this.ACTION(
              (): IdentifierReferenceAst => ({
                kind: "identifier-reference",
                name: name.image,
                sourceMap: tokenRange(this.sourceName, name),
              }),
            );
          },
        },
      ]),
  );

  private readonly pointValue = this.RULE(
    "pointValue",
    (): PropAssignmentAst["value"] =>
      this.OR([
        { ALT: () => this.SUBRULE(this.objectLiteral) },
        {
          ALT: () => {
            const x = this.CONSUME(NumberLiteral);
            const y = this.CONSUME2(NumberLiteral);
            return this.ACTION(
              (): ObjectLiteralAst => ({
                kind: "object-literal",
                properties: [
                  {
                    kind: "object-property",
                    name: "x",
                    value: literal(this.sourceName, x, numberValue(x)),
                    sourceMap: tokenRange(this.sourceName, x),
                  },
                  {
                    kind: "object-property",
                    name: "y",
                    value: literal(this.sourceName, y, numberValue(y)),
                    sourceMap: tokenRange(this.sourceName, y),
                  },
                ],
                sourceMap: rangeBetween(this.sourceName, x, y),
              }),
            );
          },
        },
      ]),
  );

  private readonly value = this.RULE("value", (): ValueAst =>
    this.OR([
      {
        ALT: () => {
          const value = this.CONSUME(QuotedString);
          return this.ACTION(() =>
            literal(this.sourceName, value, stringValue(value)),
          );
        },
      },
      {
        ALT: () => {
          const value = this.CONSUME(NumberLiteral);
          return this.ACTION(() =>
            literal(this.sourceName, value, numberValue(value)),
          );
        },
      },
      {
        ALT: () => {
          const value = this.CONSUME(True);
          return this.ACTION(() => literal(this.sourceName, value, true));
        },
      },
      {
        ALT: () => {
          const value = this.CONSUME(False);
          return this.ACTION(() => literal(this.sourceName, value, false));
        },
      },
      { ALT: () => this.SUBRULE(this.tokenReference) },
      { ALT: () => this.SUBRULE(this.themeRoleReference) },
    ]),
  );

  private readonly symbolDefinition = this.RULE(
    "symbolDefinition",
    (): SymbolDefinitionAst => {
      const start = this.CONSUME(SymbolKeyword);
      const name = this.CONSUME(Identifier);
      this.CONSUME(LParen);
      const params: ParamDeclarationAst[] = [];
      this.MANY_SEP({
        SEP: Comma,
        DEF: () => {
          const param = this.SUBRULE(this.paramDeclaration);
          this.ACTION(() => params.push(param));
        },
      });
      this.CONSUME(RParen);
      this.CONSUME(LBrace);
      const body: SymbolBodyStatementAst[] = [];
      this.MANY(() => {
        const statement = this.SUBRULE(this.symbolBodyStatement);
        this.ACTION(() => body.push(statement));
      });
      const end = this.CONSUME(RBrace);
      return this.ACTION(() => ({
        kind: "symbol-definition",
        name: name.image,
        params,
        body,
        sourceMap: rangeBetween(this.sourceName, start, end),
      }));
    },
  );

  private readonly paramDeclaration = this.RULE(
    "paramDeclaration",
    (): ParamDeclarationAst => {
      const name = this.CONSUME(Identifier);
      this.CONSUME(Colon);
      const type = this.SUBRULE(this.typeReference);
      return this.ACTION(() => ({
        kind: "param",
        name: name.image,
        type,
        sourceMap: {
          source: this.sourceName,
          start: tokenRange(this.sourceName, name).start,
          end: type.sourceMap.end,
        },
      }));
    },
  );

  private readonly typeReference = this.RULE(
    "typeReference",
    (): TypeRefAst => {
      const name = this.CONSUME(Identifier);
      let end = name;
      let array = false;
      this.OPTION(() => {
        this.CONSUME(LBracket);
        end = this.CONSUME(RBracket);
        this.ACTION(() => {
          array = true;
        });
      });
      return this.ACTION(() => ({
        kind: "type-ref",
        name: name.image,
        ...(array ? { array: true as const } : {}),
        sourceMap: rangeBetween(this.sourceName, name, end),
      }));
    },
  );

  private readonly componentInvocation = this.RULE(
    "componentInvocation",
    (): ComponentInvocationAst => {
      let namespace: string | undefined;
      let name: IToken;
      const start = this.OR([
        {
          ALT: () => {
            const qualifier = this.CONSUME(Identifier);
            this.CONSUME(Dot);
            const component = this.CONSUME(ComponentIdentifier);
            this.ACTION(() => {
              namespace = qualifier.image;
              name = component;
            });
            return qualifier;
          },
        },
        {
          ALT: () => {
            const component = this.CONSUME2(ComponentIdentifier);
            this.ACTION(() => {
              name = component;
            });
            return component;
          },
        },
      ]);
      this.CONSUME(LParen);
      const props: PropAssignmentAst[] = [];
      this.MANY_SEP({
        SEP: Comma,
        DEF: () => {
          const prop = this.SUBRULE(this.propAssignment);
          this.ACTION(() => props.push(prop));
        },
      });
      let end = this.CONSUME(RParen);
      let slots: SlotBlockAst[] | undefined;
      this.OPTION(() => {
        this.CONSUME(LBrace);
        slots = [];
        this.MANY(() => {
          const slot = this.SUBRULE(this.slotBlock);
          this.ACTION(() => slots?.push(slot));
        });
        end = this.CONSUME(RBrace);
      });
      return this.ACTION(() => ({
        kind: "component-invocation",
        name: name.image,
        ...(namespace === undefined ? {} : { namespace }),
        props,
        ...(slots === undefined ? {} : { slots }),
        sourceMap: rangeBetween(this.sourceName, start, end),
      }));
    },
  );

  private readonly propAssignment = this.RULE(
    "propAssignment",
    (): PropAssignmentAst => {
      const name = this.CONSUME(Identifier);
      this.CONSUME(Equals);
      const value = this.SUBRULE(this.argumentValue);
      return this.ACTION(() => ({
        kind: "prop-assignment",
        name: name.image,
        value,
        sourceMap: {
          source: this.sourceName,
          start: tokenRange(this.sourceName, name).start,
          end: value.sourceMap.end,
        },
      }));
    },
  );

  private readonly argumentValue = this.RULE(
    "argumentValue",
    (): PropAssignmentAst["value"] =>
      this.OR([
        { ALT: () => this.SUBRULE(this.value) },
        { ALT: () => this.SUBRULE(this.objectLiteral) },
        {
          GATE: () => this.LA(1).tokenType === Identifier,
          ALT: () => {
            const name = this.CONSUME(Identifier);
            return this.ACTION(
              (): IdentifierReferenceAst => ({
                kind: "identifier-reference",
                name: name.image,
                sourceMap: tokenRange(this.sourceName, name),
              }),
            );
          },
        },
      ]),
  );

  private readonly objectLiteral = this.RULE(
    "objectLiteral",
    (): ObjectLiteralAst => {
      const start = this.CONSUME(LBrace);
      const properties: ObjectPropertyAst[] = [];
      this.MANY_SEP({
        SEP: Comma,
        DEF: () => {
          const property = this.SUBRULE(this.objectProperty);
          this.ACTION(() => properties.push(property));
        },
      });
      const end = this.CONSUME(RBrace);
      return this.ACTION(() => ({
        kind: "object-literal",
        properties,
        sourceMap: rangeBetween(this.sourceName, start, end),
      }));
    },
  );

  private readonly objectProperty = this.RULE(
    "objectProperty",
    (): ObjectPropertyAst => {
      const name = this.CONSUME(Identifier);
      this.CONSUME(Colon);
      const value = this.OR([
        { ALT: () => this.SUBRULE(this.value) },
        {
          GATE: () => this.LA(1).tokenType === Identifier,
          ALT: () => {
            const reference = this.CONSUME2(Identifier);
            return this.ACTION(
              (): IdentifierReferenceAst => ({
                kind: "identifier-reference",
                name: reference.image,
                sourceMap: tokenRange(this.sourceName, reference),
              }),
            );
          },
        },
      ]);
      return this.ACTION(() => ({
        kind: "object-property",
        name: name.image,
        value,
        sourceMap: {
          source: this.sourceName,
          start: tokenRange(this.sourceName, name).start,
          end: value.sourceMap.end,
        },
      }));
    },
  );

  private readonly symbolBodyStatement = this.RULE(
    "symbolBodyStatement",
    (): SymbolBodyStatementAst =>
      this.OR([
        { ALT: () => this.SUBRULE(this.forLoop) },
        { ALT: () => this.SUBRULE(this.componentInvocation) },
      ]),
  );

  private readonly slotBodyStatement = this.RULE(
    "slotBodyStatement",
    (): SymbolBodyStatementAst =>
      this.OR([
        { ALT: () => this.SUBRULE(this.forLoop) },
        { ALT: () => this.SUBRULE(this.componentInvocation) },
      ]),
  );

  private readonly slotBlock = this.RULE("slotBlock", (): SlotBlockAst => {
    const name = this.CONSUME(Identifier);
    let parameter: string | undefined;
    this.OPTION(() => {
      this.CONSUME(LParen);
      const value = this.CONSUME2(Identifier);
      this.CONSUME(RParen);
      this.ACTION(() => {
        parameter = value.image;
      });
    });
    this.CONSUME(LBrace);
    const body: SymbolBodyStatementAst[] = [];
    this.MANY(() => {
      const statement = this.SUBRULE(this.slotBodyStatement);
      this.ACTION(() => body.push(statement));
    });
    const end = this.CONSUME(RBrace);
    return this.ACTION(() => ({
      kind: "slot",
      name: name.image,
      ...(parameter === undefined ? {} : { parameter }),
      body,
      sourceMap: rangeBetween(this.sourceName, name, end),
    }));
  });

  private readonly forLoop = this.RULE("forLoop", (): ForLoopAst => {
    const start = this.CONSUME(For);
    const item = this.CONSUME(Identifier);
    this.CONSUME(In);
    const collection = this.CONSUME2(Identifier);
    this.CONSUME(LBrace);
    const body: ComponentInvocationAst[] = [];
    this.MANY(() => {
      const invocation = this.SUBRULE(this.componentInvocation);
      this.ACTION(() => body.push(invocation));
    });
    const end = this.CONSUME(RBrace);
    return this.ACTION(() => ({
      kind: "for-loop",
      item: item.image,
      collection: collection.image,
      body,
      sourceMap: rangeBetween(this.sourceName, start, end),
    }));
  });

  private readonly tokenReference = this.RULE(
    "tokenReference",
    (): TokenReferenceAst => {
      const start = this.CONSUME(Token);
      this.CONSUME(LParen);
      const id = this.CONSUME(Identifier);
      const end = this.CONSUME(RParen);
      return this.ACTION(() => ({
        kind: "token-reference",
        token: id.image,
        sourceMap: rangeBetween(this.sourceName, start, end),
      }));
    },
  );

  private readonly themeRoleReference = this.RULE(
    "themeRoleReference",
    (): ThemeRoleReferenceAst => {
      const start = this.CONSUME(Theme);
      this.CONSUME(LParen);
      const role = this.SUBRULE(this.qualifiedName);
      const end = this.CONSUME(RParen);
      return this.ACTION(() => ({
        kind: "theme-role-reference",
        role,
        sourceMap: rangeBetween(this.sourceName, start, end),
      }));
    },
  );

  private readonly fallback = this.RULE("fallback", (): FallbackAst => {
    const start = this.CONSUME(Fallback);
    const text = this.CONSUME(QuotedString);
    return this.ACTION(() => ({
      kind: "fallback",
      text: stringValue(text),
      sourceMap: rangeBetween(this.sourceName, start, text),
    }));
  });

  private readonly camera = this.RULE("camera", (): CameraAst => {
    const start = this.CONSUME(Camera);
    const id = this.CONSUME(Identifier);
    this.CONSUME(LBrace);
    const keyframes: CameraKeyframeAst[] = [];
    this.AT_LEAST_ONE(() => {
      const keyframe = this.SUBRULE(this.cameraKeyframe);
      this.ACTION(() => keyframes.push(keyframe));
    });
    const end = this.CONSUME(RBrace);
    return this.ACTION(() => ({
      kind: "camera",
      id: id.image,
      keyframes,
      sourceMap: rangeBetween(this.sourceName, start, end),
    }));
  });

  private readonly cameraKeyframe = this.RULE(
    "cameraKeyframe",
    (): CameraKeyframeAst => {
      const start = this.CONSUME(At);
      const time = this.CONSUME(NumberLiteral);
      this.CONSUME(Frame);
      const targets = this.SUBRULE(this.referenceList);
      this.CONSUME(Zoom);
      const zoom = this.CONSUME2(NumberLiteral);
      return this.ACTION(() => ({
        kind: "camera-keyframe",
        time: numberValue(time),
        targets,
        zoom: numberValue(zoom),
        sourceMap: rangeBetween(this.sourceName, start, zoom),
      }));
    },
  );

  private readonly referenceList = this.RULE(
    "referenceList",
    (): ReferenceListAst => {
      const references: string[] = [];
      const start = this.CONSUME(Identifier);
      let end = start;
      this.ACTION(() => references.push(start.image));
      this.MANY(() => {
        this.CONSUME(Comma);
        const reference = this.CONSUME2(Identifier);
        this.ACTION(() => {
          references.push(reference.image);
          end = reference;
        });
      });
      return this.ACTION(() => ({
        kind: "reference-list",
        references,
        sourceMap: rangeBetween(this.sourceName, start, end),
      }));
    },
  );

  private readonly timeline = this.RULE("timeline", (): TimelineAst => {
    const start = this.CONSUME(Timeline);
    const id = this.CONSUME(Identifier);
    this.CONSUME(LBrace);
    const cues: TimelineCueAst[] = [];
    this.AT_LEAST_ONE(() => {
      const cue = this.SUBRULE(this.timelineCue);
      this.ACTION(() => cues.push(cue));
    });
    const end = this.CONSUME(RBrace);
    return this.ACTION(() => ({
      kind: "timeline",
      id: id.image,
      cues,
      sourceMap: rangeBetween(this.sourceName, start, end),
    }));
  });

  private readonly timelineCue = this.RULE(
    "timelineCue",
    (): TimelineCueAst => {
      const start = this.CONSUME(At);
      const time = this.CONSUME(NumberLiteral);
      return this.OR([
        {
          ALT: () => {
            this.CONSUME(Stagger);
            this.CONSUME(Targets);
            this.CONSUME(LBracket);
            const targets: string[] = [];
            this.AT_LEAST_ONE_SEP({
              SEP: Comma,
              DEF: () => {
                const id = this.CONSUME(Identifier);
                this.ACTION(() => targets.push(id.image));
              },
            });
            this.CONSUME(RBracket);
            let step: number | undefined;
            this.OPTION(() => {
              this.CONSUME(Step);
              const stepToken = this.CONSUME2(NumberLiteral);
              this.ACTION(() => {
                step = numberValue(stepToken);
              });
            });
            this.CONSUME(Duration);
            const duration = this.CONSUME3(NumberLiteral);
            let easing: TimelineCueEasing | undefined;
            let endToken: IToken = duration;
            this.OPTION2(() => {
              this.CONSUME(Easing);
              const easingToken = this.SUBRULE(this.easingValue);
              this.ACTION(() => {
                easing = easingToken.image as TimelineCueEasing;
                endToken = easingToken;
              });
            });
            return this.ACTION(() => ({
              kind: "timeline-cue" as const,
              time: numberValue(time),
              action: "stagger" as const,
              target: "",
              duration: numberValue(duration),
              targets,
              ...(step === undefined ? {} : { step }),
              ...(easing === undefined ? {} : { easing }),
              sourceMap: rangeBetween(this.sourceName, start, endToken),
            }));
          },
        },
        {
          ALT: () => {
            const actionToken = this.OR2([
              { ALT: () => this.CONSUME(Reveal) },
              { ALT: () => this.CONSUME(Trace) },
              { ALT: () => this.CONSUME(Fade) },
              { ALT: () => this.CONSUME(Exit) },
              { ALT: () => this.CONSUME(EnterChildren) },
            ]);
            const target = this.CONSUME2(Identifier);
            this.CONSUME2(Duration);
            const duration = this.CONSUME4(NumberLiteral);
            let step: number | undefined;
            let easing: TimelineCueEasing | undefined;
            let endToken: IToken = duration;
            this.OPTION3(() => {
              this.CONSUME2(Step);
              const stepToken = this.CONSUME5(NumberLiteral);
              this.ACTION(() => {
                step = numberValue(stepToken);
                endToken = stepToken;
              });
            });
            this.OPTION4(() => {
              this.CONSUME2(Easing);
              const easingToken = this.SUBRULE2(this.easingValue);
              this.ACTION(() => {
                easing = easingToken.image as TimelineCueEasing;
                endToken = easingToken;
              });
            });
            return this.ACTION(() => ({
              kind: "timeline-cue" as const,
              time: numberValue(time),
              action: actionToken.image as TimelineAction,
              target: target.image,
              duration: numberValue(duration),
              ...(step === undefined ? {} : { step }),
              ...(easing === undefined ? {} : { easing }),
              sourceMap: rangeBetween(this.sourceName, start, endToken),
            }));
          },
        },
      ]);
    },
  );

  private readonly easingValue = this.RULE("easingValue", (): IToken =>
    this.OR([
      { ALT: () => this.CONSUME(Linear) },
      { ALT: () => this.CONSUME(EaseInOut) },
      { ALT: () => this.CONSUME(EaseIn) },
      { ALT: () => this.CONSUME(EaseOut) },
    ]),
  );

  private readonly interaction = this.RULE(
    "interaction",
    (): InteractionAst => {
      const start = this.CONSUME(Interaction);
      const id = this.CONSUME(Identifier);
      this.CONSUME(LBrace);
      const event = this.SUBRULE(this.interactionEvent);
      const action = this.SUBRULE(this.interactionAction);
      const end = this.CONSUME(RBrace);
      return this.ACTION(() => ({
        kind: "interaction",
        id: id.image,
        event,
        action,
        sourceMap: rangeBetween(this.sourceName, start, end),
      }));
    },
  );

  private readonly interactionEvent = this.RULE(
    "interactionEvent",
    (): InteractionEventAst => {
      const start = this.CONSUME(On);
      this.CONSUME(Select);
      const target = this.CONSUME(Identifier);
      return this.ACTION(() => ({
        kind: "interaction-event",
        name: "select",
        target: target.image,
        sourceMap: rangeBetween(this.sourceName, start, target),
      }));
    },
  );

  private readonly interactionAction = this.RULE(
    "interactionAction",
    (): InteractionActionAst => {
      const start = this.CONSUME(Do);
      this.CONSUME(Inspect);
      const target = this.CONSUME(Identifier);
      return this.ACTION(() => ({
        kind: "interaction-action",
        name: "inspect",
        target: target.image,
        sourceMap: rangeBetween(this.sourceName, start, target),
      }));
    },
  );

  private readonly responsive = this.RULE(
    "responsive",
    (): ResponsiveAst => {
      const start = this.CONSUME(Responsive);
      const id = this.CONSUME(Identifier);
      this.CONSUME(When);
      const condition = this.SUBRULE(this.responsiveCondition);
      this.CONSUME(LBrace);
      const overrides: ResponsiveOverrideAst[] = [];
      this.AT_LEAST_ONE(() => {
        const override = this.SUBRULE(this.responsiveOverride);
        this.ACTION(() => overrides.push(override));
      });
      const end = this.CONSUME(RBrace);
      return this.ACTION(() => ({
        kind: "responsive",
        id: id.image,
        condition,
        overrides,
        sourceMap: rangeBetween(this.sourceName, start, end),
      }));
    },
  );

  private readonly responsiveCondition = this.RULE(
    "responsiveCondition",
    (): ResponsiveConditionAst => {
      const property = this.CONSUME(Identifier);
      const operator = this.SUBRULE(this.comparisonOperator);
      const value = this.CONSUME(NumberLiteral);
      return this.ACTION(() => ({
        kind: "responsive-condition",
        property: property.image,
        operator: operator.image as ComparisonOperator,
        value: numberValue(value),
        sourceMap: rangeBetween(this.sourceName, property, value),
      }));
    },
  );

  private readonly comparisonOperator = this.RULE(
    "comparisonOperator",
    (): IToken =>
      this.OR([
        { ALT: () => this.CONSUME(LessEqual) },
        { ALT: () => this.CONSUME(GreaterEqual) },
        { ALT: () => this.CONSUME(EqualEqual) },
        { ALT: () => this.CONSUME(NotEqual) },
        { ALT: () => this.CONSUME(Less) },
        { ALT: () => this.CONSUME(Greater) },
      ]),
  );

  private readonly responsiveOverride = this.RULE(
    "responsiveOverride",
    (): ResponsiveOverrideAst => {
      const start = this.CONSUME(Set);
      const target = this.CONSUME(Identifier);
      this.CONSUME(Dot);
      const property = this.CONSUME2(Identifier);
      this.CONSUME(Equals);
      const value = this.CONSUME(NumberLiteral);
      return this.ACTION(() => ({
        kind: "responsive-override",
        target: target.image,
        property: property.image,
        value: numberValue(value),
        sourceMap: rangeBetween(this.sourceName, start, value),
      }));
    },
  );

  private readonly narration = this.RULE(
    "narration",
    (): NarrationAst => {
      const start = this.CONSUME(Narrate);
      const text = this.CONSUME(QuotedString);
      return this.ACTION(() => ({
        kind: "narration",
        text: stringValue(text),
        sourceMap: rangeBetween(this.sourceName, start, text),
      }));
    },
  );
}

function positionAt(source: string, offset: number): SourcePosition {
  const prefix = source.slice(0, offset);
  const lines = prefix.split(/\r\n|\r|\n/);
  return {
    offset,
    line: lines.length,
    column: (lines.at(-1)?.length ?? 0) + 1,
  };
}

function lexDiagnostic(
  source: string,
  sourceName: string,
  error: ILexingError,
): Diagnostic {
  const start = positionAt(source, error.offset);
  return {
    code: "LEX_INVALID_CHARACTER",
    severity: "error",
    message: error.message,
    range: {
      source: sourceName,
      start,
      end: positionAt(source, error.offset + error.length),
    },
  };
}

function parseDiagnostic(
  sourceName: string,
  error: IRecognitionException,
): Diagnostic {
  return {
    code: "PARSE_UNEXPECTED_TOKEN",
    severity: "error",
    message: error.message,
    range: tokenRange(sourceName, error.token),
  };
}

/** Adapts Chevrotain lexer tokens to the TokenStream expected by parseExplainerBlock. */
class ChevrotainTokenStream implements TokenStream {
  private index = 0;

  constructor(private readonly tokens: readonly IToken[]) {}

  get done(): boolean {
    return this.index >= this.tokens.length;
  }

  peekImage(): string | undefined {
    return this.tokens[this.index]?.image;
  }

  expect(keyword: string): void {
    const current = this.tokens[this.index];
    if (current === undefined || current.image !== keyword) {
      throw new Error(
        `Expected "${keyword}" but got "${current?.image ?? "<eof>"}"`,
      );
    }
    this.index += 1;
  }

  expectString(): string {
    const current = this.tokens[this.index];
    if (current === undefined || current.tokenType !== QuotedString) {
      throw new Error(
        `Expected string but got "${current?.image ?? "<eof>"}"`,
      );
    }
    this.index += 1;
    return stringValue(current);
  }

  expectIdentifier(): string {
    const current = this.tokens[this.index];
    if (
      current === undefined ||
      current.tokenType === QuotedString ||
      /[{}\[\]:,@]/.test(current.image)
    ) {
      throw new Error(
        `Expected identifier but got "${current?.image ?? "<eof>"}"`,
      );
    }
    this.index += 1;
    return current.image;
  }

  match(keyword: string): boolean {
    return this.tokens[this.index]?.image === keyword;
  }

  advance(): void {
    this.index += 1;
  }
}

function documentSourceMap(source: string, sourceName: string): SourceRange {
  return {
    source: sourceName,
    start: positionAt(source, 0),
    end: positionAt(source, source.length),
  };
}

function toExplainerAst(
  compat: ExplainerAstCompat,
  sourceMap: SourceRange,
): ExplainerAst {
  return {
    kind: "explainer",
    id: compat.id,
    metadata: {
      route: compat.metadata.route,
      topic: compat.metadata.topic,
      storagePrefix: compat.metadata.storagePrefix,
      classPrefix: compat.metadata.classPrefix,
      eyebrowLabel: compat.metadata.eyebrowLabel,
      startGateTitle: compat.metadata.startGateTitle,
      hub: compat.metadata.hub,
      ...(compat.metadata.css === undefined
        ? {}
        : { css: compat.metadata.css }),
    },
    slides: compat.slides.map((slide) => toSlideAst(slide, sourceMap)),
    ...(compat.finalCard === undefined ? {} : { finalCard: compat.finalCard }),
    sourceMap,
  };
}

function toSlideAst(
  slide: ExplainerAstCompat["slides"][number],
  sourceMap: SourceRange,
): SlideAst {
  return {
    kind: "slide",
    eyebrow: slide.eyebrow ?? "",
    title: slide.title ?? "",
    lede: slide.lede ?? "",
    narration: slide.narration,
    ...(slide.term === undefined ? {} : { term: slide.term }),
    points: slide.points ?? [],
    caption: slide.caption ?? "",
    ...(slide.sceneIr === undefined ? {} : { sceneIr: slide.sceneIr }),
    sourceMap,
  };
}

function explainerOnlyDocument(
  explainers: readonly ExplainerAst[],
  sourceMap: SourceRange,
): DocumentAst {
  const primary = explainers[0]!;
  return {
    kind: "document",
    title: primary.id,
    id: primary.id,
    language: {
      kind: "language",
      version: 1,
      sourceMap,
    },
    requirements: [],
    tokens: [],
    themes: [],
    symbols: [],
    scenes: [],
    explainers,
    sourceMap,
  };
}

function parseExplainerDocument(
  source: string,
  sourceName: string,
  tokens: readonly IToken[],
  lexDiagnostics: readonly Diagnostic[],
): Result<DocumentAst> {
  const sourceMap = documentSourceMap(source, sourceName);
  if (lexDiagnostics.length > 0) {
    return { ok: false, diagnostics: [...lexDiagnostics] };
  }

  try {
    const stream = new ChevrotainTokenStream(tokens);
    const explainers: ExplainerAst[] = [];
    while (stream.match("explainer")) {
      explainers.push(toExplainerAst(parseExplainerBlock(stream), sourceMap));
    }
    if (explainers.length === 0) {
      return {
        ok: false,
        diagnostics: [
          {
            code: "PARSE_EXPLAINER",
            severity: "error",
            message: "Expected top-level explainer block",
            range: sourceMap,
          },
        ],
      };
    }
    if (!stream.done) {
      return {
        ok: false,
        diagnostics: [
          {
            code: "PARSE_EXPLAINER",
            severity: "error",
            message: `Unexpected token "${stream.peekImage() ?? "<eof>"}"`,
            range: sourceMap,
          },
        ],
      };
    }
    return {
      ok: true,
      value: explainerOnlyDocument(explainers, sourceMap),
      diagnostics: [],
    };
  } catch (error) {
    return {
      ok: false,
      diagnostics: [
        {
          code: "PARSE_EXPLAINER",
          severity: "error",
          message: error instanceof Error ? error.message : String(error),
          range: sourceMap,
        },
      ],
    };
  }
}

export function parseDocument(
  source: string,
  sourceName: string,
): Result<DocumentAst> {
  const lexResult = flowLexer.tokenize(source);
  const lexDiagnostics = lexResult.errors.map((error) =>
    lexDiagnostic(source, sourceName, error),
  );

  if (lexResult.tokens[0]?.tokenType === Explainer) {
    return parseExplainerDocument(
      source,
      sourceName,
      lexResult.tokens,
      lexDiagnostics,
    );
  }

  const parser = new FlowParser(sourceName);
  parser.input = lexResult.tokens;
  const value = parser.document();
  const diagnostics = [
    ...lexDiagnostics,
    ...parser.errors.map((error) => parseDiagnostic(sourceName, error)),
  ];

  if (diagnostics.length > 0) {
    return { ok: false, diagnostics };
  }

  return {
    ok: true,
    value: {
      ...value,
      sourceMap: documentSourceMap(source, sourceName),
    },
    diagnostics,
  };
}

/**
 * Parses a native cinematic `@scene` body (rect/connector/timeline/camera)
 * by wrapping it in a one-scene Flow document and reusing `parseDocument`.
 *
 * Shared path for explainer slides that author the same dialect as
 * `examples/cinematic/request-lifecycle.flow`.
 */
export function parseNativeEmbeddedScene(
  body: string,
  sourceName = "<embedded-scene>",
): Result<SceneAst> {
  const wrapped = `flow "Embedded scene" as embedded-scene {
  language 1
  scene "Embedded" as embedded {
${body}
  }
}
`;
  const parsed = parseDocument(wrapped, sourceName);
  if (!parsed.ok) {
    return parsed;
  }
  const scene = parsed.value.scenes[0];
  if (scene === undefined) {
    return {
      ok: false,
      diagnostics: [
        {
          code: "PARSE_EMBEDDED_SCENE",
          severity: "error",
          message: "Native @scene body produced no scene AST.",
          range: documentSourceMap(wrapped, sourceName),
        },
      ],
    };
  }
  return { ok: true, value: scene, diagnostics: parsed.diagnostics };
}
