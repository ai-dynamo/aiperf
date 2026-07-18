// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Lexer, createToken } from "chevrotain";

const keyword = (name: string, pattern: RegExp) =>
  createToken({
    name,
    pattern,
    longer_alt: Identifier,
    categories: Identifier,
  });

export const WhiteSpace = createToken({
  name: "WhiteSpace",
  pattern: /\s+/,
  group: Lexer.SKIPPED,
  line_breaks: true,
});
export const BlockComment = createToken({
  name: "BlockComment",
  pattern: /\/\*[\s\S]*?\*\//,
  group: Lexer.SKIPPED,
  line_breaks: true,
});
export const LineComment = createToken({
  name: "LineComment",
  pattern: /\/\/[^\r\n]*/,
  group: Lexer.SKIPPED,
});
export const QuotedString = createToken({
  name: "QuotedString",
  pattern: /"(?:\\["\\/bfnrt]|\\u[0-9a-fA-F]{4}|[^"\\\r\n])*"/,
});
export const DurationLiteral = createToken({
  name: "DurationLiteral",
  pattern: /[0-9]+ms/,
});
export const NumberLiteral = createToken({
  name: "NumberLiteral",
  pattern: /(?:0|[1-9]\d*)(?:\.\d+)?/,
});
export const Identifier = createToken({
  name: "Identifier",
  pattern: /[A-Za-z_][A-Za-z0-9_-]*/,
});
export const ComponentIdentifier = createToken({
  name: "ComponentIdentifier",
  pattern: /[A-Z][A-Za-z0-9_]*/,
  longer_alt: Identifier,
  categories: Identifier,
});

export const Flow = keyword("Flow", /flow/);
export const Import = keyword("Import", /import/);
export const As = keyword("As", /as/);
export const Language = keyword("Language", /language/);
export const Require = keyword("Require", /require/);
export const Token = keyword("Token", /token/);
export const Scene = keyword("Scene", /scene/);
export const Summary = keyword("Summary", /summary/);
export const Rect = keyword("Rect", /rect/);
export const Connector = keyword("Connector", /connector/);
export const X = keyword("X", /x/);
export const Y = keyword("Y", /y/);
export const Width = keyword("Width", /width/);
export const Height = keyword("Height", /height/);
export const Fill = keyword("Fill", /fill/);
export const Label = keyword("Label", /label/);
export const Role = keyword("Role", /role/);
export const Description = keyword("Description", /description/);
export const Fallback = keyword("Fallback", /fallback/);
export const From = keyword("From", /from/);
export const To = keyword("To", /to/);
export const Stroke = keyword("Stroke", /stroke/);
export const Camera = keyword("Camera", /camera/);
export const At = keyword("At", /at/);
export const Frame = keyword("Frame", /frame/);
export const Zoom = keyword("Zoom", /zoom/);
export const Timeline = keyword("Timeline", /timeline/);
export const Reveal = keyword("Reveal", /reveal/);
export const Trace = keyword("Trace", /trace/);
export const Duration = keyword("Duration", /duration/);
export const Interaction = keyword("Interaction", /interaction/);
export const On = keyword("On", /on/);
export const Select = keyword("Select", /select/);
export const Do = keyword("Do", /do/);
export const Inspect = keyword("Inspect", /inspect/);
export const Responsive = keyword("Responsive", /responsive/);
export const When = keyword("When", /when/);
export const Set = keyword("Set", /set/);
export const Narrate = keyword("Narrate", /narrate/);
export const ReadingOrder = keyword("ReadingOrder", /reading-order/);
export const SymbolKeyword = keyword("Symbol", /symbol/);
export const For = keyword("For", /for/);
export const In = keyword("In", /in/);
export const True = keyword("True", /true/);
export const False = keyword("False", /false/);
export const Theme = keyword("Theme", /theme/);
export const Use = keyword("Use", /use/);
export const Extends = keyword("Extends", /extends/);
export const ColorKind = keyword("ColorKind", /color/);
export const NumberKind = keyword("NumberKind", /number/);
export const FontKind = keyword("FontKind", /font/);
export const EnumKind = keyword("EnumKind", /enum/);

export const LessEqual = createToken({ name: "LessEqual", pattern: /<=/ });
export const GreaterEqual = createToken({ name: "GreaterEqual", pattern: />=/ });
export const EqualEqual = createToken({ name: "EqualEqual", pattern: /==/ });
export const NotEqual = createToken({ name: "NotEqual", pattern: /!=/ });
export const Less = createToken({ name: "Less", pattern: /</ });
export const Greater = createToken({ name: "Greater", pattern: />/ });
export const LBrace = createToken({ name: "LBrace", pattern: /\{/ });
export const RBrace = createToken({ name: "RBrace", pattern: /\}/ });
export const LParen = createToken({ name: "LParen", pattern: /\(/ });
export const RParen = createToken({ name: "RParen", pattern: /\)/ });
export const LBracket = createToken({ name: "LBracket", pattern: /\[/ });
export const RBracket = createToken({ name: "RBracket", pattern: /\]/ });
export const Comma = createToken({ name: "Comma", pattern: /,/ });
export const Dot = createToken({ name: "Dot", pattern: /\./ });
export const Colon = createToken({ name: "Colon", pattern: /:/ });
export const Equals = createToken({ name: "Equals", pattern: /=/ });

export const allTokens = [
  WhiteSpace,
  BlockComment,
  LineComment,
  QuotedString,
  DurationLiteral,
  NumberLiteral,
  Flow,
  Import,
  As,
  Language,
  Require,
  Token,
  Scene,
  Summary,
  Rect,
  Connector,
  X,
  Y,
  Width,
  Height,
  Fill,
  Label,
  Role,
  Description,
  Fallback,
  From,
  To,
  Stroke,
  Camera,
  At,
  Frame,
  Zoom,
  Timeline,
  Reveal,
  Trace,
  Duration,
  Interaction,
  On,
  Select,
  Do,
  Inspect,
  Responsive,
  When,
  Set,
  Narrate,
  ReadingOrder,
  SymbolKeyword,
  For,
  In,
  True,
  False,
  Theme,
  Use,
  Extends,
  ColorKind,
  NumberKind,
  FontKind,
  EnumKind,
  LessEqual,
  GreaterEqual,
  EqualEqual,
  NotEqual,
  Less,
  Greater,
  LBrace,
  RBrace,
  LParen,
  RParen,
  LBracket,
  RBracket,
  Comma,
  Dot,
  Colon,
  Equals,
  ComponentIdentifier,
  Identifier,
];

export const flowLexer = new Lexer(allTokens, {
  ensureOptimizations: true,
  positionTracking: "full",
});
