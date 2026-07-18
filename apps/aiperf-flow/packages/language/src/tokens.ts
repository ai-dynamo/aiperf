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
/** Native `panel` → `core.panel`. */
export const Panel = keyword("Panel", /panel/);
/** Native `header` → `core.header`. */
export const Header = keyword("Header", /header/);
/** Native `circle` → `core.circle`. */
export const Circle = keyword("Circle", /circle/);
/** Native `ellipse` → `core.ellipse`. */
export const Ellipse = keyword("Ellipse", /ellipse/);
/** Native `arrow` → `core.arrow`. */
export const Arrow = keyword("Arrow", /arrow/);
/** Native `elbow` → `core.elbow`. */
export const Elbow = keyword("Elbow", /elbow/);
/** Native `bracket` → `core.bracket`. */
export const Bracket = keyword("Bracket", /bracket/);
/** Native `callout` → `core.callout`. */
export const Callout = keyword("Callout", /callout/);
/** Native `stack` → `layout.stack`. */
export const Stack = keyword("Stack", /stack/);
/** Native `grid` → `layout.grid`. */
export const Grid = keyword("Grid", /grid/);
/** Native `pad` → `layout.pad`. */
export const Pad = keyword("Pad", /pad/);
/** Native `signal` → `motion.signal`. */
export const Signal = keyword("Signal", /signal/);
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
/** Panel / header title text. */
export const Title = keyword("Title", /title/);
/** Panel detail / subtitle text. */
export const Detail = keyword("Detail", /detail/);
/** Header caption text. */
export const Caption = keyword("Caption", /caption/);
/** Circle / ellipse center point (`center`). */
export const Center = keyword("Center", /center/);
/** Elbow bend waypoint (`via`). */
export const Via = keyword("Via", /via/);
/** Elbow first-segment axis (`axis`). */
export const Axis = keyword("Axis", /axis/);
/** Stack direction (`direction`). */
export const Direction = keyword("Direction", /direction/);
/** Grid column count (`cols`). */
export const Cols = keyword("Cols", /cols/);
/** Stack / grid gap (`gap`). */
export const Gap = keyword("Gap", /gap/);
/** Nested child render declarations (`children { ... }`). */
export const Children = keyword("Children", /children/);
export const Camera = keyword("Camera", /camera/);
export const At = keyword("At", /at/);
export const Frame = keyword("Frame", /frame/);
export const Zoom = keyword("Zoom", /zoom/);
export const Timeline = keyword("Timeline", /timeline/);
export const Reveal = keyword("Reveal", /reveal/);
export const Trace = keyword("Trace", /trace/);
/** Timeline `fade` cue action. */
export const Fade = keyword("Fade", /fade/);
/** Timeline `exit` cue action. */
export const Exit = keyword("Exit", /exit/);
/** Timeline `stagger` cue action. */
export const Stagger = keyword("Stagger", /stagger/);
/** Timeline `enter-children` cue action (sugar for stagger-enter on group children). */
export const EnterChildren = keyword("EnterChildren", /enter-children/);
/** Stagger member id list (`targets [a, b]`). */
export const Targets = keyword("Targets", /targets/);
/** Stagger step delay (`step 80`). */
export const Step = keyword("Step", /step/);
/** Per-cue easing (`easing ease-out`). */
export const Easing = keyword("Easing", /easing/);
export const Linear = keyword("Linear", /linear/);
export const EaseInOut = keyword("EaseInOut", /ease-in-out/);
export const EaseIn = keyword("EaseIn", /ease-in/);
export const EaseOut = keyword("EaseOut", /ease-out/);
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
export const Explainer = keyword("Explainer", /explainer/);
export const Slide = keyword("Slide", /slide/);

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
/** Literal `@` for `@scene` and `@theme.role` explainer package scenes. */
export const AtSign = createToken({ name: "AtSign", pattern: /@/ });
/** Hex color literals (`#rgb` / `#rrggbb` / `#rrggbbaa`) inside package scenes. */
export const HexColor = createToken({
  name: "HexColor",
  pattern: /#(?:[0-9a-fA-F]{8}|[0-9a-fA-F]{6}|[0-9a-fA-F]{3,4})\b/,
});

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
  Panel,
  Header,
  Circle,
  Ellipse,
  Arrow,
  Elbow,
  Bracket,
  Callout,
  Stack,
  Grid,
  Pad,
  Signal,
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
  Title,
  Detail,
  Caption,
  Center,
  Via,
  Axis,
  Direction,
  Cols,
  Gap,
  Children,
  Camera,
  At,
  Frame,
  Zoom,
  Timeline,
  Reveal,
  Trace,
  Fade,
  Exit,
  Stagger,
  EnterChildren,
  Targets,
  Step,
  Easing,
  Linear,
  EaseInOut,
  EaseIn,
  EaseOut,
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
  Explainer,
  Slide,
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
  AtSign,
  HexColor,
  ComponentIdentifier,
  Identifier,
];

export const flowLexer = new Lexer(allTokens, {
  ensureOptimizations: true,
  positionTracking: "full",
});
