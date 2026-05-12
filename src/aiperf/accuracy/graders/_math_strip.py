# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LaTeX/math answer normalization for the math grader.

Ported faithfully from the trt-llm benchmark recipe's
``parser.strip_string`` (``src/accuracy/aime/parser.py``), which itself
descends from Hendrycks' MATH release and the ToRA/DeepSeek-Math
evaluation utilities. The intent is character-for-character output
parity with that pipeline so the math grader behaves identically to the
trt-llm reference.

Two minor deviations from the upstream:
- ``convert_word_number`` is a no-op stub here; the recipe pulls in the
  optional ``word2number`` package, but AIME answers are bounded
  integers (0-999) for which word-number conversion never fires. If a
  caller needs it, install ``word2number`` and replace the stub.
- The ``_fix_a_slash_b`` helper is only invoked on strings that look
  like simple ratios; we skip the recipe's blanket ``\\frac`` rewrite
  for non-numeric a/b strings.

Reference:
    trt-llm-benchmark-recipe/src/accuracy/aime/parser.py:212 (strip_string)
"""

from __future__ import annotations

import re

# MathQA-derived unit list. The trt-llm strip_string strips these
# tokens from MATH-style answers (e.g. "5 mph" → "5"). For AIME (pure
# integer answers) most never fire, but we keep the full list for
# parity so the grader behaves identically across MATH-family inputs.
_UNIT_TEXTS_BASE: list[str] = [
    "east",
    "degree",
    "mph",
    "kmph",
    "ft",
    "m square",
    " m east",
    "sq m",
    "deg",
    "mile",
    "q .",
    "monkey",
    "prime",
    "ratio",
    "profit of rs",
    "rd",
    "o",
    "gm",
    "p . m",
    "lb",
    "tile",
    "per",
    "dm",
    "lt",
    "gain",
    "ab",
    "way",
    "west",
    "a .",
    "b .",
    "c .",
    "d .",
    "e .",
    "f .",
    "g .",
    "h .",
    "t",
    "a",
    "h",
    "no change",
    "men",
    "soldier",
    "pie",
    "bc",
    "excess",
    "st",
    "inches",
    "noon",
    "percent",
    "by",
    "gal",
    "kmh",
    "c",
    "acre",
    "rise",
    "a . m",
    "th",
    "π r 2",
    "sq",
    "mark",
    "l",
    "toy",
    "coin",
    "sq . m",
    "gallon",
    "° f",
    "profit",
    "minw",
    "yr",
    "women",
    "feet",
    "am",
    "pm",
    "hr",
    "cu cm",
    "square",
    "v â € ™",
    "are",
    "rupee",
    "rounds",
    "cubic",
    "cc",
    "mtr",
    "s",
    "ohm",
    "number",
    "kmph",
    "day",
    "hour",
    "minute",
    "min",
    "second",
    "man",
    "woman",
    "sec",
    "cube",
    "mt",
    "sq inch",
    "mp",
    "∏ cm ³",
    "hectare",
    "more",
    "sec",
    "unit",
    "cu . m",
    "cm 2",
    "rs .",
    "rs",
    "kg",
    "g",
    "month",
    "km",
    "m",
    "cm",
    "mm",
    "apple",
    "liter",
    "loss",
    "yard",
    "pure",
    "year",
    "increase",
    "decrease",
    "d",
    "less",
    "Surface",
    "litre",
    "pi sq m",
    "s .",
    "metre",
    "meter",
    "inch",
]
# Append plural forms — the recipe does this at module import time.
_UNIT_TEXTS: list[str] = _UNIT_TEXTS_BASE + [t + "s" for t in _UNIT_TEXTS_BASE]


def _fix_fracs(string: str) -> str:
    """Normalize ``\\fracXY`` to ``\\frac{X}{Y}`` — port of recipe helper."""
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                if len(substr) < 2:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    return new_str


def _fix_a_slash_b(string: str) -> str:
    """Convert simple integer-ratio ``a/b`` to ``\\frac{a}{b}``."""
    parts = string.split("/")
    if len(parts) != 2:
        return string
    a, b = parts
    try:
        if "sqrt" not in a:
            a = int(a)
        if "sqrt" not in b:
            b = int(b)
        return f"\\frac{{{a}}}{{{b}}}"
    except (ValueError, TypeError):
        return string


def _fix_sqrt(string: str) -> str:
    """Normalize ``\\sqrtN`` to ``\\sqrt{N}`` — port of recipe helper."""
    return re.sub(r"\\sqrt(\w+)", r"\\sqrt{\1}", string)


def _convert_word_number(text: str) -> str:
    """No-op stub for the recipe's word2number conversion.

    The recipe imports ``word2number.w2n`` and turns "two" into "2".
    For AIME (integer answers) and most MATH-500 cases this never
    triggers, so we skip the dependency. If a caller needs it, swap
    this implementation for one that uses ``w2n.word_to_num``.
    """
    return text


def strip_string(string: str) -> str:
    """Normalize a math/LaTeX answer string for downstream comparison.

    Mirrors trt-llm-benchmark-recipe/src/accuracy/aime/parser.py
    line-for-line, with two documented deviations:
    1. ``_convert_word_number`` is a no-op (no ``word2number`` dep).
    2. The recipe's ``string.replace("'", "")`` / ``string.replace('"', "")``
       calls are no-ops there too (they don't reassign), and we keep
       them as no-ops for parity.

    Returns an empty string when input is empty after normalization.
    """
    string = str(string).strip()
    string = string.replace("\n", "")
    string = string.rstrip(".")

    string = string.replace("\\!", "")

    # matrix
    string = re.sub(r"\\begin\{array\}\{.*?\}", r"\\begin{pmatrix}", string)
    string = re.sub(r"\\end\{array\}", r"\\end{pmatrix}", string)
    string = string.replace("bmatrix", "pmatrix")

    # fraction variants
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = (
        string.replace("\\neq", "\\ne")
        .replace("\\leq", "\\le")
        .replace("\\geq", "\\ge")
    )

    # spacing and quote-style braces
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("\\{", "{")
    string = string.replace("\\}", "}")

    # Remove trailing \text{...}
    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    if _string and _string != string:
        string = _string

    # Strip MathQA unit tokens (with non-alphanumeric prefix/suffix
    # boundaries so we don't eat parts of words).
    for unit_text in _UNIT_TEXTS:
        _string = re.sub(r"(^|\W)" + unit_text + r"($|\W)", r"\1\2", string)
        if _string:
            string = _string

    # Degree symbol variants
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # Currency
    string = string.replace("\\$", "")
    string = string.replace("$", "")
    string = string.replace("\\(", "").replace("\\)", "")

    # Word-number conversion (no-op stub)
    string = _convert_word_number(string)

    # \text{x} → x
    string = re.sub(r"\\text\{(.*?)\}", r"\1", string)

    # Drop common variable assignments
    for key in ("x=", "y=", "z=", "x\\in", "y\\in", "z\\in", "x\\to", "y\\to", "z\\to"):
        string = string.replace(key, "")
    string = string.replace("\\emptyset", r"{}")
    string = string.replace("(-\\infty,\\infty)", "\\mathbb{R}")

    # Percentages
    string = string.replace("\\%", "")
    string = string.replace("\\%", "")
    string = string.replace("%", "")

    # Months
    months = (
        r"\b(January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\b"
    )
    string = re.sub(months, "", string, flags=re.IGNORECASE)

    # Add leading 0 to bare ".5" → "0.5"
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    # Strip wrapping braces/parens around alphanumeric content.
    #
    # NOTE: This block is verbatim from the trt-llm recipe
    # (``parser.py`` lines 292-303) including the latent bug —
    # ``string.isalnum()`` checks the WHOLE string, which always
    # includes the wrapper characters, so the condition can never be
    # True and the branch is effectively dead. We deliberately keep
    # the dead branch to preserve byte-equal parity with the recipe's
    # ``strip_string``; rewriting to ``string[1:-1].isalnum()`` would
    # cause divergent strip output (e.g. ``"{12}" -> "12"`` here vs.
    # unchanged in the recipe), which would then change downstream
    # ``math_equal`` numeric-comparison decisions.
    if (
        string.startswith("{")
        and string.endswith("}")
        and string.isalnum()
        or string.startswith("(")
        and string.endswith(")")
        and string.isalnum()
        or string.startswith("[")
        and string.endswith("]")
        and string.isalnum()
    ):
        string = string[1:-1]

    # Infinity
    string = string.replace("infinity", "\\infty")
    if "\\infty" not in string:
        string = string.replace("inf", "\\infty")
    string = string.replace("+\\inity", "\\infty")

    # Misc keyword strip
    string = string.replace("and", "")
    string = string.replace("\\mathbf", "")
    string = re.sub(r"\\mbox{.*?}", "", string)

    # Quotes — these are no-ops in the recipe (the .replace return
    # value is discarded). Preserved as no-ops for byte-parity.
    string.replace("'", "")
    string.replace('"', "")

    # Imaginary unit normalization
    if "j" in string and "i" not in string:
        string = string.replace("j", "i")

    # Trim trailing .000... in numerics
    string = re.sub(r"(\d+)\.0*([^\d])", r"\1\2", string)
    string = re.sub(r"(\d+)\.0*$", r"\1", string)

    if not string:
        return string
    if string[0] == ".":
        string = "0" + string

    # "k = X" → "X"
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]

    string = _fix_sqrt(string)
    string = string.replace(" ", "")

    string = _fix_fracs(string)
    string = _fix_a_slash_b(string)

    return string
