"""
Answer extraction from model completions and GRPO reward functions.

Extraction cascade:
  1.  \\boxed{…}
  2.  #### number  (GSM8K style)
  3.  "the answer is …"
  4.  last number in text  (fallback)
"""

import math
import re
from typing import Any, List, Optional


# ═══════════════════════════════════════════════════════════════
#  Extraction
# ═══════════════════════════════════════════════════════════════

_BOXED_RE = re.compile(r"\\(?:boxed|fbox)\s*\{")


def extract_boxed(text: str) -> Optional[str]:
    """Pull content from the *last* \\boxed{…}, handling nested braces.

    Returns None if there is no \\boxed{ or its braces never close
    (e.g. the completion was truncated mid-answer).
    """
    matches = list(_BOXED_RE.finditer(text))
    if not matches:
        return None
    idx = matches[-1].end()
    depth, buf = 1, []
    while idx < len(text):
        ch = text[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(buf).strip() or None
        buf.append(ch)
        idx += 1
    return None  # unbalanced braces → treat as no answer


_ANSWER_IS_RE = re.compile(
    r"answer\s+is\s*:?\s*(.+?)\s*(?:\.\s|\.?\s*$)",
    re.IGNORECASE | re.MULTILINE,
)
_NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def extract_answer(text: str) -> Optional[str]:
    """Try every extraction strategy, return normalised answer or None."""
    if not text:
        return None

    # 1. \\boxed{…}
    ans = extract_boxed(text)
    if ans is not None:
        return normalise(ans) or None

    # 2. #### number
    m = re.search(r"####\s*(-?[\d,]*\.?\d+)", text)
    if m:
        return normalise(m.group(1)) or None

    # 3. "the (final) answer is …"  — use the last occurrence
    hits = _ANSWER_IS_RE.findall(text)
    if hits:
        hit = hits[-1]
        # "the answer is 42 apples" → "42"
        num_then_words = re.match(r"^\$?(-?[\d,]*\.?\d+)\s+[a-zA-Z]", hit)
        if num_then_words:
            hit = num_then_words.group(1)
        return normalise(hit) or None

    # 4. last number
    nums = _NUMBER_RE.findall(text)
    if nums:
        return normalise(nums[-1]) or None

    return None


def completion_to_text(completion: Any) -> str:
    """GRPOTrainer passes plain strings for plain-text prompts but a list
    of chat messages (``[{"role": "assistant", "content": ...}]``) for
    conversational prompts.  Normalise both to a string."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    if isinstance(completion, (list, tuple)):
        return "".join(completion_to_text(c) for c in completion)
    return str(completion)


# ═══════════════════════════════════════════════════════════════
#  Normalisation & comparison
# ═══════════════════════════════════════════════════════════════

def _fmt_number(v: float) -> Optional[str]:
    if not math.isfinite(v):
        return None
    if v == int(v):
        return str(int(v))
    return repr(v)


def normalise(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    # unwrap \text{…}, \mathrm{…}, \mathbf{…}
    s = re.sub(r"\\(?:text|textbf|mathrm|mathbf)\{([^{}]*)\}", r"\1", s)
    # strip LaTeX noise that never changes the value
    for tok in ("\\left", "\\right", "\\!", "\\,", "\\;", "\\:",
                "\\$", "\\%", "%", "^\\circ", "^{\\circ}", "\\displaystyle"):
        s = s.replace(tok, "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    # \frac12 → \frac{1}{2}
    s = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", s)
    s = s.replace(",", "").replace(" ", "").strip("$").rstrip(".")
    # "x=5" → "5"  (single simple assignment only)
    m = re.fullmatch(r"[a-zA-Z]\w*=(.+)", s)
    if m and "=" not in m.group(1):
        s = m.group(1)

    # pure fraction → decimal
    m = re.fullmatch(r"(-?)\\frac\{(-?[\d.]+)\}\{(-?[\d.]+)\}", s)
    if m:
        try:
            v = float(m.group(2)) / float(m.group(3))
            s = str(-v if m.group(1) else v)
        except (ValueError, ZeroDivisionError):
            pass
    # a/b plain fraction
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)", s)
    if m:
        try:
            s = str(float(m.group(1)) / float(m.group(2)))
        except ZeroDivisionError:
            pass

    try:
        out = _fmt_number(float(s))
        if out is not None:
            return out
    except ValueError:
        pass
    return s.lower()


# Only hand strings to sympy that are plainly arithmetic/symbolic: sympify
# goes through eval(), and simplify() can take unbounded time on adversarial
# input (e.g. power towers), which would stall GRPO training.
_SAFE_SYMPY_RE = re.compile(r"^[0-9a-z+\-*/().^]+$")
_ALLOWED_WORDS = {"sqrt", "pi", "e", "i", "x", "y", "z", "n", "k"}
_MAX_SYMPY_LEN = 60


def _latex_to_sympy(s: str) -> str:
    s = s.replace("\\pi", "pi")
    s = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", s)
    s = re.sub(r"\\sqrt(\d)", r"sqrt(\1)", s)
    for _ in range(3):  # a few levels of nested \frac
        s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", s)
    s = s.replace("{", "(").replace("}", ")").replace("\\cdot", "*")
    s = s.replace("\\times", "*")
    return s


def _sympy_equal(p: str, g: str) -> bool:
    p, g = _latex_to_sympy(p), _latex_to_sympy(g)
    for s in (p, g):
        if len(s) > _MAX_SYMPY_LEN or not _SAFE_SYMPY_RE.match(s):
            return False
        if not set(re.findall(r"[a-z]+", s)) <= _ALLOWED_WORDS:
            return False
        # forbid chained / large exponents (power-tower blow-ups)
        if len(re.findall(r"\^|\*\*", s)) > 1 or re.search(r"(\^|\*\*)\(?\d{3,}", s):
            return False
    try:
        from sympy import simplify
        from sympy.parsing.sympy_parser import (
            convert_xor,
            implicit_multiplication,
            parse_expr,
            standard_transformations,
        )
        tf = standard_transformations + (implicit_multiplication, convert_xor)
        diff = simplify(parse_expr(p, transformations=tf)
                        - parse_expr(g, transformations=tf))
        return bool(diff == 0)
    except Exception:
        return False


def answers_match(predicted: Optional[str], ground_truth: Optional[str]) -> bool:
    p, g = normalise(predicted), normalise(ground_truth)
    if not p or not g:
        return False
    if p == g:
        return True
    # numeric closeness
    try:
        pf, gf = float(p), float(g)
        return math.isclose(pf, gf, rel_tol=1e-9, abs_tol=1e-6)
    except (ValueError, TypeError):
        pass
    # symbolic fallback (guarded)
    return _sympy_equal(p, g)


# ═══════════════════════════════════════════════════════════════
#  GRPO reward functions
# ═══════════════════════════════════════════════════════════════
# Signature required by trl GRPOTrainer:
#   fn(completions, **dataset_columns) -> list[float]
# `completions` is a list of strings or of chat-message lists.

def correctness_reward_fn(
    completions: List[Any],
    answer: List[str],
    **kwargs,
) -> List[float]:
    """1.0 if the extracted answer is correct, else 0.0."""
    rewards = []
    for comp, gt in zip(completions, answer):
        pred = extract_answer(completion_to_text(comp))
        rewards.append(1.0 if pred and answers_match(pred, gt) else 0.0)
    return rewards


def format_reward_fn(
    completions: List[Any],
    **kwargs,
) -> List[float]:
    """Small bonus for a well-formed \\boxed{} answer."""
    return [
        0.1 if extract_boxed(completion_to_text(c)) is not None else 0.0
        for c in completions
    ]


def conciseness_reward_fn(
    completions: List[Any],
    **kwargs,
) -> List[float]:
    """Mild penalty if the completion is excessively long."""
    rewards = []
    for c in completions:
        tok_est = len(completion_to_text(c).split())
        if tok_est > 500:
            rewards.append(max(-0.05 * (tok_est - 500) / 500, -0.2))
        else:
            rewards.append(0.0)
    return rewards
