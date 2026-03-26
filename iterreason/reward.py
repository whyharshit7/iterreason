"""
Answer extraction from model completions and GRPO reward functions.

Extraction cascade:
  1.  \\boxed{…}
  2.  #### number  (GSM8K style)
  3.  "the answer is …"
  4.  last number in text  (fallback)
"""

import re
from typing import List, Optional


# ═══════════════════════════════════════════════════════════════
#  Extraction
# ═══════════════════════════════════════════════════════════════

def extract_boxed(text: str) -> Optional[str]:
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    idx += len("\\boxed{")
    depth, buf = 1, []
    while idx < len(text) and depth > 0:
        ch = text[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        buf.append(ch)
        idx += 1
    return "".join(buf).strip() or None


def extract_answer(text: str) -> Optional[str]:
    """Try every extraction strategy, return normalised answer or None."""
    # 1. \\boxed{…}
    ans = extract_boxed(text)
    if ans is not None:
        return normalise(ans)

    # 2. #### number
    m = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if m:
        return normalise(m.group(1))

    # 3. "the (final) answer is …"
    m = re.search(
        r"(?:the\s+)?(?:final\s+)?answer\s+is[:\s]+(.+?)(?:\.|$)",
        text,
        re.IGNORECASE,
    )
    if m:
        return normalise(m.group(1).strip())

    # 4. last number
    nums = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    if nums:
        return normalise(nums[-1])

    return None


# ═══════════════════════════════════════════════════════════════
#  Normalisation & comparison
# ═══════════════════════════════════════════════════════════════

def normalise(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().strip("$").replace(",", "").replace(" ", "")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\(?:mathrm|mathbf)\{([^}]*)\}", r"\1", s)
    # fraction shorthand
    m = re.match(r"^\\frac\{([^}]+)\}\{([^}]+)\}$", s)
    if m:
        try:
            s = str(float(m.group(1)) / float(m.group(2)))
        except (ValueError, ZeroDivisionError):
            pass
    try:
        v = float(s)
        return str(int(v)) if v == int(v) else str(v)
    except ValueError:
        return s.lower()


def answers_match(predicted: str, ground_truth: str) -> bool:
    p, g = normalise(predicted), normalise(ground_truth)
    if p == g:
        return True
    # numeric closeness
    try:
        return abs(float(p) - float(g)) < 1e-6
    except (ValueError, TypeError):
        pass
    # optional sympy fallback (heavy, only if installed)
    try:
        from sympy import simplify, sympify
        return simplify(sympify(p) - sympify(g)) == 0
    except Exception:
        pass
    return False


# ═══════════════════════════════════════════════════════════════
#  GRPO reward functions
# ═══════════════════════════════════════════════════════════════
# Signature required by trl GRPOTrainer:
#   fn(completions: list[str], **dataset_columns) -> list[float]

def correctness_reward_fn(
    completions: List[str],
    answer: List[str],
    **kwargs,
) -> List[float]:
    """1.0 if the extracted answer is correct, else 0.0."""
    rewards = []
    for comp, gt in zip(completions, answer):
        pred = extract_answer(comp)
        rewards.append(1.0 if pred and answers_match(pred, gt) else 0.0)
    return rewards


def format_reward_fn(
    completions: List[str],
    **kwargs,
) -> List[float]:
    """Small bonus for using \\boxed{} (encourages structured output)."""
    return [0.1 if "\\boxed{" in c else 0.0 for c in completions]


def conciseness_reward_fn(
    completions: List[str],
    **kwargs,
) -> List[float]:
    """Mild penalty if the completion is excessively long."""
    rewards = []
    for c in completions:
        tok_est = len(c.split())
        if tok_est > 500:
            rewards.append(max(-0.05 * (tok_est - 500) / 500, -0.2))
        else:
            rewards.append(0.0)
    return rewards
