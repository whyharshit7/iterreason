"""
Load GSM8K and MATH, normalise into a unified schema:
    prompt  – list[dict]  (chat messages, system+user)
    answer  – str         (normalised ground-truth answer)
    source  – str         (dataset name)
"""

import re
from typing import Dict

from datasets import Dataset, concatenate_datasets, load_dataset

from .utils import log

SYSTEM_PROMPT = (
    "You are a helpful math assistant. "
    "Solve the problem step by step, showing clear reasoning. "
    "Put your final answer inside \\boxed{}."
)


# ── Answer extraction helpers ──────────────────────────────────

def _extract_boxed(text: str) -> str | None:
    """Pull content from the *last* \\boxed{…}, handling nested braces."""
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


def _normalise_number(s: str) -> str:
    s = s.replace(",", "").replace(" ", "").strip().strip("$")
    try:
        v = float(s)
        return str(int(v)) if v == int(v) else str(v)
    except ValueError:
        return s


# ── Dataset loaders ────────────────────────────────────────────

def load_gsm8k(split: str = "train", n: int = -1) -> Dataset:
    ds = load_dataset("openai/gsm8k", "main", split=split)
    if 0 < n < len(ds):
        ds = ds.select(range(n))

    def _proc(ex):
        match = re.search(r"####\s*(-?[\d,]+\.?\d*)", ex["answer"])
        ans = _normalise_number(match.group(1)) if match else ""
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ex["question"]},
            ],
            "answer": ans,
            "source": "gsm8k",
        }

    return ds.map(_proc, remove_columns=ds.column_names)


def load_math_dataset(split: str = "train", n: int = -1) -> Dataset:
    ds = load_dataset("hendrycks/competition_math", split=split)
    if 0 < n < len(ds):
        ds = ds.select(range(n))

    def _proc(ex):
        ans = _extract_boxed(ex["solution"]) or ""
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ex["problem"]},
            ],
            "answer": ans,
            "source": "math",
        }

    return ds.map(_proc, remove_columns=ds.column_names)


# ── Public API ─────────────────────────────────────────────────

def load_training_data(config) -> Dataset:
    parts = []
    if "gsm8k" in config.task_datasets:
        d = load_gsm8k("train", config.gsm8k_train_samples)
        log(f"GSM8K train: {len(d)} examples")
        parts.append(d)
    if "math" in config.task_datasets:
        d = load_math_dataset("train", config.math_train_samples)
        log(f"MATH train : {len(d)} examples")
        parts.append(d)
    combined = concatenate_datasets(parts).shuffle(seed=config.seed)
    log(f"Combined train: {len(combined)} examples")
    return combined


def load_eval_data(config) -> Dict[str, Dataset]:
    out = {}
    if "gsm8k" in config.task_datasets:
        d = load_gsm8k("test", config.eval_samples)
        log(f"GSM8K test : {len(d)} examples")
        out["gsm8k"] = d
    if "math" in config.task_datasets:
        d = load_math_dataset("test", config.eval_samples)
        log(f"MATH test  : {len(d)} examples")
        out["math"] = d
    return out
