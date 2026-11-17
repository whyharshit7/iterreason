"""
Load GSM8K and MATH, normalise into a unified schema:
    prompt  – list[dict]  (chat messages, system+user)
    answer  – str         (normalised ground-truth answer)
    source  – str         (dataset name)
"""

import re
from typing import Dict

from datasets import Dataset, concatenate_datasets, load_dataset

from .reward import extract_boxed, normalise
from .utils import log

SYSTEM_PROMPT = (
    "You are a helpful math assistant. "
    "Solve the problem step by step, showing clear reasoning. "
    "Put your final answer inside \\boxed{}."
)

SUPPORTED_DATASETS = ("gsm8k", "math")


def _subsample(ds: Dataset, n: int, seed: int) -> Dataset:
    """Random (seeded) subset of size n; n <= 0 means use everything.

    Shuffling first matters for MATH, whose splits are grouped by subject —
    taking the first n rows would evaluate on a single subject."""
    if 0 < n < len(ds):
        ds = ds.shuffle(seed=seed).select(range(n))
    return ds


def _drop_unanswerable(ds: Dataset, name: str) -> Dataset:
    before = len(ds)
    ds = ds.filter(lambda ex: bool(ex["answer"]))
    if len(ds) < before:
        log(f"{name}: dropped {before - len(ds)} examples with no parsable answer")
    return ds


# ── Dataset loaders ────────────────────────────────────────────

def load_gsm8k(split: str = "train", n: int = -1, seed: int = 42) -> Dataset:
    ds = load_dataset("openai/gsm8k", "main", split=split)
    ds = _subsample(ds, n, seed)

    def _proc(ex):
        match = re.search(r"####\s*(-?[\d,]*\.?\d+)", ex["answer"])
        ans = normalise(match.group(1)) if match else ""
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ex["question"]},
            ],
            "answer": ans,
            "source": "gsm8k",
        }

    ds = ds.map(_proc, remove_columns=ds.column_names)
    return _drop_unanswerable(ds, "GSM8K")


def load_math_dataset(
    split: str = "train",
    n: int = -1,
    seed: int = 42,
    dataset_name: str = "DigitalLearningGmbH/MATH-lighteval",
) -> Dataset:
    ds = load_dataset(dataset_name, split=split)
    ds = _subsample(ds, n, seed)

    def _proc(ex):
        ans = extract_boxed(ex["solution"]) or ""
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ex["problem"]},
            ],
            "answer": ans,
            "source": "math",
        }

    ds = ds.map(_proc, remove_columns=ds.column_names)
    return _drop_unanswerable(ds, "MATH")


# ── Public API ─────────────────────────────────────────────────

def load_training_data(config) -> Dataset:
    parts = []
    if "gsm8k" in config.task_datasets:
        d = load_gsm8k("train", config.gsm8k_train_samples, config.seed)
        log(f"GSM8K train: {len(d)} examples")
        parts.append(d)
    if "math" in config.task_datasets:
        d = load_math_dataset(
            "train", config.math_train_samples, config.seed, config.math_dataset,
        )
        log(f"MATH train : {len(d)} examples")
        parts.append(d)
    if not parts:
        raise ValueError(
            f"No supported datasets in task_datasets={config.task_datasets}; "
            f"choose from {SUPPORTED_DATASETS}"
        )
    combined = concatenate_datasets(parts).shuffle(seed=config.seed)
    log(f"Combined train: {len(combined)} examples")
    return combined


def load_eval_data(config) -> Dict[str, Dataset]:
    out = {}
    if "gsm8k" in config.task_datasets:
        d = load_gsm8k("test", config.eval_samples, config.seed)
        log(f"GSM8K test : {len(d)} examples")
        out["gsm8k"] = d
    if "math" in config.task_datasets:
        d = load_math_dataset(
            "test", config.eval_samples, config.seed, config.math_dataset,
        )
        log(f"MATH test  : {len(d)} examples")
        out["math"] = d
    return out
