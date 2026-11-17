"""
Evaluate a model checkpoint on GSM8K / MATH test sets.

Metrics reported per dataset:
    accuracy          –  greedy pass@1
    avg_tokens        –  mean completion length (proxy for CoT efficiency)
    extraction_rate   –  fraction where we could parse an answer
    pass@k            –  unbiased estimator  (optional, slow)
"""

import math
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from .generation import generate, load_model_for_generation
from .reward import answers_match, extract_answer
from .utils import free_memory, log, log_metrics_table


def evaluate_model(
    model_path: str,
    eval_datasets: Dict[str, "Dataset"],
    config,
    full: bool = False,
) -> Dict[str, float]:
    """
    Parameters
    ----------
    full : bool
        If True, also compute pass@k (expensive).
    """
    model, tokenizer = load_model_for_generation(model_path, config)

    all_metrics: Dict[str, float] = {}
    try:
        for name, ds in eval_datasets.items():
            m = _eval_dataset(
                model, tokenizer, ds, name, config, compute_pass_k=full,
            )
            for k, v in m.items():
                all_metrics[f"{name}/{k}"] = v
    finally:
        del model
        free_memory()

    log_metrics_table(all_metrics, header="Evaluation")
    return all_metrics


# ── Internal ───────────────────────────────────────────────────

def _eval_dataset(
    model,
    tokenizer,
    dataset,
    dataset_name: str,
    config,
    compute_pass_k: bool = False,
) -> Dict[str, float]:
    correct = 0
    total = 0
    total_tokens = 0
    extracted = 0

    pass_k_records: List[Tuple[int, int]] = []  # (n_samples, n_correct)
    bs = config.eval_batch_size
    common = dict(
        max_prompt_length=config.grpo_max_prompt_length,
        max_new_tokens=config.grpo_max_completion_length,
    )

    for start in tqdm(range(0, len(dataset), bs), desc=f"eval-{dataset_name}"):
        batch = dataset.select(range(start, min(start + bs, len(dataset))))
        prompts = batch["prompt"]
        gts = batch["answer"]

        # ── greedy pass@1 ──────────────────────────────────────
        greedy = generate(model, tokenizer, prompts, do_sample=False, **common)
        for (comp, n_tok), gt in zip((g[0] for g in greedy), gts):
            total += 1
            total_tokens += n_tok
            pred = extract_answer(comp)
            if pred is not None:
                extracted += 1
                if answers_match(pred, gt):
                    correct += 1

        # ── sampled pass@k ─────────────────────────────────────
        if compute_pass_k:
            n = config.eval_num_samples_for_pass_k
            # keep ≈ eval_batch_size sequences in flight per generate call
            sub = max(1, bs // n)
            sampled = []
            for i in range(0, len(prompts), sub):
                sampled += generate(
                    model, tokenizer, prompts[i:i + sub],
                    num_return_sequences=n,
                    do_sample=True,
                    temperature=config.eval_temperature,
                    top_p=0.95,
                    **common,
                )
            for samples, gt in zip(sampled, gts):
                c = 0
                for s_comp, _ in samples:
                    s_pred = extract_answer(s_comp)
                    if s_pred and answers_match(s_pred, gt):
                        c += 1
                pass_k_records.append((n, c))

    metrics = {
        "accuracy": correct / total if total else 0.0,
        "avg_tokens": total_tokens / total if total else 0.0,
        "extraction_rate": extracted / total if total else 0.0,
        "n_examples": total,
    }

    if compute_pass_k and pass_k_records:
        for k in config.eval_pass_at_k:
            vals = [_pass_at_k(n, c, k) for n, c in pass_k_records]
            metrics[f"pass@{k}"] = float(np.mean(vals))

    return metrics


def _pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator:  1 − C(n−c, k) / C(n, k)."""
    if k > n:
        raise ValueError(f"pass@{k} needs at least {k} samples, got {n}")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)
