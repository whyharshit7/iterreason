"""
Evaluate a model checkpoint on GSM8K / MATH test sets.

Metrics reported per dataset:
    accuracy          –  greedy pass@1
    avg_tokens        –  mean completion length (proxy for CoT efficiency)
    extraction_rate   –  fraction where we could parse an answer
    pass@k            –  unbiased estimator  (optional, slow)
"""

import math
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .reward import answers_match, extract_answer
from .utils import log, log_metrics_table


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
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
        device_map="auto",
    )
    model.eval()

    all_metrics: Dict[str, float] = {}
    for name, ds in eval_datasets.items():
        m = _eval_dataset(
            model, tokenizer, ds, name, config, compute_pass_k=full,
        )
        for k, v in m.items():
            all_metrics[f"{name}/{k}"] = v

    del model
    torch.cuda.empty_cache()

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

    pass_k_records: List[tuple] = []  # (n_samples, n_correct)

    tokenizer.padding_side = "left"

    for ex in tqdm(dataset, desc=f"eval-{dataset_name}"):
        prompt_text = tokenizer.apply_chat_template(
            ex["prompt"], tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=config.grpo_max_prompt_length,
        ).to(model.device)
        prompt_len = inputs.input_ids.shape[1]
        gt = ex["answer"]
        total += 1

        # ── greedy pass@1 ──────────────────────────────────────
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=config.grpo_max_completion_length,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        comp_ids = out[0][prompt_len:]
        comp = tokenizer.decode(comp_ids, skip_special_tokens=True)
        total_tokens += len(comp_ids)

        pred = extract_answer(comp)
        if pred is not None:
            extracted += 1
            if answers_match(pred, gt):
                correct += 1

        # ── sampled pass@k ─────────────────────────────────────
        if compute_pass_k:
            n = config.eval_num_samples_for_pass_k
            c = 0
            for _ in range(n):
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=config.grpo_max_completion_length,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.95,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                s_comp = tokenizer.decode(
                    out[0][prompt_len:], skip_special_tokens=True,
                )
                s_pred = extract_answer(s_comp)
                if s_pred and answers_match(s_pred, gt):
                    c += 1
            pass_k_records.append((n, c))

    metrics = {
        "accuracy": correct / total if total else 0,
        "avg_tokens": total_tokens / total if total else 0,
        "extraction_rate": extracted / total if total else 0,
        "n_examples": total,
    }

    if compute_pass_k:
        for k in config.eval_pass_at_k:
            vals = [_pass_at_k(n, c, k) for n, c in pass_k_records]
            metrics[f"pass@{k}"] = float(np.mean(vals))

    return metrics


def _pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator:  1 − C(n−c, k) / C(n, k)."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)
