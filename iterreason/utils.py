"""
Shared helpers: logging, seed, checkpointing, metrics I/O.
"""

import json
import logging
import os
import random
import sys
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import torch

logger = logging.getLogger("iterreason")


# ── Logging ────────────────────────────────────────────────────

def setup_logging(config):
    """Configure console + file logging."""
    os.makedirs(config.output_dir, exist_ok=True)
    log_file = os.path.join(
        config.output_dir,
        f"run_{datetime.now():%Y%m%d_%H%M%S}.log",
    )

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)

    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(fh)
    logger.addHandler(sh)

    logger.info(f"Logging to {log_file}")


def log(msg: str, level: str = "info"):
    getattr(logger, level)(msg)


# ── Reproducibility ───────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    log(f"Seed set to {seed}")


# ── Metrics I/O ───────────────────────────────────────────────

def save_metrics(metrics: Dict[str, Any], output_dir: str, tag: str):
    path = os.path.join(output_dir, f"metrics_{tag}.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    log(f"Metrics saved → {path}")


def save_all_metrics(records: List[Dict], output_dir: str):
    path = os.path.join(output_dir, "all_metrics.json")
    with open(path, "w") as f:
        json.dump(records, f, indent=2, default=float)
    log(f"Full metrics log → {path}")


def log_metrics_table(metrics: Dict[str, Any], header: str = ""):
    """Pretty-print a flat metrics dict."""
    if header:
        log(f"\n┌── {header}")
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            log(f"│  {k:40s}  {v:.4f}")
        else:
            log(f"│  {k:40s}  {v}")
    log("└" + "─" * 50)


# ── Model helpers ─────────────────────────────────────────────

def merge_lora_and_save(model, tokenizer, output_dir: str):
    """Merge LoRA adapters back into base weights and persist."""
    log("Merging LoRA weights …")
    merged = model.merge_and_unload()
    merged.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    log(f"Merged model saved → {output_dir}")
    del merged
    torch.cuda.empty_cache()
