#!/usr/bin/env python3
"""
IterReason — Iterative Self-Distillation for Reasoning

Usage
-----
    # Full pipeline with defaults
    python run.py

    # Quick test (small data, 1 iteration)
    python run.py --num_iterations 1 \
                  --gsm8k_train_samples 200 \
                  --math_train_samples 200 \
                  --eval_samples 50

    # Use vLLM for generation (faster distillation)
    python run.py --use_vllm

    # Custom base model
    python run.py --base_model Qwen/Qwen2.5-3B-Instruct --lora_r 32
"""

import argparse
import sys
from dataclasses import fields

from iterreason.config import IterReasonConfig
from iterreason.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="IterReason: Iterative Self-Distillation for Reasoning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # auto-generate CLI flags from every field in the config dataclass
    for f in fields(IterReasonConfig):
        flag = f"--{f.name}"
        if f.type is bool:
            parser.add_argument(
                flag,
                action="store_true" if not f.default else "store_false",
                default=f.default,
            )
        elif hasattr(f.type, "__origin__"):          # List[str], List[int]
            inner = f.type.__args__[0]
            parser.add_argument(flag, nargs="+", type=inner, default=f.default)
        else:
            parser.add_argument(flag, type=f.type, default=f.default)

    args = parser.parse_args()
    config = IterReasonConfig(**vars(args))

    final_model, metrics = run_pipeline(config)
    print(f"\n✓  Done.  Final model at: {final_model}")


if __name__ == "__main__":
    main()
