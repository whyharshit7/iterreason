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

    # Boolean options take --flag / --no-flag, e.g.
    python run.py --no-use_wandb --no-use_lora
"""

import argparse
import typing
from dataclasses import MISSING, fields

from iterreason.config import IterReasonConfig
from iterreason.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="IterReason: Iterative Self-Distillation for Reasoning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # auto-generate CLI flags from every field in the config dataclass
    hints = typing.get_type_hints(IterReasonConfig)
    for f in fields(IterReasonConfig):
        flag = f"--{f.name}"
        ftype = hints[f.name]
        default = (f.default_factory() if f.default_factory is not MISSING
                   else f.default)
        if ftype is bool:
            # --flag / --no-flag  (the old store_false trick meant that
            # passing --use_wandb actually *disabled* wandb)
            parser.add_argument(
                flag, action=argparse.BooleanOptionalAction, default=default,
            )
        elif typing.get_origin(ftype) in (list, typing.List):
            inner = typing.get_args(ftype)[0]
            parser.add_argument(flag, nargs="+", type=inner, default=default)
        else:
            parser.add_argument(flag, type=ftype, default=default)

    args = parser.parse_args()
    try:
        config = IterReasonConfig(**vars(args))
    except ValueError as e:
        parser.error(str(e))

    final_model, metrics = run_pipeline(config)
    print(f"\n✓  Done.  Final model at: {final_model}")


if __name__ == "__main__":
    main()
