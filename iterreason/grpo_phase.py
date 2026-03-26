"""
Phase A: Group Relative Policy Optimisation.

Uses trl.GRPOTrainer.  For each prompt the model generates G rollouts;
the reward signals which ones were correct; GRPO updates the policy
using group-level advantages (no critic needed).
"""

import os

import torch
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from .reward import (
    conciseness_reward_fn,
    correctness_reward_fn,
    format_reward_fn,
)
from .utils import log, merge_lora_and_save


def run_grpo(model_path: str, output_dir: str, train_dataset, config):
    """
    Train one GRPO round.

    Returns
    -------
    str  –  path to the saved (merged) model.
    """
    os.makedirs(output_dir, exist_ok=True)
    log(f"GRPO  model_in  = {model_path}")
    log(f"GRPO  output    = {output_dir}")
    log(f"GRPO  dataset   = {len(train_dataset)} prompts × "
        f"{config.grpo_num_generations} generations")

    # ── Training arguments ─────────────────────────────────────
    grpo_args = GRPOConfig(
        output_dir=output_dir,
        # generation
        num_generations=config.grpo_num_generations,
        max_completion_length=config.grpo_max_completion_length,
        max_prompt_length=config.grpo_max_prompt_length,
        temperature=config.grpo_temperature,
        # optimisation
        learning_rate=config.grpo_learning_rate,
        num_train_epochs=config.grpo_num_train_epochs,
        per_device_train_batch_size=config.grpo_per_device_batch_size,
        gradient_accumulation_steps=config.grpo_gradient_accumulation_steps,
        beta=config.grpo_beta,
        # precision & memory
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        # logging / saving
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=2,
        report_to="wandb" if config.use_wandb else "none",
        run_name="grpo",
        seed=config.seed,
        # vLLM backend (optional)
        use_vllm=config.use_vllm,
    )

    # ── LoRA (optional) ────────────────────────────────────────
    peft_config = None
    if config.use_lora:
        peft_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            task_type="CAUSAL_LM",
        )

    # ── Reward stack (summed by GRPOTrainer) ───────────────────
    reward_fns = [
        correctness_reward_fn,   # main signal
        format_reward_fn,        # encourage \\boxed{}
        conciseness_reward_fn,   # mild length penalty
    ]

    # ── Train ──────────────────────────────────────────────────
    trainer = GRPOTrainer(
        model=model_path,
        reward_funcs=reward_fns,
        args=grpo_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
    )

    train_result = trainer.train()
    log(f"GRPO training loss: {train_result.training_loss:.4f}")

    # ── Save merged model ──────────────────────────────────────
    merged_dir = os.path.join(output_dir, "merged")
    if config.use_lora:
        merge_lora_and_save(trainer.model, trainer.processing_class, merged_dir)
    else:
        trainer.save_model(merged_dir)
        trainer.processing_class.save_pretrained(merged_dir)

    del trainer
    torch.cuda.empty_cache()

    log(f"GRPO complete → {merged_dir}")
    return merged_dir
