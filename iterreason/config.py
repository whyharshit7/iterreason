"""
Central configuration for every knob in the pipeline.
One dataclass to rule them all — pass it everywhere.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class IterReasonConfig:
    # ── Model ──────────────────────────────────────────────────
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )

    # ── Pipeline ───────────────────────────────────────────────
    num_iterations: int = 3          # how many GRPO→Distill cycles

    # ── GRPO hyper-parameters ──────────────────────────────────
    grpo_num_generations: int = 8    # rollouts per prompt (G)
    grpo_learning_rate: float = 5e-7
    grpo_num_train_epochs: int = 1
    grpo_per_device_batch_size: int = 2
    grpo_gradient_accumulation_steps: int = 8
    grpo_max_completion_length: int = 1024
    grpo_max_prompt_length: int = 512
    grpo_beta: float = 0.04         # KL penalty
    grpo_temperature: float = 0.7

    # ── Distillation hyper-parameters ──────────────────────────
    distill_best_of_k: int = 8      # sample k, keep best correct
    distill_learning_rate: float = 2e-5
    distill_num_train_epochs: int = 2
    distill_per_device_batch_size: int = 4
    distill_gradient_accumulation_steps: int = 4
    distill_max_seq_length: int = 2048
    distill_temperature: float = 0.7
    distill_generation_batch_size: int = 4  # prompts per forward

    # ── Data ───────────────────────────────────────────────────
    task_datasets: List[str] = field(
        default_factory=lambda: ["gsm8k", "math"]
    )
    gsm8k_train_samples: int = -1   # -1 = use all
    math_train_samples: int = -1

    # ── Evaluation ─────────────────────────────────────────────
    eval_samples: int = -1
    eval_pass_at_k: List[int] = field(
        default_factory=lambda: [1, 4, 8]
    )
    eval_num_samples_for_pass_k: int = 16
    full_eval_only_final: bool = True  # only do pass@k at the end

    # ── Infrastructure ─────────────────────────────────────────
    output_dir: str = "outputs"
    seed: int = 42
    bf16: bool = True
    gradient_checkpointing: bool = True
    use_vllm: bool = False           # vLLM for distill-time gen
    vllm_gpu_utilization: float = 0.9

    # ── Logging ────────────────────────────────────────────────
    use_wandb: bool = True
    wandb_project: str = "IterReason"
    logging_steps: int = 10
    save_steps: int = 200
