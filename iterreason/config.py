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
    grpo_use_vllm: bool = False      # vLLM (colocate) for GRPO rollouts

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
    # The original hendrycks/competition_math repo was removed from the Hub;
    # any mirror with `problem` / `solution` columns and train/test works.
    math_dataset: str = "DigitalLearningGmbH/MATH-lighteval"
    gsm8k_train_samples: int = -1   # -1 = use all
    math_train_samples: int = -1

    # ── Evaluation ─────────────────────────────────────────────
    eval_samples: int = -1
    eval_batch_size: int = 16
    eval_temperature: float = 0.7    # sampling temperature for pass@k
    eval_pass_at_k: List[int] = field(
        default_factory=lambda: [1, 4, 8]
    )
    eval_num_samples_for_pass_k: int = 16
    full_eval_only_final: bool = True  # False → pass@k after every phase

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

    def __post_init__(self):
        errors = []
        unknown = set(self.task_datasets) - {"gsm8k", "math"}
        if unknown:
            errors.append(f"unknown task_datasets {sorted(unknown)} "
                          "(supported: gsm8k, math)")
        if not self.task_datasets:
            errors.append("task_datasets must not be empty")
        if self.num_iterations < 0:
            errors.append("num_iterations must be >= 0")
        if self.grpo_num_generations < 2:
            errors.append("grpo_num_generations must be >= 2 "
                          "(GRPO advantages are computed within a group)")
        # GRPOTrainer requires each generation batch to hold whole groups.
        gen_batch = (self.grpo_per_device_batch_size
                     * self.grpo_gradient_accumulation_steps)
        if self.grpo_num_generations >= 2 and gen_batch % self.grpo_num_generations:
            errors.append(
                f"grpo_per_device_batch_size × grpo_gradient_accumulation_steps "
                f"(= {gen_batch}) must be divisible by grpo_num_generations "
                f"({self.grpo_num_generations})"
            )
        for name in ("distill_best_of_k", "eval_batch_size",
                     "distill_generation_batch_size",
                     "eval_num_samples_for_pass_k"):
            if getattr(self, name) < 1:
                errors.append(f"{name} must be >= 1")
        bad_k = [k for k in self.eval_pass_at_k
                 if k < 1 or k > self.eval_num_samples_for_pass_k]
        if bad_k:
            errors.append(
                f"eval_pass_at_k values {bad_k} must be in "
                f"[1, eval_num_samples_for_pass_k={self.eval_num_samples_for_pass_k}]"
            )
        if not 0 < self.vllm_gpu_utilization <= 1:
            errors.append("vllm_gpu_utilization must be in (0, 1]")
        if errors:
            raise ValueError("Invalid IterReasonConfig:\n  - " + "\n  - ".join(errors))
