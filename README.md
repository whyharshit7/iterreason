# IterReason

IterReason is a Python training pipeline for iterative reasoning improvement. It alternates between:

1. GRPO-based reinforcement learning on reasoning tasks
2. Self-distillation that keeps the shortest correct traces and fine-tunes on them

The project is designed around math-style reasoning datasets such as GSM8K and MATH, with evaluation tracked after each phase of the loop.

## Pipeline

The training loop follows this pattern:

`M0 -> GRPO -> M1 -> Distill -> M2 -> GRPO -> M3 -> Distill -> ...`

At a high level:

- GRPO samples multiple rollouts per prompt and optimizes the model with correctness, format, and conciseness rewards.
- Distillation generates multiple reasoning traces, filters for correct answers, keeps the shortest correct trace per problem, and runs supervised fine-tuning on the curated set.
- Evaluation runs throughout the pipeline so you can compare accuracy and token usage over time.

## Requirements

- Python 3.10+
- A CUDA-capable GPU is strongly recommended
- Hugging Face model access for the base model you choose

Install dependencies with:

```bash
pip install -r requirements.txt
```

Optional:

- `wandb` logging is enabled by default
- `vLLM` can be used for faster distillation-time generation via `--use_vllm`

## Quick Start

Run the full pipeline with defaults:

```bash
python run.py
```

Run a smaller smoke test:

```bash
python run.py \
  --num_iterations 1 \
  --gsm8k_train_samples 200 \
  --math_train_samples 200 \
  --eval_samples 50
```

Use a different base model:

```bash
python run.py --base_model Qwen/Qwen2.5-3B-Instruct --lora_r 32
```

Enable vLLM for distillation generation:

```bash
python run.py --use_vllm
```

## Configuration

Most runtime settings live in `iterreason/config.py`. The CLI is auto-generated from the `IterReasonConfig` dataclass, so each config field can be overridden from the command line.

Key settings include:

- base model selection
- LoRA enablement and adapter hyperparameters
- number of GRPO/distillation iterations
- generation, optimization, and evaluation parameters
- output directory and logging behavior

## Project Structure

- `run.py`: CLI entry point
- `iterreason/pipeline.py`: overall GRPO/distillation loop
- `iterreason/grpo_phase.py`: GRPO training phase
- `iterreason/distill_phase.py`: self-distillation phase
- `iterreason/evaluate.py`: evaluation
- `iterreason/data.py`: dataset loading
- `iterreason/reward.py`: reward functions and answer matching

## Outputs

By default, artifacts are written under `outputs/`, including:

- per-iteration GRPO and distillation checkpoints
- merged model outputs
- evaluation metrics saved across the run

## Notes

- The default base model is `Qwen/Qwen2.5-7B-Instruct`.
- `use_wandb=True` by default, so set up Weights & Biases or disable it with `--use_wandb`.
- Training and generation settings assume a reasonably capable GPU environment.
