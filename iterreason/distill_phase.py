"""
Phase B: Self-Distillation.

1.  Generate k CoT traces per training problem  (best-of-k)
2.  Filter: keep only correct completions
3.  For each problem pick the *shortest* correct trace  (compress)
4.  SFT on the curated dataset  →  compressed model
"""

import json
import os
from typing import Dict, List, Tuple

import torch
from datasets import Dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from .reward import answers_match, extract_answer
from .utils import log, merge_lora_and_save


# ═══════════════════════════════════════════════════════════════
#  Step 1 — Generation
# ═══════════════════════════════════════════════════════════════

def _generate_traces_hf(
    model_path: str,
    train_dataset: Dataset,
    config,
) -> List[Tuple[int, List[str]]]:
    """Generate k completions per problem using HuggingFace generate."""

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.eval()

    results: List[Tuple[int, List[str]]] = []
    bs = config.distill_generation_batch_size

    for start in tqdm(range(0, len(train_dataset), bs), desc="gen-traces"):
        end = min(start + bs, len(train_dataset))
        batch = train_dataset.select(range(start, end))

        # format with chat template
        formatted = [
            tokenizer.apply_chat_template(
                ex["prompt"], tokenize=False, add_generation_prompt=True,
            )
            for ex in batch
        ]

        tokenizer.padding_side = "left"
        inputs = tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.grpo_max_prompt_length,
        ).to(model.device)

        prompt_len = inputs.input_ids.shape[1]
        completions_per_example: List[List[str]] = [[] for _ in range(end - start)]

        for _ in range(config.distill_best_of_k):
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=config.grpo_max_completion_length,
                    do_sample=True,
                    temperature=config.distill_temperature,
                    top_p=0.95,
                    pad_token_id=tokenizer.pad_token_id,
                )
            for j in range(end - start):
                comp = tokenizer.decode(
                    out[j][prompt_len:], skip_special_tokens=True,
                )
                completions_per_example[j].append(comp)

        for j, idx in enumerate(range(start, end)):
            results.append((idx, completions_per_example[j]))

    del model
    torch.cuda.empty_cache()
    return results


def _generate_traces_vllm(
    model_path: str,
    train_dataset: Dataset,
    config,
) -> List[Tuple[int, List[str]]]:
    """Generate k completions per problem using vLLM (much faster)."""
    from vllm import LLM, SamplingParams  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # build formatted prompts
    formatted = [
        tokenizer.apply_chat_template(
            ex["prompt"], tokenize=False, add_generation_prompt=True,
        )
        for ex in train_dataset
    ]

    llm = LLM(
        model=model_path,
        gpu_memory_utilization=config.vllm_gpu_utilization,
        dtype="bfloat16" if config.bf16 else "float16",
        max_model_len=config.grpo_max_prompt_length
        + config.grpo_max_completion_length,
    )
    params = SamplingParams(
        temperature=config.distill_temperature,
        top_p=0.95,
        max_tokens=config.grpo_max_completion_length,
        n=config.distill_best_of_k,
    )

    outputs = llm.generate(formatted, params)

    results = []
    for idx, output in enumerate(outputs):
        completions = [o.text for o in output.outputs]
        results.append((idx, completions))

    del llm
    torch.cuda.empty_cache()
    return results


# ═══════════════════════════════════════════════════════════════
#  Step 2 — Filter correct completions
# ═══════════════════════════════════════════════════════════════

def _filter_correct(
    all_completions: List[Tuple[int, List[str]]],
    train_dataset: Dataset,
) -> Dataset:
    """Keep shortest correct completion per problem, build SFT dataset."""
    messages_list = []
    stats = {"total": 0, "with_correct": 0, "total_correct_traces": 0}

    for idx, completions in all_completions:
        stats["total"] += 1
        gt = train_dataset[idx]["answer"]
        prompt_msgs = train_dataset[idx]["prompt"]  # [system, user]

        correct = [
            c for c in completions
            if (pred := extract_answer(c)) is not None
            and answers_match(pred, gt)
        ]
        if not correct:
            continue

        stats["with_correct"] += 1
        stats["total_correct_traces"] += len(correct)

        # pick the *shortest* correct trace → encourages compressed reasoning
        best = min(correct, key=len)

        messages_list.append(
            prompt_msgs + [{"role": "assistant", "content": best}]
        )

    log(
        f"Filter stats: {stats['with_correct']}/{stats['total']} problems "
        f"have ≥1 correct trace  "
        f"({stats['total_correct_traces']} correct traces total)"
    )

    return Dataset.from_dict({"messages": messages_list})


# ═══════════════════════════════════════════════════════════════
#  Step 3 — SFT on the curated dataset
# ═══════════════════════════════════════════════════════════════

def _run_sft(
    model_path: str,
    output_dir: str,
    sft_dataset: Dataset,
    config,
) -> str:
    """Standard supervised fine-tuning on the filtered traces."""

    sft_args = SFTConfig(
        output_dir=output_dir,
        max_seq_length=config.distill_max_seq_length,
        learning_rate=config.distill_learning_rate,
        num_train_epochs=config.distill_num_train_epochs,
        per_device_train_batch_size=config.distill_per_device_batch_size,
        gradient_accumulation_steps=config.distill_gradient_accumulation_steps,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=2,
        report_to="wandb" if config.use_wandb else "none",
        run_name="distill-sft",
        seed=config.seed,
    )

    peft_config = None
    if config.use_lora:
        peft_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            task_type="CAUSAL_LM",
        )

    trainer = SFTTrainer(
        model=model_path,
        args=sft_args,
        train_dataset=sft_dataset,
        peft_config=peft_config,
    )

    trainer.train()

    merged_dir = os.path.join(output_dir, "merged")
    if config.use_lora:
        merge_lora_and_save(trainer.model, trainer.tokenizer, merged_dir)
    else:
        trainer.save_model(merged_dir)
        trainer.tokenizer.save_pretrained(merged_dir)

    del trainer
    torch.cuda.empty_cache()
    return merged_dir


# ═══════════════════════════════════════════════════════════════
#  Public entry point
# ═══════════════════════════════════════════════════════════════

def run_distillation(
    model_path: str,
    output_dir: str,
    train_dataset: Dataset,
    config,
) -> str:
    """
    Full distillation pipeline: generate → filter → SFT.

    Returns
    -------
    str – path to the merged distilled model.
    """
    os.makedirs(output_dir, exist_ok=True)
    log(f"Distill  model_in = {model_path}")
    log(f"Distill  output   = {output_dir}")

    # 1. Generate
    log("Step 1/3: Generating CoT traces …")
    if config.use_vllm:
        traces = _generate_traces_vllm(model_path, train_dataset, config)
    else:
        traces = _generate_traces_hf(model_path, train_dataset, config)

    # Persist raw traces so we never have to regenerate
    trace_path = os.path.join(output_dir, "raw_traces.json")
    with open(trace_path, "w") as f:
        json.dump(
            [{"idx": idx, "completions": comps} for idx, comps in traces],
            f,
        )
    log(f"Raw traces saved → {trace_path}")

    # 2. Filter
    log("Step 2/3: Filtering correct completions …")
    sft_dataset = _filter_correct(traces, train_dataset)
    log(f"SFT dataset size: {len(sft_dataset)}")

    if len(sft_dataset) == 0:
        log("WARNING: no correct traces found — skipping SFT, "
            "returning original model")
        return model_path

    # 3. SFT
    log("Step 3/3: SFT on filtered traces …")
    merged_dir = _run_sft(model_path, output_dir, sft_dataset, config)

    log(f"Distillation complete → {merged_dir}")
    return merged_dir
