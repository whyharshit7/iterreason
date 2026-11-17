"""
Phase B: Self-Distillation.

1.  Generate k CoT traces per training problem  (best-of-k)
2.  Filter: keep only correct completions
3.  For each problem pick the *shortest* correct trace  (compress)
4.  SFT on the curated dataset  →  compressed model
"""

import json
import os
from typing import List, Optional, Tuple

from datasets import Dataset
from peft import LoraConfig
from tqdm import tqdm
from trl import SFTConfig, SFTTrainer

from .generation import generate, load_model_for_generation
from .reward import answers_match, extract_answer
from .utils import (
    filter_config_kwargs,
    finish_wandb,
    free_memory,
    log,
    merge_lora_and_save,
)


# ═══════════════════════════════════════════════════════════════
#  Step 1 — Generation
# ═══════════════════════════════════════════════════════════════

def _generate_traces_hf(
    model_path: str,
    train_dataset: Dataset,
    config,
) -> List[Tuple[int, List[str]]]:
    """Generate k completions per problem using HuggingFace generate."""
    model, tokenizer = load_model_for_generation(model_path, config)

    results: List[Tuple[int, List[str]]] = []
    bs = config.distill_generation_batch_size
    try:
        for start in tqdm(range(0, len(train_dataset), bs), desc="gen-traces"):
            end = min(start + bs, len(train_dataset))
            prompts = train_dataset.select(range(start, end))["prompt"]
            outs = generate(
                model, tokenizer, prompts,
                max_prompt_length=config.grpo_max_prompt_length,
                max_new_tokens=config.grpo_max_completion_length,
                num_return_sequences=config.distill_best_of_k,
                do_sample=True,
                temperature=config.distill_temperature,
                top_p=0.95,
            )
            for idx, samples in zip(range(start, end), outs):
                results.append((idx, [text for text, _ in samples]))
    finally:
        del model
        free_memory()
    return results


def _generate_traces_vllm(
    model_path: str,
    train_dataset: Dataset,
    config,
) -> List[Tuple[int, List[str]]]:
    """Generate k completions per problem using vLLM (much faster)."""
    from vllm import LLM, SamplingParams  # type: ignore

    llm = LLM(
        model=model_path,
        gpu_memory_utilization=config.vllm_gpu_utilization,
        dtype="bfloat16" if config.bf16 else "float16",
        max_model_len=config.grpo_max_prompt_length
        + config.grpo_max_completion_length,
        seed=config.seed,
    )
    params = SamplingParams(
        temperature=config.distill_temperature,
        top_p=0.95,
        max_tokens=config.grpo_max_completion_length,
        n=config.distill_best_of_k,
        truncate_prompt_tokens=config.grpo_max_prompt_length,
    )

    try:
        # llm.chat applies the model's chat template itself
        outputs = llm.chat(list(train_dataset["prompt"]), params, use_tqdm=True)
        results = [
            (idx, [o.text for o in output.outputs])
            for idx, output in enumerate(outputs)
        ]
    finally:
        _shutdown_vllm(llm)
    return results


def _shutdown_vllm(llm):
    """`del llm` alone does not release vLLM's GPU memory, which makes the
    following SFT step OOM.  Tear it down as thoroughly as the installed
    version allows."""
    try:
        from vllm.distributed.parallel_state import (  # type: ignore
            destroy_distributed_environment,
            destroy_model_parallel,
        )
        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception:
        pass
    try:
        llm.llm_engine.engine_core.shutdown()  # vLLM V1
    except Exception:
        pass
    del llm
    free_memory()


# ═══════════════════════════════════════════════════════════════
#  Raw-trace cache (lets a crashed run resume without regenerating)
# ═══════════════════════════════════════════════════════════════

def _load_cached_traces(path: str, model_path: str, n_examples: int, k: int
                        ) -> Optional[List[Tuple[int, List[str]]]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            blob = json.load(f)
        meta = blob.get("meta", {}) if isinstance(blob, dict) else {}
        if (meta.get("model_path") != model_path
                or meta.get("n_examples") != n_examples
                or meta.get("k") != k):
            log("Cached traces don't match this run — regenerating")
            return None
        traces = [(t["idx"], t["completions"]) for t in blob["traces"]]
        log(f"Re-using cached raw traces from {path}")
        return traces
    except (OSError, ValueError, KeyError, TypeError) as e:
        log(f"Could not read cached traces ({e}) — regenerating", "warning")
        return None


def _save_traces(path: str, traces, model_path: str, n_examples: int, k: int):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(
            {
                "meta": {"model_path": model_path,
                         "n_examples": n_examples, "k": k},
                "traces": [{"idx": i, "completions": c} for i, c in traces],
            },
            f,
        )
    os.replace(tmp, path)  # atomic: never leave a half-written cache


# ═══════════════════════════════════════════════════════════════
#  Step 2 — Filter correct completions
# ═══════════════════════════════════════════════════════════════

def _filter_correct(
    all_completions: List[Tuple[int, List[str]]],
    train_dataset: Dataset,
) -> Dataset:
    """Keep shortest correct completion per problem, build SFT dataset
    in TRL's conversational prompt/completion format (so the loss is
    computed on the assistant turn only)."""
    prompts, completions = [], []
    stats = {"total": 0, "with_correct": 0, "total_correct_traces": 0}
    answers = train_dataset["answer"]
    prompt_col = train_dataset["prompt"]

    for idx, comps in all_completions:
        stats["total"] += 1
        gt = answers[idx]

        correct = [
            c for c in comps
            if (pred := extract_answer(c)) is not None
            and answers_match(pred, gt)
        ]
        if not correct:
            continue

        stats["with_correct"] += 1
        stats["total_correct_traces"] += len(correct)

        # pick the *shortest* correct trace → encourages compressed reasoning
        best = min(correct, key=len)

        prompts.append(prompt_col[idx])  # [system, user]
        completions.append([{"role": "assistant", "content": best}])

    log(
        f"Filter stats: {stats['with_correct']}/{stats['total']} problems "
        f"have ≥1 correct trace  "
        f"({stats['total_correct_traces']} correct traces total)"
    )

    return Dataset.from_dict({"prompt": prompts, "completion": completions})


# ═══════════════════════════════════════════════════════════════
#  Step 3 — SFT on the curated dataset
# ═══════════════════════════════════════════════════════════════

def _run_sft(
    model_path: str,
    output_dir: str,
    sft_dataset: Dataset,
    config,
    run_name: str = "distill-sft",
) -> str:
    """Standard supervised fine-tuning on the filtered traces."""

    sft_kwargs = dict(
        output_dir=output_dir,
        max_length=config.distill_max_seq_length,
        completion_only_loss=True,
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
        run_name=run_name,
        seed=config.seed,
    )
    sft_args = SFTConfig(**filter_config_kwargs(SFTConfig, sft_kwargs))

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

    merged_dir = os.path.join(output_dir, "merged")
    try:
        trainer.train()
        if config.use_lora:
            merge_lora_and_save(trainer.model, trainer.processing_class,
                                merged_dir)
        else:
            trainer.save_model(merged_dir)
            trainer.processing_class.save_pretrained(merged_dir)
    finally:
        del trainer
        free_memory()
        finish_wandb()
    return merged_dir


# ═══════════════════════════════════════════════════════════════
#  Public entry point
# ═══════════════════════════════════════════════════════════════

def run_distillation(
    model_path: str,
    output_dir: str,
    train_dataset: Dataset,
    config,
    run_name: str = "distill-sft",
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

    n, k = len(train_dataset), config.distill_best_of_k
    trace_path = os.path.join(output_dir, "raw_traces.json")

    # 1. Generate (or reuse traces from an interrupted run)
    log("Step 1/3: Generating CoT traces …")
    traces = _load_cached_traces(trace_path, model_path, n, k)
    if traces is None:
        if config.use_vllm:
            traces = _generate_traces_vllm(model_path, train_dataset, config)
        else:
            traces = _generate_traces_hf(model_path, train_dataset, config)
        _save_traces(trace_path, traces, model_path, n, k)
        log(f"Raw traces saved → {trace_path}")

    # 2. Filter
    log("Step 2/3: Filtering correct completions …")
    sft_dataset = _filter_correct(traces, train_dataset)
    log(f"SFT dataset size: {len(sft_dataset)}")

    if len(sft_dataset) == 0:
        log("WARNING: no correct traces found — skipping SFT, "
            "returning original model", "warning")
        return model_path

    # 3. SFT
    log("Step 3/3: SFT on filtered traces …")
    merged_dir = _run_sft(model_path, output_dir, sft_dataset, config, run_name)

    log(f"Distillation complete → {merged_dir}")
    return merged_dir
