"""
Shared HuggingFace generation helpers (batched, left-padded).
"""

from typing import List, Sequence, Set, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils import pick_attn_implementation


def load_model_for_generation(model_path: str, config):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # decoder-only generation needs left padding; truncate the *start*
    # of over-long prompts so the generation prompt is never cut off
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16 if config.bf16 else torch.float32,
        device_map="auto",
        attn_implementation=pick_attn_implementation(),
    )
    model.eval()
    return model, tokenizer


def _stop_ids(model, tokenizer) -> Set[int]:
    ids: Set[int] = set()
    eos = getattr(model.generation_config, "eos_token_id", None)
    if isinstance(eos, int):
        ids.add(eos)
    elif eos is not None:
        ids.update(eos)
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        ids.add(tokenizer.pad_token_id)
    return ids


def _trim(ids: List[int], stop: Set[int]) -> List[int]:
    """Cut at the first stop token (inclusive) so post-EOS padding isn't
    counted towards completion length."""
    for i, t in enumerate(ids):
        if t in stop:
            return ids[: i + 1]
    return ids


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompts: Sequence[List[dict]],
    *,
    max_prompt_length: int,
    max_new_tokens: int,
    num_return_sequences: int = 1,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> List[List[Tuple[str, int]]]:
    """Generate for a batch of chat prompts.

    Returns, per prompt, a list of ``num_return_sequences`` tuples
    ``(completion_text, completion_num_tokens)``.
    """
    texts = [
        tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_length,
        add_special_tokens=False,  # chat template already has them
    ).to(model.device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=do_sample,
    )
    if do_sample:
        gen_kwargs.update(temperature=temperature, top_p=top_p)
    else:
        # silence "temperature is set but do_sample=False" warnings coming
        # from the model's own generation_config
        gen_kwargs.update(temperature=None, top_p=None, top_k=None)

    out = model.generate(**inputs, **gen_kwargs)
    prompt_len = inputs["input_ids"].shape[1]
    stop = _stop_ids(model, tokenizer)

    results: List[List[Tuple[str, int]]] = [[] for _ in prompts]
    for row, seq in enumerate(out):
        ids = _trim(seq[prompt_len:].tolist(), stop)
        text = tokenizer.decode(ids, skip_special_tokens=True)
        results[row // num_return_sequences].append((text, len(ids)))
    return results
