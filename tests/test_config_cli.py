import sys

import pytest

from iterreason.config import IterReasonConfig


def test_defaults_are_valid():
    IterReasonConfig()


@pytest.mark.parametrize("kwargs", [
    {"task_datasets": ["foo"]},
    {"task_datasets": []},
    {"grpo_per_device_batch_size": 3, "grpo_gradient_accumulation_steps": 1},
    {"grpo_num_generations": 1},
    {"eval_pass_at_k": [32]},
    {"distill_best_of_k": 0},
])
def test_invalid_configs_fail_fast(kwargs):
    with pytest.raises(ValueError):
        IterReasonConfig(**kwargs)


def _parse(monkeypatch, argv):
    import run
    captured = {}
    monkeypatch.setattr(sys, "argv", ["run.py", *argv])
    monkeypatch.setattr(run, "run_pipeline",
                        lambda cfg: captured.setdefault("cfg", cfg) and ("m", []))
    run.main()
    return captured["cfg"]


def test_cli_bool_flags(monkeypatch):
    cfg = _parse(monkeypatch, [])
    assert cfg.use_wandb is True and cfg.use_lora is True
    cfg = _parse(monkeypatch, ["--no-use_wandb", "--use_vllm"])
    assert cfg.use_wandb is False and cfg.use_vllm is True
    cfg = _parse(monkeypatch, ["--use_wandb"])
    assert cfg.use_wandb is True


def test_cli_lists_and_scalars(monkeypatch):
    cfg = _parse(monkeypatch, ["--task_datasets", "gsm8k",
                               "--eval_pass_at_k", "1", "2", "--lora_r", "8"])
    assert cfg.task_datasets == ["gsm8k"]
    assert cfg.eval_pass_at_k == [1, 2]
    assert cfg.lora_r == 8


def test_cli_list_defaults_are_independent(monkeypatch):
    a = _parse(monkeypatch, [])
    a.task_datasets.append("x")
    b = _parse(monkeypatch, [])
    assert b.task_datasets == ["gsm8k", "math"]
