"""
Orchestrator: alternate GRPO ↔ Distillation for N iterations.

M0 ─[GRPO]→ M1 ─[Distill]→ M2 ─[GRPO]→ M3 ─[Distill]→ M4 → …
"""

import os

from .config import IterReasonConfig
from .data import load_eval_data, load_training_data
from .distill_phase import run_distillation
from .evaluate import evaluate_model
from .grpo_phase import run_grpo
from .utils import (
    log,
    log_metrics_table,
    save_all_metrics,
    save_metrics,
    set_seed,
    setup_logging,
)


def run_pipeline(config: IterReasonConfig):
    """Main entry point — runs the full iterative self-distillation loop."""

    setup_logging(config)
    set_seed(config.seed)
    log(f"Config:\n{config}")

    # ── Data ───────────────────────────────────────────────────
    train_dataset = load_training_data(config)
    eval_datasets = load_eval_data(config)

    # ── Baseline evaluation ────────────────────────────────────
    log("\n" + "=" * 60)
    log("Evaluating base model (M0) …")
    log("=" * 60)
    base_metrics = evaluate_model(
        config.base_model, eval_datasets, config, full=False,
    )
    records = [{"iteration": -1, "phase": "base", **base_metrics}]
    save_metrics(base_metrics, config.output_dir, "base")

    current_model = config.base_model

    # ── Iteration loop ─────────────────────────────────────────
    for it in range(config.num_iterations):
        log(f"\n{'═' * 60}")
        log(f"  ITERATION {it}")
        log(f"{'═' * 60}")

        # ── Phase A: GRPO ──────────────────────────────────────
        grpo_dir = os.path.join(config.output_dir, f"iter_{it}", "grpo")
        current_model = run_grpo(
            current_model, grpo_dir, train_dataset, config,
        )
        grpo_metrics = evaluate_model(
            current_model, eval_datasets, config, full=False,
        )
        records.append({"iteration": it, "phase": "grpo", **grpo_metrics})
        save_metrics(grpo_metrics, config.output_dir, f"iter{it}_grpo")

        # ── Phase B: Self-Distillation ─────────────────────────
        distill_dir = os.path.join(config.output_dir, f"iter_{it}", "distill")
        current_model = run_distillation(
            current_model, distill_dir, train_dataset, config,
        )
        distill_metrics = evaluate_model(
            current_model, eval_datasets, config, full=False,
        )
        records.append(
            {"iteration": it, "phase": "distill", **distill_metrics}
        )
        save_metrics(
            distill_metrics, config.output_dir, f"iter{it}_distill",
        )

        # ── Iteration summary ─────────────────────────────────
        _log_iteration_summary(it, grpo_metrics, distill_metrics)

    # ── Final full evaluation (with pass@k) ────────────────────
    log(f"\n{'═' * 60}")
    log("  FINAL EVALUATION  (with pass@k)")
    log(f"{'═' * 60}")
    final_metrics = evaluate_model(
        current_model, eval_datasets, config, full=True,
    )
    records.append(
        {"iteration": config.num_iterations, "phase": "final",
         **final_metrics}
    )

    save_all_metrics(records, config.output_dir)

    _log_progression(records)
    log(f"\nPipeline complete.  Final model → {current_model}")
    return current_model, records


# ── Pretty helpers ─────────────────────────────────────────────

def _log_iteration_summary(it, grpo_m, distill_m):
    log(f"\n┌── Iteration {it} summary")
    for ds in set(
        k.split("/")[0] for k in grpo_m if "/" in k
    ):
        ga = grpo_m.get(f"{ds}/accuracy", 0)
        da = distill_m.get(f"{ds}/accuracy", 0)
        gt = grpo_m.get(f"{ds}/avg_tokens", 0)
        dt = distill_m.get(f"{ds}/avg_tokens", 0)
        log(f"│  {ds:8s}  acc  GRPO {ga:.3f} → Distill {da:.3f}  "
            f"│  tokens  {gt:.0f} → {dt:.0f}")
    log("└" + "─" * 60)


def _log_progression(records):
    log("\n" + "=" * 60)
    log("  FULL PROGRESSION")
    log("=" * 60)
    header = f"{'Phase':>20s}"
    # find all accuracy keys
    acc_keys = sorted(
        {k for r in records for k in r if k.endswith("/accuracy")}
    )
    tok_keys = sorted(
        {k for r in records for k in r if k.endswith("/avg_tokens")}
    )
    for k in acc_keys:
        header += f"  {k:>14s}"
    for k in tok_keys:
        header += f"  {k:>14s}"
    log(header)
    log("─" * len(header))

    for r in records:
        it = r.get("iteration", "?")
        ph = r.get("phase", "?")
        label = f"it{it}/{ph}"
        row = f"{label:>20s}"
        for k in acc_keys:
            row += f"  {r.get(k, 0):>14.4f}"
        for k in tok_keys:
            row += f"  {r.get(k, 0):>14.0f}"
        log(row)
