#!/usr/bin/env python3
"""ASHA-style tuning for HelpSteer MODPO (MODPO-only, fixed w)."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import optuna
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import TPESampler
from optuna.study import MaxTrialsCallback
from optuna.trial import Trial, TrialState

try:
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend
except Exception as exc:  # pragma: no cover - import guard for old Optuna versions
    JournalStorage = None
    JournalFileBackend = None
    JOURNAL_IMPORT_ERROR = exc
else:
    JOURNAL_IMPORT_ERROR = None


CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")
OBJECTIVE_KEY = "eval_rewards/margins"


@dataclass(frozen=True)
class StageMetrics:
    objective: float
    objective_step: int
    eval_loss: Optional[float]
    eval_loss_step: Optional[int]
    source_state_path: str


def _parse_float_list(raw: str) -> list[float]:
    out = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    if not out:
        raise ValueError(f"Expected non-empty comma-separated float list, got: {raw!r}")
    return out


def _parse_int_list(raw: str) -> list[int]:
    out = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise ValueError(f"Expected non-empty comma-separated int list, got: {raw!r}")
    dedup_sorted = sorted(set(out))
    if dedup_sorted != out:
        raise ValueError(f"--rung_steps must be strictly increasing without duplicates, got: {raw!r}")
    return out


def _fmt_float(x: float) -> str:
    return f"{x:.12g}"


def _stable_worker_seed(base_seed: int, worker_id: str) -> int:
    acc = 17
    for ch in worker_id:
        acc = (acc * 31 + ord(ch)) % 1_000_000
    return base_seed + acc


def _list_checkpoints(output_dir: Path) -> list[tuple[int, Path]]:
    checkpoints: list[tuple[int, Path]] = []
    if not output_dir.exists():
        return checkpoints
    for child in output_dir.iterdir():
        if not child.is_dir():
            continue
        match = CHECKPOINT_RE.match(child.name)
        if not match:
            continue
        checkpoints.append((int(match.group(1)), child))
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def _last_checkpoint(output_dir: Path) -> tuple[Optional[Path], int]:
    checkpoints = _list_checkpoints(output_dir)
    if not checkpoints:
        return None, 0
    step, path = checkpoints[-1]
    return path, step


def _extract_metrics_from_state(
    state_path: Path,
    budget_step: int,
    objective_key: str = OBJECTIVE_KEY,
) -> Optional[StageMetrics]:
    with state_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    log_history = data.get("log_history", [])
    if not isinstance(log_history, list):
        return None

    best_obj_step = -1
    best_obj_val = None
    best_loss_step = -1
    best_loss_val = None

    for entry in log_history:
        if not isinstance(entry, dict):
            continue
        step_raw = entry.get("step")
        if step_raw is None:
            continue
        try:
            step = int(step_raw)
        except (TypeError, ValueError):
            continue
        if step > budget_step:
            continue

        if objective_key in entry:
            try:
                obj_val = float(entry[objective_key])
            except (TypeError, ValueError):
                obj_val = None
            if obj_val is not None and step >= best_obj_step:
                best_obj_step = step
                best_obj_val = obj_val

        if "eval_loss" in entry:
            try:
                loss_val = float(entry["eval_loss"])
            except (TypeError, ValueError):
                loss_val = None
            if loss_val is not None and step >= best_loss_step:
                best_loss_step = step
                best_loss_val = loss_val

    if best_obj_val is None:
        return None
    return StageMetrics(
        objective=float(best_obj_val),
        objective_step=int(best_obj_step),
        eval_loss=(float(best_loss_val) if best_loss_val is not None else None),
        eval_loss_step=(int(best_loss_step) if best_loss_step >= 0 else None),
        source_state_path=str(state_path),
    )


def _read_stage_metrics(output_dir: Path, budget_step: int, objective_key: str = OBJECTIVE_KEY) -> StageMetrics:
    checkpoints = _list_checkpoints(output_dir)
    checkpoints = [x for x in checkpoints if x[0] <= budget_step]
    checkpoints.reverse()

    for step, ckpt in checkpoints:
        state_path = ckpt / "trainer_state.json"
        if not state_path.exists():
            continue
        metrics = _extract_metrics_from_state(state_path, budget_step=budget_step, objective_key=objective_key)
        if metrics is not None:
            return metrics

    # Fallback if no checkpoint state is available (rare but possible after interrupted runs).
    root_state = output_dir / "trainer_state.json"
    if root_state.exists():
        metrics = _extract_metrics_from_state(root_state, budget_step=budget_step, objective_key=objective_key)
        if metrics is not None:
            return metrics

    raise RuntimeError(
        f"Could not find {objective_key!r} at or before step={budget_step} in {output_dir}. "
        "Check eval/save cadence and trial logs."
    )


def _build_train_cmd(
    args: argparse.Namespace,
    trial_number: int,
    run_name: str,
    output_dir: Path,
    budget_step: int,
    warmup_steps: int,
    learning_rate: float,
    weight_decay: float,
    beta: float,
    margin_beta: float,
    resume_checkpoint: Optional[Path],
) -> list[str]:
    cmd = [
        "accelerate",
        "launch",
        "scripts/modpo/ultrafeedback/modpo.py",
        "--sft_model_name",
        args.sft_model_name,
        "--margin_reward_model_name",
        args.margin_reward_model_name,
        "--dataset_name",
        args.dataset_name,
        "--w",
        _fmt_float(args.w_fixed),
        "--beta",
        _fmt_float(beta),
        "--margin_beta",
        _fmt_float(margin_beta),
        "--generate-during-eval",
        "False",
        "--max_length",
        str(args.max_length),
        "--precision",
        args.precision,
        "--training_args.output_dir",
        str(output_dir),
        "--training_args.run_name",
        run_name,
        "--training_args.max_steps",
        str(budget_step),
        "--training_args.per_device_train_batch_size",
        str(args.train_batch_size),
        "--training_args.per_device_eval_batch_size",
        str(args.eval_batch_size),
        "--training_args.gradient_accumulation_steps",
        str(args.grad_accum),
        "--training_args.learning_rate",
        _fmt_float(learning_rate),
        "--training_args.weight_decay",
        _fmt_float(weight_decay),
        "--training_args.warmup_steps",
        str(warmup_steps),
        "--training_args.logging_steps",
        str(args.logging_steps),
        "--training_args.save_strategy",
        "steps",
        "--training_args.save_steps",
        str(args.save_steps),
        "--training_args.save_total_limit",
        str(args.save_total_limit),
        "--training_args.evaluation_strategy",
        "steps",
        "--training_args.eval_steps",
        str(args.eval_steps),
        "--training_args.load_best_model_at_end",
        "True",
        "--training_args.report_to",
        args.report_to,
        "--training_args.seed",
        str(args.train_seed),
    ]
    if args.gradient_checkpointing:
        cmd.append("--training_args.gradient_checkpointing")
    if resume_checkpoint is not None:
        cmd += ["--resume-from-checkpoint", str(resume_checkpoint)]
    return cmd


def _run_training_stage(
    args: argparse.Namespace,
    trial_number: int,
    trial_dir: Path,
    budget_step: int,
    warmup_steps: int,
    learning_rate: float,
    weight_decay: float,
    beta: float,
    margin_beta: float,
) -> None:
    trial_dir.mkdir(parents=True, exist_ok=True)
    log_dir = trial_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stage_log = log_dir / f"stage_{budget_step}.log"

    resume_checkpoint, current_step = _last_checkpoint(trial_dir)
    if current_step >= budget_step:
        print(
            f"[trial={trial_number}] skip stage={budget_step} "
            f"(already has checkpoint-{current_step})"
        )
        return

    run_name = f"{args.run_tag}_trial{trial_number:04d}_{args.worker_id}"
    cmd = _build_train_cmd(
        args=args,
        trial_number=trial_number,
        run_name=run_name,
        output_dir=trial_dir,
        budget_step=budget_step,
        warmup_steps=warmup_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        beta=beta,
        margin_beta=margin_beta,
        resume_checkpoint=resume_checkpoint,
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = args.pythonpath
    if args.gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    with stage_log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} stage={budget_step} ===\n")
        handle.write("CMD: " + " ".join(cmd) + "\n")
        handle.flush()
        proc = subprocess.run(
            cmd,
            cwd=args.repo_root,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    if proc.returncode != 0:
        raise RuntimeError(
            f"Training failed for trial={trial_number} stage={budget_step}. "
            f"See log: {stage_log}"
        )


def _trial_output_dir(args: argparse.Namespace, trial_number: int) -> Path:
    return Path(args.output_root) / args.study_name / f"trial_{trial_number:04d}"


def _build_objective(args: argparse.Namespace):
    rung_steps = args.rung_steps
    final_budget = rung_steps[-1]

    def objective(trial: Trial) -> float:
        learning_rate = trial.suggest_float("learning_rate", args.lr_min, args.lr_max, log=True)
        weight_decay = trial.suggest_float("weight_decay", args.weight_decay_min, args.weight_decay_max)
        warmup_ratio = trial.suggest_float("warmup_ratio", args.warmup_ratio_min, args.warmup_ratio_max)
        beta = trial.suggest_categorical("beta", args.beta_values)
        margin_mult = trial.suggest_float("margin_mult", args.margin_mult_min, args.margin_mult_max, log=True)
        margin_beta = beta * margin_mult

        warmup_steps = max(0, int(round(final_budget * warmup_ratio)))
        trial.set_user_attr("worker_id", args.worker_id)
        trial.set_user_attr("w_fixed", args.w_fixed)
        trial.set_user_attr("margin_beta", margin_beta)
        trial.set_user_attr("warmup_steps", warmup_steps)
        trial.set_user_attr("objective_key", OBJECTIVE_KEY)

        trial_dir = _trial_output_dir(args, trial.number)
        last_metrics = None

        for step in rung_steps:
            _run_training_stage(
                args=args,
                trial_number=trial.number,
                trial_dir=trial_dir,
                budget_step=step,
                warmup_steps=warmup_steps,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                beta=beta,
                margin_beta=margin_beta,
            )
            metrics = _read_stage_metrics(trial_dir, budget_step=step, objective_key=OBJECTIVE_KEY)
            last_metrics = metrics

            trial.report(metrics.objective, step=step)
            trial.set_user_attr(f"objective_step_{step}", metrics.objective)
            trial.set_user_attr(f"objective_source_step_{step}", metrics.objective_step)
            if metrics.eval_loss is not None:
                trial.set_user_attr(f"eval_loss_step_{step}", metrics.eval_loss)
                trial.set_user_attr(f"eval_loss_source_step_{step}", metrics.eval_loss_step)

            print(
                f"[trial={trial.number}] step={step} objective={metrics.objective:.6f} "
                f"(src_step={metrics.objective_step}) eval_loss="
                f"{'n/a' if metrics.eval_loss is None else f'{metrics.eval_loss:.6f}'}"
            )

            if step < final_budget and trial.should_prune():
                raise optuna.TrialPruned(
                    f"pruned at step={step} objective={metrics.objective:.6f}"
                )

        if last_metrics is None:
            raise RuntimeError(f"Trial {trial.number} ended without metrics.")
        trial.set_user_attr("final_eval_loss", last_metrics.eval_loss)
        trial.set_user_attr("final_objective", last_metrics.objective)
        return last_metrics.objective

    return objective


def _create_storage(storage_path: Path):
    if JournalStorage is None or JournalFileBackend is None:
        raise RuntimeError(
            "Optuna JournalStorage is unavailable in this environment. "
            f"Import error: {JOURNAL_IMPORT_ERROR}"
        )
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    return JournalStorage(JournalFileBackend(str(storage_path)))


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(tmp_path, path)


def _write_study_summary(study: optuna.Study, study_dir: Path) -> None:
    study_dir.mkdir(parents=True, exist_ok=True)
    trials = sorted(study.trials, key=lambda t: t.number)
    param_keys = sorted({k for t in trials for k in t.params.keys()})
    user_attr_keys = sorted({k for t in trials for k in t.user_attrs.keys()})

    fieldnames = [
        "number",
        "state",
        "value",
        "datetime_start",
        "datetime_complete",
        "duration_seconds",
    ] + [f"param.{k}" for k in param_keys] + [f"user_attr.{k}" for k in user_attr_keys]

    csv_path = study_dir / "study_trials.csv"
    tmp_csv = csv_path.with_suffix(f".csv.tmp.{os.getpid()}.{time.time_ns()}")
    with tmp_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for t in trials:
            duration = None
            if t.datetime_start and t.datetime_complete:
                duration = (t.datetime_complete - t.datetime_start).total_seconds()
            row = {
                "number": t.number,
                "state": t.state.name,
                "value": t.value,
                "datetime_start": t.datetime_start.isoformat() if t.datetime_start else None,
                "datetime_complete": t.datetime_complete.isoformat() if t.datetime_complete else None,
                "duration_seconds": duration,
            }
            for key in param_keys:
                row[f"param.{key}"] = t.params.get(key)
            for key in user_attr_keys:
                row[f"user_attr.{key}"] = t.user_attrs.get(key)
            writer.writerow(row)
    os.replace(tmp_csv, csv_path)

    best_payload = {
        "study_name": study.study_name,
        "n_trials": len(trials),
        "best_trial": None,
    }
    complete_trials = [t for t in trials if t.state == TrialState.COMPLETE and t.value is not None]
    if complete_trials:
        best_trial = study.best_trial
        best_payload["best_trial"] = {
            "number": best_trial.number,
            "value": best_trial.value,
            "params": best_trial.params,
            "user_attrs": best_trial.user_attrs,
        }
    _write_text_atomic(study_dir / "best_trial.json", json.dumps(best_payload, indent=2, sort_keys=True))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run ASHA-style Optuna tuning for MODPO-only training on HelpSteer. "
            "Designed for one worker process per GPU."
        )
    )
    parser.add_argument("--repo_root", type=str, default=".", help="Path to modpo repo root.")
    parser.add_argument("--output_root", type=str, default="./outputs/helpsteer/asha")
    parser.add_argument("--study_name", type=str, default="helpsteer_modpo_w07_asha")
    parser.add_argument("--storage_path", type=str, default="", help="Optuna Journal file path.")
    parser.add_argument("--summarize_only", action="store_true", help="Write study summary and exit.")

    parser.add_argument("--worker_id", type=str, default="worker0")
    parser.add_argument("--gpu_id", type=str, default="", help="Set CUDA_VISIBLE_DEVICES for this worker.")
    parser.add_argument("--pythonpath", type=str, default=".")

    parser.add_argument("--n_trials", type=int, default=24, help="Total COMPLETE+PRUNED+FAIL trials.")
    parser.add_argument("--timeout_sec", type=int, default=0, help="0 means no timeout.")
    parser.add_argument("--rung_steps", type=str, default="300,600,1000")
    parser.add_argument("--reduction_factor", type=int, default=2)
    parser.add_argument("--tpe_startup_trials", type=int, default=4)
    parser.add_argument("--sampler_seed", type=int, default=42)

    parser.add_argument("--sft_model_name", type=str, required=True)
    parser.add_argument("--margin_reward_model_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="nvidia/HelpSteer-pairwise-helpfulness")
    parser.add_argument("--w_fixed", type=float, default=0.7)
    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=300)
    parser.add_argument("--eval_steps", type=int, default=300)
    parser.add_argument("--save_total_limit", type=int, default=5)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--run_tag", type=str, default="helpsteer_modpo_asha")
    parser.add_argument("--train_seed", type=int, default=42)

    parser.add_argument("--lr_min", type=float, default=5e-5)
    parser.add_argument("--lr_max", type=float, default=8e-4)
    parser.add_argument("--weight_decay_min", type=float, default=0.0)
    parser.add_argument("--weight_decay_max", type=float, default=0.1)
    parser.add_argument("--warmup_ratio_min", type=float, default=0.02)
    parser.add_argument("--warmup_ratio_max", type=float, default=0.12)
    parser.add_argument("--beta_values", type=str, default="0.1,0.2,0.5")
    parser.add_argument("--margin_mult_min", type=float, default=0.5)
    parser.add_argument("--margin_mult_max", type=float, default=2.0)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    args.repo_root = str(Path(args.repo_root).resolve())
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = Path(args.repo_root) / output_root
    args.output_root = str(output_root.resolve())

    args.rung_steps = _parse_int_list(args.rung_steps)
    args.beta_values = _parse_float_list(args.beta_values)

    if args.w_fixed <= 0.0 or args.w_fixed > 1.0:
        raise ValueError(f"--w_fixed must be in (0,1], got {args.w_fixed}")
    if args.reduction_factor < 2:
        raise ValueError("--reduction_factor must be >= 2")
    if args.n_trials <= 0:
        raise ValueError("--n_trials must be > 0")
    if args.lr_min <= 0 or args.lr_max <= 0 or args.lr_min >= args.lr_max:
        raise ValueError("Invalid learning-rate bounds.")
    if args.weight_decay_min < 0 or args.weight_decay_min >= args.weight_decay_max:
        raise ValueError("Invalid weight_decay bounds.")
    if args.warmup_ratio_min < 0 or args.warmup_ratio_min >= args.warmup_ratio_max:
        raise ValueError("Invalid warmup_ratio bounds.")
    if args.margin_mult_min <= 0 or args.margin_mult_min >= args.margin_mult_max:
        raise ValueError("Invalid margin multiplier bounds.")
    if min(args.beta_values) <= 0:
        raise ValueError("All beta values must be > 0.")
    if args.eval_steps > args.rung_steps[0]:
        raise ValueError(
            f"--eval_steps ({args.eval_steps}) must be <= first rung step ({args.rung_steps[0]})."
        )
    if args.save_steps > args.rung_steps[0]:
        raise ValueError(
            f"--save_steps ({args.save_steps}) must be <= first rung step ({args.rung_steps[0]})."
        )

    if args.storage_path:
        storage_path = Path(args.storage_path)
        if not storage_path.is_absolute():
            storage_path = Path(args.repo_root) / storage_path
        args.storage_path = str(storage_path.resolve())
    else:
        args.storage_path = str(Path(args.output_root) / args.study_name / "optuna_study.journal")

    gpu_raw = args.gpu_id.strip()
    args.gpu_id = gpu_raw if gpu_raw else None
    if args.timeout_sec <= 0:
        args.timeout_sec = None


def _print_preflight(args: argparse.Namespace) -> None:
    print("=== ASHA MODPO preflight ===")
    print(f"repo_root={args.repo_root}")
    print(f"study_name={args.study_name}")
    print(f"output_root={args.output_root}")
    print(f"storage_path={args.storage_path}")
    print(f"worker_id={args.worker_id}")
    print(f"gpu_id={args.gpu_id}")
    print(f"sft_model_name={args.sft_model_name}")
    print(f"margin_reward_model_name={args.margin_reward_model_name}")
    print(f"dataset_name={args.dataset_name}")
    print(f"w_fixed={args.w_fixed}")
    print(f"rung_steps={args.rung_steps}")
    print(f"n_trials={args.n_trials}")
    print(f"timeout_sec={args.timeout_sec}")
    print(f"objective_key={OBJECTIVE_KEY}")
    print(f"search.lr=[{args.lr_min},{args.lr_max}]")
    print(f"search.weight_decay=[{args.weight_decay_min},{args.weight_decay_max}]")
    print(f"search.warmup_ratio=[{args.warmup_ratio_min},{args.warmup_ratio_max}]")
    print(f"search.beta_values={args.beta_values}")
    print(f"search.margin_mult=[{args.margin_mult_min},{args.margin_mult_max}]")
    print("============================")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _validate_args(args)
    _print_preflight(args)

    storage = _create_storage(Path(args.storage_path))
    worker_seed = _stable_worker_seed(args.sampler_seed, args.worker_id)
    sampler = TPESampler(
        seed=worker_seed,
        n_startup_trials=args.tpe_startup_trials,
        multivariate=True,
    )
    pruner = SuccessiveHalvingPruner(
        min_resource=args.rung_steps[0],
        reduction_factor=args.reduction_factor,
        min_early_stopping_rate=0,
    )

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    study_dir = Path(args.output_root) / args.study_name
    study_dir.mkdir(parents=True, exist_ok=True)

    if args.summarize_only:
        _write_study_summary(study, study_dir)
        print(f"Summary written to {study_dir}")
        return

    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ["PYTHONPATH"] = args.pythonpath

    max_trials_callback = MaxTrialsCallback(
        args.n_trials,
        states=(TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL),
    )

    objective = _build_objective(args)
    study.optimize(
        objective,
        n_trials=None,
        timeout=args.timeout_sec,
        gc_after_trial=True,
        callbacks=[max_trials_callback],
        catch=(RuntimeError,),
    )
    _write_study_summary(study, study_dir)
    print(f"Optimization finished. Summary written to {study_dir}")


if __name__ == "__main__":
    main()
