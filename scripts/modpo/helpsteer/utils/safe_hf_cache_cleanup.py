#!/usr/bin/env python3
"""Safely clean Hugging Face model caches without disrupting active runs.

Default behavior is DRY RUN. Use --execute to actually delete.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import subprocess
from typing import Iterable


ENV_KEYS = {
    "HF_HOME",
    "HF_HUB_CACHE",
    "HF_XET_CACHE",
    "HF_ASSETS_CACHE",
    "HF_DATASETS_CACHE",
    "TRANSFORMERS_CACHE",
    "TMPDIR",
    "TMP",
    "TEMP",
}

# Keep this wide enough to catch training/eval entrypoints from our repo.
PROC_PATTERN = re.compile(
    r"(python|accelerate|modpo|run_pipeline|sft\.py|dpo\.py|score_armorm|hh|hh_rlhf|ultrafeedback)",
    re.IGNORECASE,
)
MODEL_ID_PATTERN = re.compile(r"\b([A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*)\b")


def canon(path: str) -> str:
    return os.path.realpath(os.path.expanduser(path))


def is_under(path: str, parent: str) -> bool:
    path = canon(path)
    parent = canon(parent)
    return path == parent or path.startswith(parent.rstrip("/") + "/")


def du_bytes(path: str) -> int:
    try:
        out = subprocess.check_output(["du", "-sb", path], text=True).split()[0]
        return int(out)
    except Exception:
        return 0


def human(n_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(n_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{n_bytes}B"


def default_cache_roots(user: str) -> list[str]:
    return [
        f"/mount/studenten-temp1/users/{user}/.hfcache",
        f"/mount/studenten-temp1/users/{user}/.cache/huggingface",
        f"/mount/arbeitsdaten33/projekte/tcl/tclext/danis/tsoi/thesis-exp/hf_cache/{user}",
        f"/mount/arbeitsdaten/tcl/tclext/danis/tsoi/thesis-exp/hf_cache/{user}",
        "/mount/arbeitsdaten33/projekte/tcl/tclext/danis/tsoi/thesis-exp/modpo/.hf_cache",
        "/mount/arbeitsdaten33/projekte/tcl/tclext/danis/tsoi/thesis-exp/modpo/hf_cache",
        "/mount/arbeitsdaten/tcl/tclext/danis/tsoi/thesis-exp/modpo/.hf_cache",
        "/mount/arbeitsdaten/tcl/tclext/danis/tsoi/thesis-exp/modpo/hf_cache",
    ]


def current_user_active_pids(uid: int) -> list[int]:
    pids: list[int] = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        pdir = f"/proc/{pid}"
        try:
            if os.stat(pdir).st_uid != uid:
                continue
            cmdline = (
                open(f"{pdir}/cmdline", "rb")
                .read()
                .replace(b"\x00", b" ")
                .decode("utf-8", "ignore")
                .strip()
            )
        except Exception:
            continue
        if not cmdline:
            continue
        if PROC_PATTERN.search(cmdline):
            pids.append(int(pid))
    return pids


def read_proc_env(pid: int) -> dict[str, str]:
    env: dict[str, str] = {}
    try:
        raw = open(f"/proc/{pid}/environ", "rb").read().split(b"\x00")
        for kv in raw:
            if b"=" not in kv:
                continue
            k, v = kv.split(b"=", 1)
            env[k.decode("utf-8", "ignore")] = v.decode("utf-8", "ignore")
    except Exception:
        pass
    return env


def read_proc_cmd(pid: int) -> str:
    try:
        return (
            open(f"/proc/{pid}/cmdline", "rb")
            .read()
            .replace(b"\x00", b" ")
            .decode("utf-8", "ignore")
            .strip()
        )
    except Exception:
        return ""


def iter_open_paths(active_pids: Iterable[int]) -> set[str]:
    out: set[str] = set()
    for pid in active_pids:
        fd_dir = f"/proc/{pid}/fd"
        if not os.path.isdir(fd_dir):
            continue
        for fd in os.listdir(fd_dir):
            fdp = os.path.join(fd_dir, fd)
            try:
                target = os.readlink(fdp)
            except Exception:
                continue
            if not target.startswith("/"):
                continue
            out.add(canon(target))
    return out


def gather_candidates(cache_roots: Iterable[str]) -> set[str]:
    candidates: set[str] = set()
    for root in cache_roots:
        for path in glob.glob(os.path.join(root, "**", "models--*"), recursive=True):
            if os.path.isdir(path):
                candidates.add(canon(path))
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Safely clean Hugging Face model caches while protecting active runs."
    )
    parser.add_argument("--execute", action="store_true", help="Actually delete (default is dry run).")
    parser.add_argument("--user", default=os.environ.get("USER", "tsoidd"))
    parser.add_argument(
        "--cache-root",
        action="append",
        default=[],
        help="Additional cache root to scan. Repeatable.",
    )
    parser.add_argument("--max-protected-print", type=int, default=30)
    args = parser.parse_args()

    uid = os.getuid()
    cache_roots = [canon(p) for p in (default_cache_roots(args.user) + args.cache_root) if os.path.exists(p)]

    active_pids = current_user_active_pids(uid)
    active_paths: set[str] = set()
    active_models: set[str] = set()

    for pid in active_pids:
        env = read_proc_env(pid)
        cmd = read_proc_cmd(pid)
        for key in ENV_KEYS:
            value = env.get(key, "").strip()
            if value:
                active_paths.add(canon(value))
        for src in [cmd] + list(env.values()):
            for model_id in MODEL_ID_PATTERN.findall(src):
                active_models.add("models--" + model_id.replace("/", "--"))

    open_paths = iter_open_paths(active_pids)
    candidates = gather_candidates(cache_roots)

    protected: list[tuple[str, list[str], int]] = []
    deletable: list[tuple[str, int]] = []

    for candidate in sorted(candidates):
        reasons: list[str] = []
        base = os.path.basename(candidate)
        if base in active_models:
            reasons.append("active-model-id")
        if any(is_under(candidate, active_path) for active_path in active_paths):
            reasons.append("under-active-cache-path")
        if any(is_under(open_path, candidate) for open_path in open_paths):
            reasons.append("open-fd")
        size = du_bytes(candidate)
        if reasons:
            protected.append((candidate, reasons, size))
        else:
            deletable.append((candidate, size))

    protected.sort(key=lambda x: x[2], reverse=True)
    deletable.sort(key=lambda x: x[1], reverse=True)

    total_reclaimable = sum(size for _, size in deletable)

    print("=== SAFE HF CACHE CLEANUP ===")
    print(f"user={args.user}")
    print(f"active_pids={len(active_pids)}")
    print("cache_roots:")
    for root in cache_roots:
        print(f"  - {root}")

    print("\nactive_cache_paths:")
    for path in sorted(active_paths):
        print(f"  - {path}")

    print(f"\nprotected_model_dirs={len(protected)}")
    for candidate, reasons, size in protected[: args.max_protected_print]:
        print(f"  KEEP {human(size):>8}  {candidate}  [{','.join(reasons)}]")
    if len(protected) > args.max_protected_print:
        print(f"  ... ({len(protected) - args.max_protected_print} more protected dirs)")

    print(f"\ndeletable_model_dirs={len(deletable)} reclaimable={human(total_reclaimable)}")
    for candidate, size in deletable:
        print(f"  DEL  {human(size):>8}  {candidate}")

    if not args.execute:
        print("\nDRY RUN only. Re-run with --execute to delete.")
        return

    print("\nEXECUTING DELETE...")
    deleted = 0
    freed = 0
    for candidate, size in deletable:
        try:
            shutil.rmtree(candidate)
            deleted += 1
            freed += size
            print(f"  OK   {human(size):>8}  {candidate}")
        except Exception as exc:
            print(f"  FAIL {human(size):>8}  {candidate} :: {exc}")
    print(f"\nDONE deleted={deleted}/{len(deletable)} estimated_freed={human(freed)}")


if __name__ == "__main__":
    main()
