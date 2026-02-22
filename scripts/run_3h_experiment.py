from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _now_iso() -> str:
    # Use local time with offset for easy human reading.
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _fmt_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser("run-3h-experiment")
    p.add_argument("--exp_dir", required=True, type=str)
    p.add_argument("--duration", default="3h", type=str, help="Passed to `timeout` (e.g. 3h, 10800s)")
    p.add_argument("--keep", default=2, type=int, help="How many checkpoints to keep after pruning")
    p.add_argument(
        "cmd",
        nargs=argparse.REMAINDER,
        help="Command to run after `--` (e.g. -- python -m genegan.cli.train ...)",
    )
    args = p.parse_args(argv)

    cmd = list(args.cmd)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        p.error("Missing command. Use: scripts/run_3h_experiment.py --exp_dir ... -- <cmd...>")

    exp_dir = Path(args.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    watch_log = exp_dir / "watch.log"
    train_log = exp_dir / "train.log"

    with watch_log.open("a", encoding="utf-8") as wf:
        wf.write(f"start={_now_iso()}\n")
        wf.write(f"duration={args.duration}\n")
        wf.write(f"keep={int(args.keep)}\n")
        wf.write(f"cmd={_fmt_cmd(cmd)}\n")
        wf.flush()

    # Run training for a fixed wall-clock budget and allow graceful shutdown via SIGINT.
    timeout_cmd = ["timeout", "-s", "INT", str(args.duration), *cmd]
    with train_log.open("a", encoding="utf-8") as tf:
        tf.write(f"\n\n# [{_now_iso()}] RUN: {_fmt_cmd(timeout_cmd)}\n")
        tf.flush()
        proc = subprocess.run(timeout_cmd, stdout=tf, stderr=subprocess.STDOUT, check=False)

    with watch_log.open("a", encoding="utf-8") as wf:
        wf.write(f"train_exit_code={int(proc.returncode)}\n")
        wf.write(f"train_end={_now_iso()}\n")
        wf.flush()

    # Prune checkpoints after the run to save disk.
    prune_cmd = [
        sys.executable,
        "scripts/prune_checkpoints.py",
        "--ckpt_dir",
        str(exp_dir / "checkpoints"),
        "--keep",
        str(int(args.keep)),
    ]
    prune = subprocess.run(prune_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    with watch_log.open("a", encoding="utf-8") as wf:
        wf.write(f"prune_exit_code={int(prune.returncode)}\n")
        if prune.stdout:
            wf.write(prune.stdout.rstrip() + "\n")
        wf.write(f"prune_end={_now_iso()}\n")
        wf.flush()

    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())

