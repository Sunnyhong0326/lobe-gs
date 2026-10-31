"""Subprocess and GPU block scheduling without importing training code."""

import json
import os
import queue
import shlex
import subprocess
import sys
import threading
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]


def visible_gpu_ids() -> Optional[List[str]]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if raw:
        return [part.strip() for part in raw.split(",") if part.strip()]
    return None


def detect_gpus() -> List[str]:
    visible = visible_gpu_ids()
    if visible:
        return visible
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            text=True,
        )
        ids = [line.strip() for line in output.splitlines() if line.strip()]
        return ids or ["0"]
    except Exception:
        return ["0"]


def run_cmd(cmd: List[str], env=None, dry_run=False):
    prefix = ""
    if env and env.get("CUDA_VISIBLE_DEVICES"):
        prefix = f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} "
    print("[CMD]", prefix + " ".join(shlex.quote(str(x)) for x in cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)


def hydra_config_args(config_path: Path):
    return [
        "--config-path", str(config_path.parent),
        "--config-name", config_path.stem,
    ]


def read_block_ids(block_bounds_json: Path) -> List[int]:
    with block_bounds_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return [int(block["block_id"]) for block in payload.get("blocks", [])]


def launch_blocks(
    config_path: Path,
    block_ids: List[int],
    gpu_ids: List[str],
    port_start: int,
    dry_run: bool,
    checkpoint_iter=None,
    start_ply_root=None,
    ply_iteration=None,
    *,
    block_to_parent=None,
    python=None,
    command_runner=None,
):
    """Train blocks with one worker per GPU and a unique port for each job.

    Optional PLY initialization uses the same block ID, or its parent ID when
    ``block_to_parent`` is supplied. Dry runs emit commands without requiring
    PLY files. Callers retain control of their interpreter and command runner.
    """
    if not block_ids:
        print("[INFO] no blocks to train")
        return
    if not dry_run and not gpu_ids:
        raise RuntimeError("No GPUs available for fine block training.")

    execute = command_runner or run_cmd
    executable = python or sys.executable

    def build_command(block_id, port, validate_ply):
        cmd = [
            executable, "-u", "train_fine.py",
            *hydra_config_args(config_path),
            f"args.block_id={block_id}",
            f"args.port={port}",
        ]
        if checkpoint_iter is not None:
            cmd.append(f"args.checkpoint_iterations=[{int(checkpoint_iter)}]")
        if start_ply_root is not None:
            source_id = block_to_parent[block_id] if block_to_parent is not None else block_id
            ply_path = (
                Path(start_ply_root) / f"cell{source_id}" / "point_cloud"
                / f"iteration_{ply_iteration}" / "point_cloud.ply"
            )
            if validate_ply and not ply_path.is_file():
                raise FileNotFoundError(f"Missing initialization PLY for block {block_id}: {ply_path}")
            cmd.extend([
                "model_params.use_pretrain=true",
                f"model_params.coarse_ply_path={ply_path}",
                "model_params.coarse_model_path=",
            ])
        return cmd

    if dry_run:
        for offset, block_id in enumerate(block_ids):
            execute(build_command(block_id, port_start + offset, False), dry_run=True)
        return

    jobs = queue.Queue()
    for block_id in block_ids:
        jobs.put(block_id)
    failures = []
    lock = threading.Lock()
    next_port = int(port_start)

    def take_port():
        nonlocal next_port
        with lock:
            port = next_port
            next_port += 1
            return port

    def worker(gpu_id):
        while True:
            try:
                block_id = jobs.get_nowait()
            except queue.Empty:
                return
            try:
                cmd = build_command(block_id, take_port(), True)
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                execute(cmd, env=env, dry_run=False)
            except Exception as exc:
                with lock:
                    failures.append(exc)
            finally:
                jobs.task_done()

    threads = [
        threading.Thread(target=worker, args=(gpu,), name=f"block-gpu-{gpu}")
        for gpu in gpu_ids
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise RuntimeError(str(failures[0])) from failures[0]
