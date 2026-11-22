"""Run Countdown training and evaluation on Modal; see README.md for commands."""

import datetime
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import uuid

import modal

from zero_shot_eval import make_eval_run_name

ROOT = Path(__file__).resolve().parent
REMOTE_ROOT = "/workspace"
RESULTS_ROOT = "/results"
RESULTS_VOLUME = "gstar-grpo-results"
app = modal.App("gstar-assignment-2-grpo")
cache = modal.Volume.from_name("gstar-grpo-cache", create_if_missing=True)
results = modal.Volume.from_name(RESULTS_VOLUME, create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential", "ninja-build")
    .pip_install("packaging", "ninja", "wheel", "setuptools")
    .pip_install_from_requirements(str(ROOT / "requirements-modal.txt"))
    .pip_install("flash-attn==2.7.4.post1", extra_options="--no-build-isolation",
                 env={"MAX_JOBS": "4", "TORCH_CUDA_ARCH_LIST": "8.0;9.0"})
    # Newer Arrow wheels require NumPy 2 at import time, while this vLLM uses NumPy 1.
    .pip_install("pyarrow==19.0.1", "pandas==2.2.3")
    .run_commands("python -m pip check")
    .env({"HF_HOME": "/cache/huggingface", "VLLM_USE_V1": "0",
          "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "TOKENIZERS_PARALLELISM": "false",
          "PYTHONUNBUFFERED": "1"})
    .workdir(REMOTE_ROOT)
)
# Sync changing source at container startup, outside the cached dependency image.
for name in ("starter.py", "zero_shot_eval.py", "test_training.py"):
    image = image.add_local_file(ROOT / name, f"{REMOTE_ROOT}/{name}", copy=False)


def build_command(task: str, output_dir: str, *, model: str = "Qwen/Qwen3-1.7B",
                  loss_type: str = "grpo", max_tokens: int = 256, steps: int = 80,
                  rollout_batch_size: int = 128, group_size: int = 8,
                  grad_acc_steps: int = 32, eval_every: int = 10, seed: int = 42,
                  lr: float = 7e-6, max_eval_examples: int = 0,
                  max_train_examples: int = 0, smoke: bool = False,
                  gpu_memory_utilization: float = 0.0, gradient_checkpointing: bool = True,
                  enforce_eager: bool = False) -> list[str]:
    """Build an argument list, shared by local validation and the remote worker."""
    if task not in ("train", "zero-shot", "check"):
        raise ValueError("task must be train, zero-shot, or check")
    if loss_type not in ("grpo", "dr_grpo"):
        raise ValueError("loss_type must be grpo or dr_grpo")
    if min(max_eval_examples, max_train_examples) < 0:
        raise ValueError("Example limits must be zero (full split) or positive")
    if max_tokens < 4 or max_tokens > 1536:
        raise ValueError("max_tokens must be between 4 and 1536 to leave prompt space in the 2048 context")
    if min(steps, rollout_batch_size, grad_acc_steps, eval_every) <= 0 or group_size < 2:
        raise ValueError("Training sizes/intervals must be positive and group_size must be at least 2")
    if gpu_memory_utilization != 0 and not 0 < gpu_memory_utilization < 1:
        raise ValueError("gpu_memory_utilization must be zero (automatic) or in (0, 1)")
    if lr <= 0:
        raise ValueError("lr must be positive")
    if smoke:
        steps, rollout_batch_size, group_size, grad_acc_steps, eval_every = 1, 4, 2, 4, 1
        max_train_examples, max_eval_examples = 4, 2
        enforce_eager = True
    if rollout_batch_size % group_size or rollout_batch_size % grad_acc_steps:
        raise ValueError("rollout_batch_size must be divisible by group_size and grad_acc_steps")
    if max_train_examples and max_train_examples < rollout_batch_size // group_size:
        raise ValueError("Training subset is smaller than the number of prompts per rollout")
    if task == "check":
        return [sys.executable, "-m", "unittest", "-v", "test_training"]
    script = "starter.py" if task == "train" else "zero_shot_eval.py"
    cmd = [sys.executable, "-u", script, "--model", model, "--max-tokens", str(max_tokens),
           "--seed", str(seed), "--output-dir", output_dir,
           "--gpu-memory-utilization", str(gpu_memory_utilization or (0.4 if task == "train" else 0.9))]
    if enforce_eager:
        cmd.append("--enforce-eager")
    if task == "train":
        for flag, value in (("loss-type", loss_type), ("steps", steps),
                            ("rollout-batch-size", rollout_batch_size), ("group-size", group_size),
                            ("grad-acc-steps", grad_acc_steps), ("eval-every", eval_every), ("lr", lr)):
            cmd.extend([f"--{flag}", str(value)])
        if gradient_checkpointing:
            cmd.append("--gradient-checkpointing")
        if max_train_examples:
            cmd.extend(["--max-train-examples", str(max_train_examples)])
        if max_eval_examples:
            cmd.extend(["--max-eval-examples", str(max_eval_examples)])
    elif max_eval_examples:
        cmd.extend(["--max-examples", str(max_eval_examples)])
    return cmd


@app.function(image=image, gpu=os.environ.get("GRPO_GPU", "A100-80GB"),
              cpu=8, memory=65536, timeout=24 * 60 * 60,
              volumes={"/cache": cache, RESULTS_ROOT: results})
def run_job(task: str, run_name: str, options: dict):
    output_dir = Path(RESULTS_ROOT) / run_name
    # A name identifies exactly one experiment; never mix old and new artifacts.
    output_dir.mkdir(parents=True, exist_ok=False)
    cmd = build_command(task, str(output_dir), **options)
    (output_dir / "launch.json").write_text(json.dumps({"task": task, "options": options, "command": cmd}, indent=2))
    (output_dir / "status.json").write_text(json.dumps({"state": "running"}))
    try:
        import importlib.metadata
        versions = {name: importlib.metadata.version(name) for name in
                    ("torch", "triton", "vllm", "transformers", "flash-attn", "datasets", "numpy", "pyarrow", "pandas")}
        (output_dir / "versions.json").write_text(json.dumps(versions, indent=2))
        print(json.dumps(versions, indent=2), flush=True)
        if task == "check":
            subprocess.run([sys.executable, "-c",
                "import torch, vllm, flash_attn, transformers, datasets, pyarrow, pandas; "
                "from torch.utils.tensorboard import SummaryWriter; "
                "from vllm.worker.worker import Worker; "
                "assert hasattr(Worker, '_assert_memory_footprint_increased_during_profiling'); "
                "assert torch.cuda.is_available(); print(torch.cuda.get_device_name())"], check=True)
        print("$", shlex.join(cmd), flush=True)
        # Stream output to Modal logs and a persistent text file; propagate failures.
        with (output_dir / "console.log").open("w") as log:
            with subprocess.Popen(cmd, cwd=REMOTE_ROOT, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                code = process.wait()
        (output_dir / "status.json").write_text(json.dumps({"state": "completed" if code == 0 else "failed", "exit_code": code}))
        if code:
            raise RuntimeError(f"{task} exited with code {code}; see {output_dir}/console.log")
    except Exception as error:
        (output_dir / "status.json").write_text(json.dumps({"state": "failed", "error": str(error)}))
        raise
    finally:
        try:
            cache.commit()
        finally:
            results.commit()
    return str(output_dir)


@app.local_entrypoint()
def main(task: str = "zero-shot", model: str = "Qwen/Qwen3-1.7B", loss_type: str = "grpo",
         max_tokens: int = 256, steps: int = 80, rollout_batch_size: int = 128,
         group_size: int = 8, grad_acc_steps: int = 32, eval_every: int = 10,
         seed: int = 42, lr: float = 7e-6, max_eval_examples: int = 0,
         max_train_examples: int = 0, smoke: bool = False,
         gpu_memory_utilization: float = 0.0, gradient_checkpointing: bool = True,
         enforce_eager: bool = False, run_name: str = "", dry_run: bool = False):
    options = {k: v for k, v in locals().items() if k not in {"task", "run_name", "dry_run"}}
    if not run_name:
        if task == "zero-shot":
            run_name = make_eval_run_name(model, max_tokens)
        else:
            timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            run_name = f"{task}_{loss_type}_l{max_tokens}_{timestamp}_{uuid.uuid4().hex[:6]}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_name):
        raise ValueError("run_name must contain only letters, numbers, underscores, dots, and hyphens")
    cmd = build_command(task, f"{RESULTS_ROOT}/{run_name}", **options)
    print("$", shlex.join(cmd))
    print(f"Results: Modal Volume {RESULTS_VOLUME}, directory /{run_name}")
    if dry_run:
        return
    print(run_job.remote(task, run_name, options))
