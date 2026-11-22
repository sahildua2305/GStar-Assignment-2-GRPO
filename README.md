# GRPO for Countdown Math Problems

This project implements the **Group Relative Policy Optimization (GRPO)** algorithm to fine-tune a language model on the "Countdown" math task. The goal is to train an LLM to generate correct mathematical equations that reach a target number using a given set of integers.

## Setup

For Modal, install only the Modal client on your laptop; CUDA dependencies are installed remotely:

```bash
python -m pip install 'modal>=1.0,<2'
python -m modal setup  # only needed if this machine is not already authenticated
```

The remote image uses Python 3.11, CUDA 12.4, and [requirements-modal.txt](./requirements-modal.txt),
then installs FlashAttention 2.7.4.post1 without build isolation. The original README pinned
vLLM 0.7.2, which predates Qwen3 support; this setup uses vLLM 0.8.5.post1, Torch 2.6.0,
and Triton 3.2.0. See the [vLLM supported models](https://docs.vllm.ai/en/v0.8.5/models/supported_models.html)
and [matching Torch requirements](https://github.com/vllm-project/vllm/blob/v0.8.5.post1/requirements/cuda.txt).
The image also pins PyArrow 19.0.1 and pandas 2.2.3 for compatibility with NumPy 1.26.4;
newer Arrow wheels can fail at import even when dependency resolution succeeds.
The V0 engine and single-process executor are required by the training code's in-process weight synchronization.

## Running on Modal

[modal_run.py](./modal_run.py) defaults to one **A100 80GB**, following the assignment's A100
requirement and leaving room for both the policy/optimizer and vLLM. To choose another GPU,
prefix an invocation with `GRPO_GPU=H100`. This setup runs on a single GPU.

The dependency image is cached. Source files use
[runtime mounts with `copy=False`](https://modal.com/docs/guide/images#add-local-files-with-add_local_dir-and-add_local_file),
so editing Python code and rerunning syncs it without rebuilding dependencies.
Public model/data downloads need no Hugging Face secret.

First check dependencies and the loss/reduction tests, then run small end-to-end checks:

```bash
# Build the image, verify CUDA/vLLM/FlashAttention imports, and run unit tests.
python -m modal run modal_run.py --task check

# Two test examples, no training.
python -m modal run modal_run.py --task zero-shot --smoke --max-tokens 32

# One optimizer step: four rollouts, two prompts, two evaluation examples.
python -m modal run modal_run.py --task train --smoke --max-tokens 32
python -m modal run modal_run.py --task train --loss-type dr_grpo --smoke --max-tokens 512
```

Smoke runs check execution only; their token limits, tiny test subset, and smaller groups do not
measure assignment accuracy. `--smoke` sets one training step, rollout batch size 4, group size 2,
gradient accumulation 4, four training examples, two test examples, and eager vLLM execution.
It preserves your chosen `--max-tokens`.

Section 1.6 of the PDF asks for three zero-shot baselines and GRPO/Dr.GRPO training at two lengths.
Run the full experiments with:

```bash
# Zero-shot baselines on the entire test split, temperature=1.0, seed=42.
for length in 256 512 1024; do
    python -m modal run modal_run.py --task zero-shot --max-tokens "$length"
done

# Four training runs, with full test evaluation at step 0, every 10 steps, and the final step.
for loss in grpo dr_grpo; do
    for length in 256 512; do
        python -m modal run modal_run.py --task train --loss-type "$loss" --max-tokens "$length"
    done
done
```

Training defaults match `starter.py`: model `Qwen/Qwen3-1.7B`, 80 steps, rollout batch 128,
group size 8, 32 accumulation steps (microbatch 4), learning rate `7e-6`, seed 42, and vLLM memory
fraction 0.4. Modal additionally enables gradient checkpointing to reduce activation memory.
Use `--no-gradient-checkpointing` to disable it. Evaluation uses vLLM memory fraction 0.9.
The test split is used for periodic evaluation as in the assignment; no test examples are used for training.

For example, to run in the background and name the results:

```bash
python -m modal run --detach modal_run.py --task train \
    --loss-type grpo --max-tokens 512 --run-name grpo-l512-seed42
```

For a shorter pilot without changing source:

```bash
python -m modal run modal_run.py --task train --steps 5 --max-eval-examples 32
```

Other options include `--max-train-examples`, `--eval-every`, `--rollout-batch-size`,
`--group-size`, `--grad-acc-steps`, `--lr`, `--seed`, and `--gpu-memory-utilization`.
Example limits default to zero, meaning the full split. Rollout batch size must be divisible by
both group size and accumulation steps. If training runs out of memory, increase `--grad-acc-steps`
to 64 or 128 (microbatch 2 or 1) or lower the vLLM memory fraction. Each function has a 24-hour
timeout; full-run duration and memory use have not been benchmarked.

Use `python -m modal run modal_run.py --help` for all flags. You can inspect generated commands
without allocating a GPU by importing the launcher directly:

```bash
python -c 'from modal_run import main; main(task="train", smoke=True, dry_run=True)'
```

## Outputs

The `gstar-grpo-cache` Modal Volume keeps Hugging Face downloads between invocations.
The `gstar-grpo-results` Volume stores a separate directory for every run. The launcher prints
its directory and uses a timestamp plus random suffix unless you pass `--run-name`. Reusing
an existing run name fails to protect previous results.

Every run saves `launch.json`, `versions.json`, `console.log`, and its exit status. Training also saves:

- `config.json`: training arguments.
- `train_metrics.jsonl`: loss, reward statistics, gradient norm, and output length per step.
- `eval_metrics.jsonl` and `eval_step_*.json`: test metrics and generated examples.
- `tb/`: TensorBoard events, including accuracy at every evaluation.
- `checkpoint/`: final Hugging Face model and tokenizer, saved on successful completion.

Zero-shot evaluation saves `config.json`, `metrics.json`, and `examples.json`. The results include
accuracy, reward counts, output lengths, and responses for success/failure analysis in the report.
The model checkpoint is saved at the end; interrupted runs keep written logs but cannot resume optimizer state.

Download results and view training curves locally:

```bash
python -m modal volume ls gstar-grpo-results
python -m modal volume get gstar-grpo-results / ./output
python -m pip install tensorboard
tensorboard --logdir ./output
```

To evaluate a saved checkpoint with the same test evaluator, pass its remote path:

```bash
python -m modal run modal_run.py --task zero-shot \
    --model /results/grpo-l512-seed42/checkpoint --max-tokens 512
```

That measures a trained checkpoint; use the base model ID for the assignment's zero-shot baselines.

Automatic evaluation directory names include the source model or checkpoint's parent and name.
For example, evaluating `/results/grpo-l512-seed42/checkpoint` produces
`zero-shot_grpo-l512-seed42-checkpoint_l512_<timestamp>_<id>`. Base-model evaluation includes
the model ID instead, such as `zero-shot_Qwen-Qwen3-1.7B_l256_<timestamp>_<id>`.
This applies to both Modal and direct `zero_shot_eval.py` runs. Explicit `--run-name` (Modal)
or `--output-dir` (direct script) overrides the automatic name. The exact source model path
is also saved in `config.json` and, on Modal, `launch.json`.

## Local checks and implementation notes

With CPU PyTorch installed, run `python -m unittest -v test_training`. These checks cover PPO
clipping and gradients, both masked reductions, accumulation equivalence, reward multiplicities,
advantage shapes, and synchronization before final evaluation. They do not replace GPU smoke runs.

Validated on a Modal A100 80GB: all nine unit tests, two-example zero-shot evaluation at 32 tokens,
one-step GRPO training at 32 tokens, and one-step Dr.GRPO training at 512 tokens. The Dr.GRPO run
produced a nonzero gradient; both training runs saved checkpoints and evaluation metrics.
The full 80-step experiments, full-test baselines, default microbatch size 4, and CUDA graph mode
have not been validated by these smoke runs (which use microbatch size 1 and eager execution).

The setup also repairs execution/measurement issues in the starter: `torch.exp`, clipping the
probability ratio, the `(loss, metadata)` return contract, flattened raw rewards, vLLM output-token
access, cache invalidation after weight updates, fresh weights for evaluation, final-step evaluation,
and the extra division in logged training loss. The number verifier now checks multiplicities
while keeping the starter docstring's exact-number-use rule. The PDF's Task 1 subset example
conflicts with its earlier exactly-once rule; this setup retains the starter's exactly-once semantics.

## Assignment Structure

Instruction: [gstar_assignment2.pdf](./gstar_assignment2.pdf)

Main File: `starter.py`:

### Problem 1
#### Part 1: GRPO Pipeline Implementation
-   **Task 1**: Implement reward helper functions (`_extract_answer`, `_validate_numbers`, `_evaluate_equation`).
-   **Task 2**: Implement the main `reward_fn`.
-   **Task 3**: Implement `compute_group_normalized_advantages`.
-   **Task 5**: Implement `masked mean` for GRPO and `masked_mean_drgrpo` for DR.GRPO.

#### Part 2: Main Experiments and Report (please refer to section 1.6 of [gstar_assignment2.pdf](./gstar_assignment2.pdf) for the Main Report and Experiments

### Problem 2: (Optional) Open-ended Investigation. Please refer to Section 2 of [gstar_assignment2.pdf](./gstar_assignment2.pdf)
