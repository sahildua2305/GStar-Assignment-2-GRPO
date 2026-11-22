# In this assignment, you will implement the core components of the
# Group-Reward Policy Optimization (GRPO) algorithm. You'll be working with a
# math-solving task called "Countdown," where the model has to generate an
# equation to reach a target number using a given set of numbers.
#
# You will need to implement six key functions:
# 1. `_extract_answer`: To parse the model's response.
# 2. `_validate_numbers`: To ensure the generated equation uses the correct numbers.
# 3. `_evaluate_equation`: To safely calculate the result of the generated equation.
# 4. `reward_fn`: To score the model's generated equations using the helpers.
# 5. `compute_group_normalized_advantages`: To calculate advantages.
# 6. `compute_loss`: To compute per-token loss.
# 7. `masked_mean`: To compute the mean of masked tensor, used in GRPO
# 8. `masked_mean_drgrpo`: To compute the mean of masked tensor, used in DR-GRPO
# ==============================================================================

from __future__ import annotations

import argparse
import json
import os
import datetime
import random
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Tuple, Any
import logging
import warnings

import torch
import torch.nn.functional as F
from unittest.mock import patch
import re

if TYPE_CHECKING:
    from transformers import AutoTokenizer, PreTrainedModel
    from vllm import LLM, SamplingParams
    from torch.utils.tensorboard import SummaryWriter

logging.getLogger("vllm.engine.scheduler").setLevel(logging.ERROR)
os.environ["VLLM_USE_V1"] = "0"

from torch.optim.lr_scheduler import LambdaLR

def get_constant_schedule_with_warmup(optimizer: torch.optim.Optimizer, num_warmup_steps: int, last_epoch: int = -1):
    def lr_lambda(current_step: int):
        return min(1.0, float(current_step) / float(max(1, num_warmup_steps)))
    return LambdaLR(optimizer, lr_lambda, last_epoch)

# -------------------------
# Prompting helpers (Countdown-style)
# -------------------------
TEMPLATE = """Using the numbers {numbers}, create an equation that equals {target}. 
You can use basic arithmetic operations (+, -, *, /) and each number can only be used once.
Show your reasoning in <think> </think> tags. And return the final equation in <answer> </answer> tags. Keep your reasoning under {max_tokens} tokens.
For example, numbers = [1, 2, 3, 4] and target = 5, the answer is <answer>(1 + 2) * 3 - 4</answer>."""


# vLLM utilities
def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85,
              max_model_len: int = 2048, max_num_seqs: int = 128,
              enforce_eager: bool = False) -> LLM:
    from vllm import LLM

    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            device=device,
            seed=seed,
            enforce_eager=enforce_eager,
            distributed_executor_backend="uni",
        )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM) -> None:
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())
    # Cached KV states belong to the previous weights.
    llm.reset_prefix_cache()


def init_sampling_params(temperature: float, min_tokens: int, max_tokens: int) -> SamplingParams:
    from vllm import SamplingParams

    sp = SamplingParams(
        temperature=temperature,
        top_p=1.0,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        logprobs=0,
    )
    sp.stop = ["</answer>"]
    sp.include_stop_str_in_output = True
    return sp


# Tokenization utilities
def tokenize_prompt_and_output(prompt_strs: List[str], output_strs: List[str], tokenizer: AutoTokenizer) -> Dict[str, torch.Tensor]:
    batch_data = []
    max_len = 0
    for prompt, output in zip(prompt_strs, output_strs):
        prompt_tokens = tokenizer(prompt)["input_ids"]
        output_tokens = tokenizer(output)["input_ids"]
        combined_tokens = prompt_tokens + output_tokens
        max_len = max(max_len, len(combined_tokens))
        batch_data.append({
            "tokens": combined_tokens, "prompt_len": len(prompt_tokens), "total_len": len(combined_tokens)
        })
    batch_size = len(batch_data)
    input_ids = torch.full((batch_size, max_len - 1), tokenizer.eos_token_id, dtype=torch.long)
    labels = torch.full((batch_size, max_len - 1), tokenizer.eos_token_id, dtype=torch.long)
    response_mask = torch.zeros((batch_size, max_len - 1), dtype=torch.bool)
    for i, data in enumerate(batch_data):
        tokens, seq_len = torch.tensor(data["tokens"]), len(data["tokens"])
        input_ids[i, :seq_len-1], labels[i, :seq_len-1] = tokens[:-1], tokens[1:]
        response_start, response_end = data["prompt_len"] - 1, seq_len - 1
        if response_end > response_start:
            response_mask[i, response_start:response_end] = True
    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}



# Reward function and evaluation (Countdown solution)

# ==============================================================================
# TASK 1: Implement Reward Helper Functions
# Before you can write the reward function, you need to create three helper 
# functions to parse and evaluate the model's generated text.
# ==============================================================================

def _extract_answer(solution_str: str) -> str | None:
    """
    Extract the content from the last <answer>...</answer> tag in the given string.

    Hint: Use the `re` module. `re.finditer` can find all occurrences of a pattern.
    Args:
        solution_str: The full text generated by the model.

    Returns:
        The stripped string content of the last answer tag, or None if no tag is found.
    """
    ### YOUR CODE HERE ###
    for match in re.finditer(r"<answer>(.*?)</answer>", solution_str, re.DOTALL):
        last_answer = match.group(1).strip()
    return last_answer if 'last_answer' in locals() else None
    ### END YOUR CODE ###


def _validate_numbers(equation_str: str, available_numbers: List[int]) -> bool:
    """
    Check if the numbers used in the equation are exactly the ones available.

    Hint:
    1. Use `re.findall(r"\d+", equation_str)` to get all number strings from the equation.
    2. Remember to handle potential errors and return False.

    Args:
        equation_str: The equation string to validate.
        available_numbers: A list of integers that are allowed to be used.

    Returns:
        True if the equation uses the correct numbers, False otherwise.
    """
    ### YOUR CODE HERE ###
    all_nums_in_equation = re.findall(r"\d+", equation_str)
    return Counter(map(int, all_nums_in_equation)) == Counter(available_numbers)
    ### END YOUR CODE ###


def _evaluate_equation(equation_str: str) -> float | None:
    """
    Safely evaluate a mathematical equation string.
    Args:
        equation_str: The equation string to evaluate.
    Returns:
        The result of the equation as a float, or None if it's invalid or unsafe.
    """
    ### YOUR CODE HERE ###
    try:
        # Only allow digits, operators, and parentheses
        if not re.fullmatch(r"[0-9+\-*/(). ]+", equation_str):
            return None
        result = eval(equation_str)
        return float(result)
    except:
        return None
    ### END YOUR CODE ###

# ==============================================================================
# TASK 2: Implement the Reward Function
# ==============================================================================
def reward_fn(generated_text: str, ground_truth: Dict) -> float:
    """
    Reward function for countdown problems.
    - 1.0 (Perfect): The equation is valid, uses the correct numbers, and evaluates to the target.
    - 0.1 (Partial): The text contains an <answer> tag, but the equation is incorrect for any reason.
    - 0.0 (Failed): The `generated_text` does not contain an <answer> tag.

    Args:
        generated_text: The full text output from the language model.
        ground_truth: A dictionary containing `target` and `numbers`.

    Returns:
        A float value representing the reward, such as 1.0, 0.1, or 0.0
    """
    ### YOUR CODE HERE ###
    answer = _extract_answer(generated_text)
    if answer is None:
        return 0.0
    if not _validate_numbers(answer, ground_truth["numbers"]):
        return 0.1
    result = _evaluate_equation(answer)
    if result is None or result != ground_truth["target"]:
        return 0.1
    return 1.0
    ### END YOUR CODE ###


def evaluate_model(llm: LLM, sampling_params: SamplingParams, eval_prompts: List[str], eval_answers: List[Dict]) -> Dict[str, Any]:
    rollouts = llm.generate(eval_prompts, sampling_params)
    examples, rewards, output_token_lengths = [], [], []
    for rollout, gt in zip(rollouts, eval_answers):
        response_text = rollout.outputs[0].text
        reward_value = reward_fn(response_text, gt)
        equation = _extract_answer(response_text)
        result = _evaluate_equation(equation) if equation is not None else None
        output_tokens = len(rollout.outputs[0].token_ids)
        output_token_lengths.append(output_tokens)
        examples.append({
            "prompt": rollout.prompt, "response": response_text, "answer": gt, "equation": equation,
            "result": result, "reward": reward_value, "output_tokens": output_tokens,
        })
        rewards.append(reward_value)
    rewards_tensor = torch.tensor(rewards) if rewards else torch.tensor([0.0])
    tol = 1e-8
    count_correct = sum(1 for r in rewards if abs(r - 1.0) < tol)
    count_partial = sum(1 for r in rewards if abs(r - 0.1) < tol)
    count_failed = sum(1 for r in rewards if abs(r - 0.0) < tol)
    accuracy = (count_correct / len(rewards)) * 100 if rewards else 0.0
    avg_output_tokens = sum(output_token_lengths) / len(output_token_lengths) if output_token_lengths else 0.0
    return {
        "mean_reward": float(rewards_tensor.mean().item()),
        "std_reward": float(rewards_tensor.std().item()) if rewards_tensor.numel() > 1 else 0.0,
        "num_examples": len(rewards), "examples": examples, "count_correct": count_correct,
        "count_partial": count_partial, "count_failed": count_failed, "accuracy": accuracy,
        "avg_output_tokens": avg_output_tokens,
    }


def _format_eval_example(example: Dict[str, Any]) -> str:
    target = example["answer"]["target"] if isinstance(example.get("answer"), dict) and "target" in example["answer"] else "?"
    numbers = example["answer"].get("numbers") if isinstance(example.get("answer"), dict) else None
    return (
        f"Prompt: {example.get('prompt', '')}\n"
        f"Response: {example.get('response', '')}\n"
        f"Equation: {example.get('equation', None)} | Result: {example.get('result', None)} | Target: {target} | Numbers: {numbers}\n"
        f"Reward: {example.get('reward', 0.0):.3f}\n"
    )


def log_train(rollout_batch_loss: float, grad_norm: float, reward_metadata: Dict[str, Any], avg_output_tokens: float, writer: SummaryWriter | None, step: int) -> None:
    if writer is None:
        return
    writer.add_scalar("train/loss", float(rollout_batch_loss), global_step=step)
    writer.add_scalar("train/grad_norm", float(grad_norm), global_step=step)
    writer.add_scalar("train/reward_mean", float(reward_metadata["mean"]), global_step=step)
    writer.add_scalar("train/reward_std", float(reward_metadata["std"]), global_step=step)
    writer.add_scalar("train/avg_output_tokens", float(avg_output_tokens), global_step=step)
    print(f"Step {step} | Loss: {rollout_batch_loss:.4f} | Grad norm: {grad_norm:.4f} | Reward mean: {float(reward_metadata['mean']):.4f} | Reward std: {float(reward_metadata['std']):.4f} | Avg output tokens: {avg_output_tokens:.1f}")


def log_eval(metrics: Dict[str, Any], writer: SummaryWriter | None, step: int) -> None:
    examples = metrics.get("examples", []) or []
    if not examples: return
    tol = 1e-8
    correct_examples = [ex for ex in examples if abs(float(ex.get("reward", 0.0)) - 1.0) < tol][:10]
    partial_examples = [ex for ex in examples if abs(float(ex.get("reward", 0.0)) - 0.1) < tol][:10]
    failed_examples = [ex for ex in examples if abs(float(ex.get("reward", 0.0)) - 0.0) < tol][:10]
    if correct_examples:
        print(f"\n=== Eval examples (CORRECT, reward=1.0) @ step {step} ===")
        for idx, ex in enumerate(correct_examples[:2], 1): print(f"[CORRECT #{idx}]\n" + _format_eval_example(ex))
    if partial_examples:
        print(f"\n=== Eval examples (PARTIAL, reward=0.1) @ step {step} ===")
        for idx, ex in enumerate(partial_examples[:2], 1): print(f"[PARTIAL #{idx}]\n" + _format_eval_example(ex))
    if failed_examples:
        print(f"\n=== Eval examples (FAILED, reward=0.0) @ step {step} ===")
        for idx, ex in enumerate(failed_examples[:2], 1): print(f"[FAILED #{idx}]\n" + _format_eval_example(ex))
    if writer:
        correct_text = "\n\n".join([_format_eval_example(ex) for ex in correct_examples]) or ""
        partial_text = "\n\n".join([_format_eval_example(ex) for ex in partial_examples]) or ""
        failed_text = "\n\n".join([_format_eval_example(ex) for ex in failed_examples]) or ""
        if correct_text: writer.add_text("eval/examples_correct", correct_text, global_step=step)
        if partial_text: writer.add_text("eval/examples_partial", partial_text, global_step=step)
        if failed_text: writer.add_text("eval/examples_failed", failed_text, global_step=step)
    print(f"Eval @ step {step}: accuracy={metrics['accuracy']:.1f}% mean_reward={metrics['mean_reward']:.4f} "
          f"avg_tokens={metrics['avg_output_tokens']:.1f} | correct:{metrics['count_correct']} "
          f"partial:{metrics['count_partial']} failed:{metrics['count_failed']}")


# -------------------------
# Advantages and GRPO-clip loss
# -------------------------
# ==============================================================================
# TASK 3: Implement Group Normalized Advantages
# ==============================================================================
def compute_group_normalized_advantages(
    rollout_responses: List[str],
    repeated_ground_truths: List[Dict],
    reward_fn: Callable[[str, Dict], float],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """ Computes advantages by normalizing rewards within groups.
    
    Args:
        rollout_responses: List of generated responses (length = batch_size * group_size)
        repeated_ground_truths: Ground truth repeated for each response
        reward_fn: Function to compute rewards
        group_size: Number of responses per question (G)
        advantage_eps: Small constant for numerical stability (epsilon)
        normalize_by_std: If True, normalize by std (GRPO); if False, don't (DR-GRPO)
    
    Returns:
        advantages: Flattened tensor of advantages (shape: [batch_size * group_size])
        raw_rewards: Original rewards before normalization
        metadata: Dictionary with 'mean', 'std', 'max', 'min' of raw rewards
            (e.g. metadata = {
                "mean": torch.mean(raw_rewards),
                "std": torch.std(raw_rewards),
                "max": torch.max(raw_rewards),
                "min": torch.min(raw_rewards),
            })
    """

    # Steps:
    # 1. Calculate the raw reward for each response using the provided `reward_fn`.
    # 2. Reshape the 1D tensor of raw rewards into a 2D tensor of shape (-1, `group_size`).
    # 3. Calculate the mean reward for each group.
    # 4. Compute the advantage by subtracting the group's mean reward.
    # 5. If `normalize_by_std` is True, normalize advantages by `(group_std + advantage_eps)`.
    # 6. Flatten the advantages tensor back into a 1D tensor.
    # 7. Create a `metadata` dictionary with overall statistics of the raw rewards.
    advantages, raw_rewards, metadata = None, None, {}
    ### YOUR CODE HERE ###
    rewards = []
    for resp, ground_truth in zip(rollout_responses, repeated_ground_truths):
        rewards.append(reward_fn(resp, ground_truth))

    raw_rewards = torch.tensor(rewards, dtype=torch.float32)
    raw_rewards = raw_rewards.view(-1, group_size)

    group_means = raw_rewards.mean(dim=1, keepdim=True)
    advantages = raw_rewards - group_means

    if normalize_by_std:
        group_stds = raw_rewards.std(dim=1, keepdim=True) + advantage_eps
        advantages = advantages / group_stds

    advantages = advantages.flatten()

    metadata = {
        "mean": raw_rewards.mean().item(),
        "std": raw_rewards.std().item(),
        "max": raw_rewards.max().item(),
        "min": raw_rewards.min().item()
    }
    ### END YOUR CODE ###
    return advantages, raw_rewards.flatten(), metadata


# ==============================================================================
# TASK 4: Implement per-token Loss
# ==============================================================================
def compute_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    clip_range: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Computes the per-token PPO clipped surrogate loss.

    Why omit the KL divergence term?
    For simplicity and following trends in recent RLVR (e.g Dr.GRPO),
    we omit the KL penalty term often found in PPO. This simplifies the implementation
    and has been shown to work well in practice.

    Steps:
    1. Calculate the probability ratio `pi_ratio = exp(policy_log_probs - old_log_probs)`.
    2. Calculate the unclipped term: `advantages * pi_ratio`.
    3. Calculate the clipped term by clipping `pi_ratio` to `[1-clip_range, 1+clip_range]`
       and then multiplying by `advantages`.
    4. The final loss is `-torch.minimum(unclipped_term, clipped_term)`.
    """
    loss = 0.0
    ### YOUR CODE HERE ###
    if advantages.ndim == 1:
        advantages = advantages.unsqueeze(-1)
    pi_ratio = torch.exp(policy_log_probs - old_log_probs)
    unclipped_term = pi_ratio * advantages
    clipped_term = torch.clip(pi_ratio, min=1-clip_range, max=1+clip_range) * advantages
    loss = -torch.minimum(unclipped_term, clipped_term)
    ### END YOUR CODE ###
    return loss, {}


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute the mean of tensor values where mask=True for each row, then average across the batch.
    """
    ### YOUR CODE HERE ###
    masked_tensor = torch.where(mask, tensor, 0.0)
    num_valid_pos = mask.sum(dim=1).clamp(min=1)
    row_means =  masked_tensor.sum(dim=1) / num_valid_pos
    return row_means.mean()
    ### END YOUR CODE ###

def masked_mean_drgrpo(tensor: torch.Tensor, mask: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """
    Compute the sum of tensor values where mask=True, divided by num_tokens, then average across the batch.
    This is used for the DR-GRPO loss
    """
    ### YOUR CODE HERE ###
    masked_tensor = torch.where(mask, tensor, 0.0)
    row_means = masked_tensor.sum(dim=1) / max(num_tokens, 1)
    return row_means.mean()
    ### END YOUR CODE ###

def get_response_log_probs(model: PreTrainedModel, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logits = model(input_ids).logits
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return log_probs


# Training helpers and loop
def duplicate_data(arr: List, group_size: int) -> List:
    return [x for x in arr for _ in range(group_size)]


def rollout_with_vllm(policy: PreTrainedModel, llm: LLM, sampling_params: SamplingParams, prompts_batch: List[str], group_size: int) -> Tuple[List[str], List[str], List[int]]:
    load_policy_into_vllm_instance(policy, llm)
    prompts_dup = duplicate_data(prompts_batch, group_size)
    vllm_rollouts = llm.generate(prompts_dup, sampling_params, use_tqdm=False)
    rollout_input_text, rollout_response_text, rollout_output_tokens = [], [], []
    for rollout in vllm_rollouts:
        for r in rollout.outputs:
            rollout_input_text.append(rollout.prompt)
            rollout_response_text.append(r.text)
            rollout_output_tokens.append(len(r.token_ids))
    return rollout_input_text, rollout_response_text, rollout_output_tokens


def tokenize_rollouts(rollout_input_text: List[str], rollout_response_text: List[str], tokenizer: AutoTokenizer) -> Dict[str, torch.Tensor]:
    return tokenize_prompt_and_output(rollout_input_text, rollout_response_text, tokenizer)


def grpo_microbatch_step(
    policy: PreTrainedModel, input_ids: torch.Tensor, labels: torch.Tensor, response_mask: torch.Tensor,
    advantages_per_seq: torch.Tensor, gradient_accumulation_steps: int, clip_range: float,
    loss_type: str = "grpo", max_completion_length: int = 256,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    policy_log_probs = get_response_log_probs(policy, input_ids, labels)
    old_log_probs = policy_log_probs.detach()
    advantages = advantages_per_seq.unsqueeze(-1)
    loss_per_token, metadata = compute_loss(advantages, policy_log_probs, old_log_probs, clip_range)
    if loss_type == "grpo":
        loss = masked_mean(loss_per_token, response_mask)
    elif loss_type == "dr_grpo":
        loss = masked_mean_drgrpo(loss_per_token, response_mask, max_completion_length)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    loss = loss / gradient_accumulation_steps
    loss.backward()
    return loss.detach(), metadata


def train(
    policy: PreTrainedModel, tokenizer: AutoTokenizer, llm: LLM, sampling_params: SamplingParams, *,
    train_prompts: List[str], train_answers: List[Dict], eval_prompts: List[str], eval_answers: List[Dict],
    optimizer: torch.optim.Optimizer, scheduler, n_grpo_steps: int, rollout_batch_size: int,
    group_size: int, gradient_accumulation_steps: int, clip_range: float, use_std_normalization: bool,
    advantage_eps: float, device: str, eval_every: int = 5, writer: SummaryWriter = None, seed: int,
    loss_type: str = "grpo", max_completion_length: int = 256,
    output_dir: str | None = None,
) -> None:
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    micro_train_batch_size = rollout_batch_size // gradient_accumulation_steps
    random.seed(seed)
    train_step = 0

    def record_eval(step):
        # Evaluate the policy after the latest optimizer step, including the final step.
        load_policy_into_vllm_instance(policy, llm)
        metrics = evaluate_model(llm, sampling_params, eval_prompts, eval_answers)
        if writer:
            for k in ["accuracy", "mean_reward", "std_reward", "avg_output_tokens", "count_correct", "count_partial", "count_failed"]:
                writer.add_scalar(f"eval/{k}", metrics[k], global_step=step)
            writer.flush()
        log_eval(metrics, writer, step)
        if output_dir:
            path = Path(output_dir)
            summary = {k: v for k, v in metrics.items() if k != "examples"}
            with (path / "eval_metrics.jsonl").open("a") as f:
                f.write(json.dumps({"step": step, **summary}) + "\n")
            with (path / f"eval_step_{step:04d}.json").open("w") as f:
                json.dump(metrics, f, indent=2)
        return metrics

    record_eval(train_step)

    for _ in range(n_grpo_steps):
        sampled = random.sample(list(zip(train_prompts, train_answers)), n_prompts_per_rollout_batch)
        prompts_batch, answers_batch = [p for p, _ in sampled], [a for _, a in sampled]
        rollout_input, rollout_response, rollout_tokens = rollout_with_vllm(policy, llm, sampling_params, prompts_batch, group_size)
        answers_dup = duplicate_data(answers_batch, group_size)
        avg_output_tokens = sum(rollout_tokens) / len(rollout_tokens) if rollout_tokens else 0.0
        advantages, _, reward_meta = compute_group_normalized_advantages(
            rollout_response, answers_dup, reward_fn, group_size, advantage_eps, use_std_normalization
        )
        tokenized = tokenize_rollouts(rollout_input, rollout_response, tokenizer)
        optimizer.zero_grad()
        rollout_loss = 0.0
        for micro_idx in range(0, rollout_batch_size, micro_train_batch_size):
            s = slice(micro_idx, micro_idx + micro_train_batch_size)
            loss, _ = grpo_microbatch_step(
                policy, tokenized["input_ids"][s].to(device), tokenized["labels"][s].to(device),
                tokenized["response_mask"][s].to(device), advantages[s].to(device),
                gradient_accumulation_steps, clip_range, loss_type=loss_type, max_completion_length=max_completion_length
            )
            rollout_loss += float(loss.item())
        grad_norm = torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.grad is not None], 1.0)
        optimizer.step()
        scheduler.step()
        train_step += 1
        print(f"Step {train_step} | Loss: {rollout_loss:.4f} | Grad: {grad_norm:.4f} | "
              f"Reward mean: {reward_meta['mean']:.4f} | Reward std: {reward_meta['std']:.4f}")
        log_train(rollout_loss, grad_norm, reward_meta, avg_output_tokens, writer, train_step)
        if output_dir:
            with (Path(output_dir) / "train_metrics.jsonl").open("a") as f:
                f.write(json.dumps({"step": train_step, "loss": rollout_loss,
                    "grad_norm": float(grad_norm), "reward_mean": reward_meta["mean"],
                    "reward_std": reward_meta["std"], "avg_output_tokens": avg_output_tokens}) + "\n")
        if train_step % eval_every == 0 or train_step == n_grpo_steps:
            record_eval(train_step)


def init_policy(model_id: str, device: str, gradient_checkpointing: bool = False) -> Tuple[PreTrainedModel, AutoTokenizer]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", use_cache=False
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.to(device).train()
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GRPO or Dr.GRPO on Countdown")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--loss-type", choices=["grpo", "dr_grpo"], default="grpo")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--rollout-batch-size", type=int, default=128)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--grad-acc-steps", type=int, default=32)
    parser.add_argument("--lr", type=float, default=7e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-eval-examples", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.4)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    for name in ["max_tokens", "steps", "rollout_batch_size", "group_size", "grad_acc_steps", "eval_every", "max_model_len", "max_num_seqs"]:
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ["max_train_examples", "max_eval_examples"]:
        if getattr(args, name) is not None and getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.group_size < 2:
        parser.error("--group-size must be at least 2 for group standard deviations")
    if args.rollout_batch_size % args.group_size or args.rollout_batch_size % args.grad_acc_steps:
        parser.error("--rollout-batch-size must be divisible by --group-size and --grad-acc-steps")
    if not 0 < args.gpu_memory_utilization < 1 or args.lr <= 0:
        parser.error("GPU memory utilization must be in (0, 1), and learning rate must be positive")
    if args.max_tokens < 4 or args.max_tokens >= args.max_model_len:
        parser.error("--max-tokens must be at least 4 and leave room for prompts in --max-model-len")

    from datasets import load_dataset
    from dotenv import load_dotenv
    from torch.utils.tensorboard import SummaryWriter

    load_dotenv()
    model_id = args.model
    device = "cuda"
    seed, gpu_mem_util = args.seed, args.gpu_memory_utilization
    n_grpo_steps, rollout_batch_size = args.steps, args.rollout_batch_size
    group_size, grad_acc_steps = args.group_size, args.grad_acc_steps
    lr, clip_range, adv_eps = args.lr, 0.2, 1e-6
    temperature, min_tokens = 1.0, 4
    eval_every = args.eval_every
    loss_type, max_tokens = args.loss_type, args.max_tokens
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out_dir = Path(args.output_dir or f"output/{loss_type}_l{max_tokens}_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Initialization
    use_std_norm = loss_type == "grpo"
    policy, tokenizer = init_policy(model_id=model_id, device=device,
                                    gradient_checkpointing=args.gradient_checkpointing)
    llm = init_vllm(model_id=model_id, device=device, seed=seed, gpu_memory_utilization=gpu_mem_util,
                    max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs,
                    enforce_eager=args.enforce_eager)
    sampling_params = init_sampling_params(temperature=temperature, min_tokens=min_tokens, max_tokens=max_tokens)
    
    # Dataset
    def build_dataset(split):
        data = []
        for ex in split:
            prompt = TEMPLATE.format(numbers=ex["nums"], target=ex["target"], max_tokens=max_tokens)
            prompt = tokenizer.apply_chat_template(
                [dict(role="system", content="You are a helpful assistant."),
                dict(role="user", content=prompt)],
                add_generation_prompt=True, tokenize=False)
            data.append({"prompt": prompt,"answer": {"target": ex["target"], "numbers": ex["nums"]},})
        return data

    # Load properly split dataset
    train_data = load_dataset("justinphan3110/Countdown-Tasks-3to4", split="train")
    eval_data = load_dataset("justinphan3110/Countdown-Tasks-3to4", split="test")
    if args.max_train_examples is not None:
        train_data = train_data.select(range(min(args.max_train_examples, len(train_data))))
    if args.max_eval_examples is not None:
        eval_data = eval_data.select(range(min(args.max_eval_examples, len(eval_data))))
    if len(train_data) < rollout_batch_size // group_size:
        raise ValueError("Training subset must contain at least rollout_batch_size // group_size prompts")
    
    train_examples = build_dataset(train_data)
    eval_examples = build_dataset(eval_data)
    
    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-2, betas=(0.9, 0.95))
    scheduler = get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0)
    
    # Logging
    log_dir = str(out_dir / "tb")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    # Training
    train(
        policy=policy, tokenizer=tokenizer, llm=llm, sampling_params=sampling_params,
        train_prompts=[ex["prompt"] for ex in train_examples], train_answers=[ex["answer"] for ex in train_examples],
        eval_prompts=[ex["prompt"] for ex in eval_examples], eval_answers=[ex["answer"] for ex in eval_examples],
        optimizer=optimizer, scheduler=scheduler, n_grpo_steps=n_grpo_steps,
        rollout_batch_size=rollout_batch_size, group_size=group_size,
        gradient_accumulation_steps=grad_acc_steps, clip_range=clip_range,
        use_std_normalization=use_std_norm, advantage_eps=adv_eps, device=device,
        eval_every=eval_every, writer=writer, seed=seed, loss_type=loss_type,
        max_completion_length=max_tokens, output_dir=str(out_dir)
    )
    
    # Save model
    checkpoint_dir = out_dir / "checkpoint"
    policy.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    print(f"Saved model and tokenizer to {checkpoint_dir}")
    writer.close()

if __name__ == "__main__":
    main()
