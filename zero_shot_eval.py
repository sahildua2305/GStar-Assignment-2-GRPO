
import os, argparse
import datetime
import json
from pathlib import Path
import re
import uuid

os.environ["VLLM_USE_V1"] = "0"

TEMPLATE = """Using the numbers {numbers}, create an equation that equals {target}.
You can use basic arithmetic operations (+, -, *, /) and each number can only be used once.
Show your reasoning in <think> </think> tags. And return the final equation in <answer> </answer> tags. Keep your reasoning under {max_tokens} tokens.
For example, numbers = [1, 2, 3, 4] and target = 5, the answer is <answer>(1 + 2) * 3 - 4</answer>."""

def make_eval_run_name(model: str, max_tokens: int) -> str:
    """Identify the source model/checkpoint in automatic evaluation directories."""
    # Include the parent: a checkpoint's basename alone is usually just 'checkpoint'.
    source = "-".join(Path(model).parts[-2:])
    label = re.sub(r"[^A-Za-z0-9_.-]+", "-", source).strip("._-") or "model"
    # Leave room for the evaluation settings and unique suffix in a directory name.
    if len(label) > 120:
        label = f"{label[:60]}-{label[-50:]}"
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"zero-shot_{label}_l{max_tokens}_{timestamp}_{uuid.uuid4().hex[:6]}"


def is_correct(response: str, target: float, numbers: list) -> bool:
    """Check if response contains correct equation."""
    from starter import reward_fn

    return reward_fn(response, {"target": target, "numbers": numbers}) == 1.

def main():
    parser = argparse.ArgumentParser(description="Zero-shot evaluation of math reasoning")
    parser.add_argument("-m", "--model", default="Qwen/Qwen3-1.7B", help="Model ID to evaluate")
    parser.add_argument("--max_tokens", "--max-tokens", type=int, default=256, help="Maximum tokens for generation (default: 256)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-examples", type=int, default=None, help="Use a test subset for smoke checks; omit for the full test split")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    if args.max_tokens <= 0 or args.max_tokens >= args.max_model_len:
        parser.error("--max-tokens must be positive and leave room for prompts in --max-model-len")
    if args.max_examples is not None and args.max_examples <= 0:
        parser.error("--max-examples must be positive")
    if not 0 < args.gpu_memory_utilization < 1 or args.max_num_seqs <= 0:
        parser.error("GPU memory utilization must be in (0, 1), and --max-num-seqs must be positive")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from datasets import load_dataset
    from dotenv import load_dotenv
    import torch
    from starter import evaluate_model

    load_dotenv()
    
    MODEL_ID = args.model
    MAX_TOKENS = args.max_tokens
    
    print(f"Evaluating {MODEL_ID} with max_tokens={MAX_TOKENS}...")
    
    # Load model
    llm = LLM(model=MODEL_ID, dtype=torch.bfloat16, seed=args.seed,
              max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs,
              enforce_eager=args.enforce_eager, enable_prefix_caching=True,
              gpu_memory_utilization=args.gpu_memory_utilization,
              distributed_executor_backend="uni")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    sampling_params = SamplingParams(temperature=1.0, top_p=1.0, logprobs=0, max_tokens=MAX_TOKENS, stop=["</answer>"], include_stop_str_in_output=True)
    
    # Load test split from properly split dataset
    eval_raw = load_dataset("justinphan3110/Countdown-Tasks-3to4", split="test")
    if args.max_examples is not None:
        eval_raw = eval_raw.select(range(min(args.max_examples, len(eval_raw))))
    
    # Prepare prompts
    prompts = []
    targets = []
    numbers_list = []
    
    for ex in eval_raw:
        prompt_content = TEMPLATE.format(numbers=ex["nums"], target=ex["target"], max_tokens=MAX_TOKENS)
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": "You are a helpful assistant."},
             {"role": "user", "content": prompt_content}],
            add_generation_prompt=True, tokenize=False
        )
        prompts.append(prompt)
        targets.append(float(ex["target"]))
        numbers_list.append(ex["nums"])
    
    # Generate and evaluate
    print(f"Generating {len(prompts)} responses...")
    answers = [{"target": target, "numbers": numbers} for target, numbers in zip(targets, numbers_list)]
    metrics = evaluate_model(llm, sampling_params, prompts, answers)
    out_dir = Path(args.output_dir) if args.output_dir else Path("output") / make_eval_run_name(MODEL_ID, MAX_TOKENS)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)
    with (out_dir / "metrics.json").open("w") as f:
        json.dump({k: v for k, v in metrics.items() if k != "examples"}, f, indent=2)
    with (out_dir / "examples.json").open("w") as f:
        json.dump(metrics["examples"], f, indent=2)
    print(f"\nFinal Accuracy: {metrics['accuracy']:.2f}% ({metrics['count_correct']}/{metrics['num_examples']})")
    print(f"Saved results to {out_dir}")

if __name__ == "__main__":
    main()
