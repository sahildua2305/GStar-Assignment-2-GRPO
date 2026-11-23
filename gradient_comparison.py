"""Run the assignment's 3-token/6-token masked-reduction gradient experiment."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

import starter


def compare(advantage_value):
    mask = torch.tensor([[True, True, True, False, False, False],
                         [True, True, True, True, True, True]])
    records = {}
    for method in ("masked_mean", "masked_mean_drgrpo"):
        # The leaf is the probability ratio, because that is what the PDF asks to differentiate.
        ratios = torch.ones((2, 6), dtype=torch.float64, requires_grad=True)
        advantages = torch.full((2, 1), advantage_value, dtype=torch.float64)
        loss_per_token, _ = starter.compute_loss(
            advantages, ratios.log(), torch.zeros_like(ratios), clip_range=0.2
        )
        numeric_mask = mask.to(dtype=ratios.dtype)
        if method == "masked_mean":
            loss = starter.masked_mean(loss_per_token, mask)
            expected = -advantage_value * numeric_mask / (2 * mask.sum(dim=1, keepdim=True))
            expected_loss = -advantage_value
        else:
            loss = starter.masked_mean_drgrpo(loss_per_token, mask, num_tokens=6)
            expected = -advantage_value * numeric_mask / (2 * 6)
            expected_loss = -advantage_value * 0.75
        loss.backward()
        torch.testing.assert_close(ratios.grad, expected, rtol=0, atol=1e-12)
        torch.testing.assert_close(loss, torch.tensor(expected_loss, dtype=torch.float64), rtol=0, atol=1e-12)
        records[method] = {
            "loss": loss.item(),
            "gradients_wrt_probability_ratio": ratios.grad.tolist(),
        }
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "eval_results" / "gradient_comparison.json")
    args = parser.parse_args()
    results = {
        "pytorch_version": torch.__version__, "device": "cpu", "dtype": "float64",
        "response_lengths": [3, 6], "batch_size": 2, "max_completion_length": 6,
        "probability_ratio": 1.0, "clip_range": 0.2,
        "note": "Advantages are fixed to isolate reduction weighting; no reward-group normalization is applied.",
        "starter_sha256": hashlib.sha256(Path(starter.__file__).read_bytes()).hexdigest(),
        "positive_advantage": compare(1.0), "negative_advantage": compare(-1.0),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    for label in ("positive_advantage", "negative_advantage"):
        for method, record in results[label].items():
            gradients = [[round(x, 6) for x in row] for row in record["gradients_wrt_probability_ratio"]]
            print(f"{label}, {method}: loss={record['loss']}, gradients={gradients}")
    print(f"Saved verified results to {args.output}")


if __name__ == "__main__":
    main()
