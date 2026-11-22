"""CPU checks for the loss, reductions, and training/evaluation integration."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

import starter


class TrainingTests(unittest.TestCase):
    def test_masked_means(self):
        values = torch.tensor([[2., 4., 100.], [8., 100., 100.]])
        mask = torch.tensor([[True, True, False], [True, False, False]])
        torch.testing.assert_close(starter.masked_mean(values, mask), torch.tensor(5.5))
        torch.testing.assert_close(starter.masked_mean_drgrpo(values, mask, 4), torch.tensor(1.75))
        mask[1] = False
        torch.testing.assert_close(starter.masked_mean(values, mask), torch.tensor(1.5))
        torch.testing.assert_close(starter.masked_mean_drgrpo(values, mask, 4), torch.tensor(.75))
        mask[:] = False
        torch.testing.assert_close(starter.masked_mean(values, mask), torch.tensor(0.))

    def test_masked_gradients(self):
        values = torch.ones((2, 3), requires_grad=True)
        mask = torch.tensor([[True, False, False], [True, True, False]])
        starter.masked_mean(values, mask).backward()
        torch.testing.assert_close(values.grad, torch.tensor([[.5, 0., 0.], [.25, .25, 0.]]))
        values.grad = None
        starter.masked_mean_drgrpo(values, mask, 4).backward()
        torch.testing.assert_close(values.grad, mask.float() / 8)

    def test_clipped_loss_both_advantage_signs(self):
        ratios = torch.tensor([[1.5, .5], [1.5, .5]])
        loss, metadata = starter.compute_loss(torch.tensor([2., -3.]), ratios.log(), torch.zeros_like(ratios), .2)
        torch.testing.assert_close(loss, torch.tensor([[-2.4, -1.], [4.5, 2.4]]))
        self.assertIsInstance(metadata, dict)

    def test_clipping_gradients(self):
        log_probs = torch.tensor([[1.5, 1.], [.5, 1.]]).log().requires_grad_()
        loss, _ = starter.compute_loss(torch.tensor([2., -3.]), log_probs, torch.zeros_like(log_probs), .2)
        loss.sum().backward()
        torch.testing.assert_close(log_probs.grad, torch.tensor([[0., -2.], [0., 3.]]))

    def test_group_advantages(self):
        for normalize in (True, False):
            advantages, rewards, metadata = starter.compute_group_normalized_advantages(
                [0., 1., 1., 1.], [{}, {}, {}, {}], lambda text, gt: text, 2, 1e-6, normalize)
            expected = torch.tensor([-.5, .5, 0., 0.])
            if normalize:
                expected[:2] /= torch.tensor([0., 1.]).std() + 1e-6
            torch.testing.assert_close(advantages, expected)
            self.assertEqual(rewards.shape, (4,))
            self.assertEqual(metadata["mean"], .75)

    def test_reward_number_multiplicities(self):
        self.assertTrue(starter._validate_numbers("65 - 7 - 7", [7, 7, 65]))
        self.assertFalse(starter._validate_numbers("7 + 7 + 7", [7, 7, 65]))
        self.assertFalse(starter._validate_numbers("2 + 2 + 4", [2, 3, 4]))
        self.assertEqual(starter.reward_fn("<answer>65-7-7</answer>", {"target": 51, "numbers": [7, 7, 65]}), 1.)

    def test_accumulation_matches_full_batch(self):
        class TinyPolicy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = torch.nn.Parameter(torch.tensor([.1, .2, -.1]))

            def forward(self, ids):
                return SimpleNamespace(logits=self.logits.expand(*ids.shape, 3))

        ids = torch.tensor([[0, 1], [1, 2]])
        labels = torch.tensor([[1, 2], [2, 0]])
        mask = torch.tensor([[True, False], [True, True]])
        advantages = torch.tensor([1., -1.])
        for loss_type in ("grpo", "dr_grpo"):
            full, accumulated = TinyPolicy(), TinyPolicy()
            starter.grpo_microbatch_step(full, ids, labels, mask, advantages, 1, .2, loss_type, 4)
            for i in range(2):
                starter.grpo_microbatch_step(accumulated, ids[i:i+1], labels[i:i+1], mask[i:i+1],
                                              advantages[i:i+1], 2, .2, loss_type, 4)
            torch.testing.assert_close(full.logits.grad, accumulated.logits.grad)
            self.assertGreater(full.logits.grad.abs().sum().item(), 0)

    def test_weight_sync_clears_cache(self):
        policy, llm = Mock(), Mock()
        policy.state_dict.return_value = {"weight": torch.tensor([1.])}
        starter.load_policy_into_vllm_instance(policy, llm)
        llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights.assert_called_once()
        llm.reset_prefix_cache.assert_called_once()

    def test_final_eval_uses_latest_weights_and_saves_metrics(self):
        policy = torch.nn.Linear(1, 1)
        optimizer = torch.optim.SGD(policy.parameters(), lr=.1)
        metrics = {"examples": [], "accuracy": 50., "mean_reward": .5, "std_reward": .1,
                   "avg_output_tokens": 2., "count_correct": 1, "count_partial": 0,
                   "count_failed": 1, "num_examples": 2}
        tokenized = {"input_ids": torch.ones((2, 2), dtype=torch.long),
                     "labels": torch.ones((2, 2), dtype=torch.long),
                     "response_mask": torch.ones((2, 2), dtype=torch.bool)}
        calls = []
        def step(*args, **kwargs):
            policy.weight.grad = torch.ones_like(policy.weight)
            return torch.tensor(1.), {}
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(starter, "load_policy_into_vllm_instance", side_effect=lambda *a: calls.append("sync")), \
             patch.object(starter, "evaluate_model", side_effect=lambda *a: (calls.append("eval"), metrics)[1]), \
             patch.object(starter, "rollout_with_vllm", return_value=(["q", "q"], ["", ""], [2, 2])), \
             patch.object(starter, "tokenize_rollouts", return_value=tokenized), \
             patch.object(starter, "grpo_microbatch_step", side_effect=step):
            starter.train(policy, None, None, None, train_prompts=["q"],
                train_answers=[{"target": 1, "numbers": [1]}], eval_prompts=["q"], eval_answers=[{}],
                optimizer=optimizer, scheduler=Mock(), n_grpo_steps=1, rollout_batch_size=2,
                group_size=2, gradient_accumulation_steps=2, clip_range=.2,
                use_std_normalization=True, advantage_eps=1e-6, device="cpu", seed=42,
                eval_every=10, output_dir=directory)
            records = [json.loads(line) for line in (Path(directory) / "eval_metrics.jsonl").read_text().splitlines()]
            self.assertEqual([record["step"] for record in records], [0, 1])
            train_record = json.loads((Path(directory) / "train_metrics.jsonl").read_text())
            self.assertEqual(train_record["loss"], 2.)  # Each mocked microbatch returns a scaled loss of 1.
        self.assertEqual(calls, ["sync", "eval", "sync", "eval"])


if __name__ == "__main__":
    unittest.main()
