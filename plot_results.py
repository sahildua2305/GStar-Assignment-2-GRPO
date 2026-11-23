"""Regenerate the report figures from saved evaluation and training results.

Install requirements-plots.txt, then run: python plot_results.py
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "grpo-matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parent
CAPS = (256, 512, 1024)
METHODS = ("GRPO", "Dr.GRPO")
COLORS = {"GRPO": "#0072B2", "Dr.GRPO": "#D55E00", "Zero-shot": "#6B7280"}
MARKERS = {"GRPO": "o", "Dr.GRPO": "s", "Zero-shot": "D"}
STYLE = {256: "-", 512: "--"}
OUTCOMES = ("#16856B", "#E8AA39", "#A5ADBA")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text())


def load_evaluations(root):
    records, files, reference = {}, [], None
    for path in sorted(root.glob("*/metrics.json")):
        directory = path.parent
        config = read_json(directory / "config.json")
        metrics = read_json(path)
        examples = read_json(directory / "examples.json")
        model, cap = config["model"], config["max_tokens"]
        if model == "Qwen/Qwen3-1.7B":
            method, training = "Zero-shot", None
        else:
            match = re.fullmatch(r"train_(grpo|dr_grpo)_l(256|512)_.*", Path(model).parent.name)
            require(match is not None, f"Unrecognized checkpoint: {model}")
            method = "GRPO" if match[1] == "grpo" else "Dr.GRPO"
            training = int(match[2])
        require(config["seed"] == 42, f"Unexpected evaluation seed: {directory}")
        require(cap in CAPS, f"Unexpected generation limit: {cap}")
        n = len(examples)
        require(n == metrics["num_examples"] == 1024, f"Unexpected sample count: {directory}")
        correct = sum(abs(e["reward"] - 1.0) < 1e-8 for e in examples)
        partial = sum(abs(e["reward"] - 0.1) < 1e-8 for e in examples)
        missing = sum(abs(e["reward"]) < 1e-8 for e in examples)
        require((correct, partial, missing) == tuple(metrics[k] for k in
                ("count_correct", "count_partial", "count_failed")), f"Reward/count mismatch: {directory}")
        require(correct + partial + missing == n, f"Unknown reward: {directory}")
        require(math.isclose(metrics["accuracy"], 100 * correct / n, abs_tol=1e-10),
                f"Accuracy mismatch: {directory}")
        mean_tokens = sum(e["output_tokens"] for e in examples) / n
        require(math.isclose(mean_tokens, metrics["avg_output_tokens"], abs_tol=1e-10),
                f"Output length mismatch: {directory}")
        questions = [(e["answer"]["target"], tuple(e["answer"]["numbers"])) for e in examples]
        if reference is None:
            reference = questions
        require(questions == reference, f"Different evaluation questions: {directory}")
        at_limit_missing = sum(e["output_tokens"] == cap and
                               re.search(r"<answer>.*?</answer>", e["response"], re.DOTALL) is None
                               for e in examples)
        key = method, training, cap
        require(key not in records, f"Duplicate evaluation for {key}")
        require(read_json(directory / "status.json")["state"] == "completed", f"Incomplete run: {directory}")
        records[key] = dict(method=method, training_limit=training, eval_limit=cap,
                            num_examples=n, count_correct=correct, count_partial=partial,
                            count_missing=missing, accuracy=metrics["accuracy"],
                            mean_output_tokens=mean_tokens, at_limit_no_answer=at_limit_missing,
                            at_limit_no_answer_pct=100 * at_limit_missing / n,
                            source_directory=directory.name)
        files.extend(directory / name for name in ("config.json", "metrics.json", "examples.json", "status.json"))
    expected = {("Zero-shot", None, cap) for cap in CAPS}
    expected |= {(method, training, cap) for method in METHODS for training in (256, 512) for cap in CAPS}
    require(set(records) == expected, "Expected the 15 final checkpoint/base-model evaluations")
    return records, files


def load_training(root):
    records, files = {}, []
    for path in sorted(root.glob("*/train_metrics.jsonl")):
        directory = path.parent
        config = read_json(directory / "config.json")
        method = "GRPO" if config["loss_type"] == "grpo" else "Dr.GRPO"
        key = method, config["max_tokens"]
        require(key not in records, f"Duplicate training run for {key}")
        train = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        evaluation = [json.loads(line) for line in (directory / "eval_metrics.jsonl").read_text().splitlines() if line.strip()]
        require([r["step"] for r in train] == list(range(1, config["steps"] + 1)), f"Missing training steps: {directory}")
        expected_steps = [0] + list(range(config["eval_every"], config["steps"] + 1, config["eval_every"]))
        if expected_steps[-1] != config["steps"]:
            expected_steps.append(config["steps"])
        require([r["step"] for r in evaluation] == expected_steps, f"Missing evaluation steps: {directory}")
        for row in evaluation:
            require(row["num_examples"] == 1024 and math.isclose(row["accuracy"],
                    100 * row["count_correct"] / row["num_examples"], abs_tol=1e-10), f"Bad periodic metrics: {directory}")
        require(read_json(directory / "status.json")["state"] == "completed", f"Incomplete training: {directory}")
        records[key] = dict(config=config, train=train, evaluation=evaluation, source_directory=directory.name)
        files.extend(directory / name for name in ("config.json", "train_metrics.jsonl", "eval_metrics.jsonl", "status.json"))
    require(set(records) == {(m, t) for m in METHODS for t in (256, 512)}, "Expected all four training runs")
    return records, files


def setup_style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#CBD0D8", "axes.labelcolor": "#303742",
                         "text.color": "#202731", "xtick.color": "#4B5563", "ytick.color": "#4B5563",
                         "grid.color": "#E4E7EC", "grid.linewidth": .7,
                         "axes.axisbelow": True, "lines.linewidth": 2.2, "lines.markersize": 6,
                         "savefig.facecolor": "white", "pdf.fonttype": 42, "svg.fonttype": "none"})


def budget_axis(ax):
    ax.set_xscale("log", base=2)
    ax.set_xticks(CAPS, ["256", "512", "1,024"])
    ax.set_xlim(225, 1160)
    ax.set_xlabel("Evaluation generation limit (tokens)")
    ax.grid(axis="y")


def finish(fig, output, name, title, footnote, legend=None, top=.76, bottom=.17):
    fig.suptitle(title, x=.065, y=.98, ha="left", fontsize=17, fontweight="bold")
    if legend:
        fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(.058, .91),
                   ncol=len(legend), frameon=False, handlelength=2.5, columnspacing=2)
    fig.text(.065, .025, footnote, fontsize=9, color="#606A78", va="bottom")
    fig.subplots_adjust(left=.075, right=.975, top=top, bottom=bottom, wspace=.25, hspace=.50)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output / f"{name}.{extension}", dpi=210, bbox_inches="tight")
    plt.close(fig)


def method_legend(include_base=False):
    names = (["Zero-shot"] if include_base else []) + list(METHODS)
    return [Line2D([], [], color=COLORS[m], marker=MARKERS[m],
                   linestyle=":" if m == "Zero-shot" else "-", label=m) for m in names]


def accuracy_plot(data, output):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.7), sharey=True)
    for ax, training in zip(axes, (256, 512)):
        ax.set_title(f"Training generation limit: {training} tokens", loc="left", pad=16)
        for method in ("Zero-shot", *METHODS):
            values = [data[(method, None if method == "Zero-shot" else training, cap)]["accuracy"] for cap in CAPS]
            ax.plot(CAPS, values, color=COLORS[method], marker=MARKERS[method],
                    linestyle=":" if method == "Zero-shot" else STYLE[training])
            for cap, value in zip(CAPS, values):
                if method == "Zero-shot":
                    offset = 9 if value < 5 else -17
                else:
                    other = "Dr.GRPO" if method == "GRPO" else "GRPO"
                    offset = 9 if value > data[(other, training, cap)]["accuracy"] else -18
                ax.annotate(f"{value:.2f}", (cap, value), xytext=(0, offset), textcoords="offset points",
                            ha="center", color=COLORS[method], fontsize=10)
        budget_axis(ax)
        ax.set_ylim(0, 78)
        ax.set_yticks(range(0, 81, 20))
    axes[0].set_ylabel("Task accuracy (%)")
    finish(fig, output, "accuracy_vs_budget", "Task accuracy improves with the evaluation budget",
           "Each point: 1,024 shared test questions · one sampled response per question · seed 42\n"
           "Training/evaluation limits refer to generated tokens; the model context limit is 2,048 tokens.",
           method_legend(True))


def outcomes_plot(data, output):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.7), sharex=True)
    for ax, training in zip(axes, (256, 512)):
        positions, labels = [], []
        for i, cap in enumerate(CAPS):
            for j, method in enumerate(METHODS):
                y = i * 2.1 + j * .78
                positions.append(y)
                labels.append(f"{cap:,} · {method}")
                record = data[(method, training, cap)]
                portions = [100 * record[k] / record["num_examples"] for k in
                            ("count_correct", "count_partial", "count_missing")]
                left = 0
                for value, color in zip(portions, OUTCOMES):
                    ax.barh(y, value, left=left, height=.57, color=color, edgecolor="white", linewidth=.7)
                    if value >= 5:
                        ax.text(left + value/2, y, f"{value:.1f}%", ha="center", va="center",
                                fontsize=10, color="white" if color == OUTCOMES[0] else "#202731")
                    left += value
        ax.set_yticks(positions, labels)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xticks(range(0, 101, 25))
        ax.set_xlabel("Share of the 1,024 test responses (%)")
        ax.set_title(f"Training generation limit: {training} tokens", loc="left", pad=15)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    legend = [Patch(facecolor=color, label=label) for color, label in zip(OUTCOMES,
              ("Correct (reward 1.0)", "Completed, invalid/incorrect (0.1)", "No completed answer (0.0)"))]
    finish(fig, output, "answer_outcomes", "Separate missing answers from rejected completed answers",
           "Row labels: evaluation token limit · method. Thin segments are left unlabeled for readability.\n"
           "Partial reward can reflect invalid format, number use, or arithmetic; a missing answer has no completed answer tag.",
           legend, top=.80, bottom=.15)


def length_plot(data, output):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.6))
    for row, method in enumerate(METHODS):
        for col, field in enumerate(("mean_output_tokens", "at_limit_no_answer_pct")):
            ax = axes[row, col]
            for training in (256, 512):
                values = [data[(method, training, cap)][field] for cap in CAPS]
                ax.plot(CAPS, values, color=COLORS[method], linestyle=STYLE[training],
                        marker="o" if training == 256 else "^", label=f"Trained at {training}")
            budget_axis(ax)
            ax.set_title(f"{method} · {'response length' if col == 0 else 'unfinished at the limit'}", loc="left", pad=12)
            ax.set_ylim((0, 620) if col == 0 else (0, 70))
            ax.set_ylabel("Mean generated tokens" if col == 0 else "At limit without an answer (%)")
            ax.legend(frameon=False, loc="upper left" if col == 0 else "upper right", fontsize=10)
    finish(fig, output, "length_and_budget", "Longer-budget training changes response length and completion",
           "At limit without an answer: output length equals the generation limit and no completed answer tag is present.\n"
           "This is consistent with budget exhaustion; the saved results do not record the generation finish reason.",
           top=.88, bottom=.14)


def difference_plot(data, output):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), sharey=True)
    for ax, training in zip(axes, (256, 512)):
        delta = [data[("Dr.GRPO", training, cap)]["accuracy"] - data[("GRPO", training, cap)]["accuracy"] for cap in CAPS]
        counts = [data[("Dr.GRPO", training, cap)]["count_correct"] - data[("GRPO", training, cap)]["count_correct"] for cap in CAPS]
        ax.bar(range(3), delta, width=.54, color=["#16856B" if d >= 0 else "#B95A58" for d in delta], zorder=3)
        ax.axhline(0, color="#535E6E", linewidth=1)
        for i, value in enumerate(delta):
            ax.annotate(f"{counts[i]:+d} correct", (i, value), xytext=(0, 8 if value >= 0 else -17),
                        textcoords="offset points", ha="center", fontsize=11)
        ax.set_xticks(range(3), ["256", "512", "1,024"])
        ax.set_xlabel("Evaluation generation limit (tokens)")
        ax.set_ylim(-5.5, 3)
        ax.grid(axis="y")
        ax.set_title(f"Training generation limit: {training} tokens", loc="left", pad=14)
    axes[0].set_ylabel("Dr.GRPO − GRPO accuracy (percentage points)")
    finish(fig, output, "dr_grpo_minus_grpo", "Neither method wins at every training and evaluation budget",
           "Bar labels give the difference in correct answers out of 1,024 paired questions.\n"
           "These are observed single-seed differences, without a claim of statistical significance.",
           [Patch(facecolor="#16856B", label="Dr.GRPO higher"), Patch(facecolor="#B95A58", label="GRPO higher")])


def trailing_mean(values, window=5):
    return [sum(values[max(0, i-window+1):i+1]) / min(i+1, window) for i in range(len(values))]


def learning_plot(training_data, output):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.6))
    for row, training in enumerate((256, 512)):
        for method in METHODS:
            data = training_data[(method, training)]
            steps = [r["step"] for r in data["train"]]
            rewards = [r["reward_mean"] for r in data["train"]]
            axes[row, 0].plot(steps, rewards, color=COLORS[method], alpha=.22, linewidth=.8)
            axes[row, 0].plot(steps, trailing_mean(rewards), color=COLORS[method])
            axes[row, 1].plot([r["step"] for r in data["evaluation"]],
                              [r["accuracy"] for r in data["evaluation"]],
                              color=COLORS[method], marker=MARKERS[method])
        for col in (0, 1):
            ax = axes[row, col]
            ax.set_xlim(-1, 82)
            ax.set_xticks(range(0, 81, 20))
            ax.set_xlabel("Optimizer step")
            ax.grid(axis="y")
            ax.set_title(f"{training}-token training · {'rollout reward' if col == 0 else 'periodic test accuracy'}", loc="left", pad=12)
            ax.set_ylim((0, 1) if col == 0 else (0, 80))
            ax.set_ylabel("Mean reward (0, 0.1, or 1 per response)" if col == 0 else "Test accuracy (%)")
    finish(fig, output, "learning_curves", "Learning curves from the four saved training runs",
           "Reward: trailing 5-step mean; faint lines show raw rollout rewards. Test evaluation uses each run's training token limit.\n"
           "Periodic evaluations are sampled during training; the main accuracy figure uses separate checkpoint evaluation runs.",
           method_legend(), top=.82, bottom=.14)


def write_csv(path, records):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "eval_results")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evaluations, eval_files = load_evaluations(args.results_dir)
    training, training_files = load_training(args.results_dir)
    setup_style()
    accuracy_plot(evaluations, args.output_dir)
    outcomes_plot(evaluations, args.output_dir)
    length_plot(evaluations, args.output_dir)
    difference_plot(evaluations, args.output_dir)
    learning_plot(training, args.output_dir)
    write_csv(args.output_dir / "evaluation_summary.csv", list(evaluations.values()))
    train_rows, eval_rows = [], []
    for (method, cap), data in training.items():
        smoothed = trailing_mean([r["reward_mean"] for r in data["train"]])
        for row, value in zip(data["train"], smoothed):
            train_rows.append(dict(method=method, training_limit=cap, **row, reward_trailing_5=value))
        for row in data["evaluation"]:
            eval_rows.append(dict(method=method, training_limit=cap, **row))
    write_csv(args.output_dir / "training_curves.csv", train_rows)
    write_csv(args.output_dir / "periodic_evaluation.csv", eval_rows)
    names = ("accuracy_vs_budget", "answer_outcomes", "length_and_budget", "dr_grpo_minus_grpo", "learning_curves")
    manifest = dict(matplotlib_version=matplotlib.__version__, numpy_version=np.__version__,
                    evaluation_points=len(evaluations), training_runs=len(training),
                    figures=[f"{name}.{ext}" for name in names for ext in ("png", "pdf", "svg")],
                    source_sha256={str(p.relative_to(args.results_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in eval_files + training_files})
    (args.output_dir / "plot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Verified {len(evaluations)} final evaluations, 15,360 example rewards, and {len(training)} training runs.")
    print(f"Saved five figures as PNG/PDF/SVG and three data CSVs to {args.output_dir}")


if __name__ == "__main__":
    main()
