"""Audit a supplied training-log ZIP and rebuild the final report and figures.

Usage: python scripts/build_final_writeup.py [--archive /path/to/logs.zip]
Without --archive, uses the checked-in, source-attributed analysis snapshot.
"""

import argparse
import base64
import hashlib
import json
import math
import os
import re
import statistics as stats
import zipfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/qwen-writeup-matplotlib")
import matplotlib

matplotlib.use("Agg")
import markdown
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "docs/final-report"
DATA = DEST / "data"
FIG = DEST / "figures"
ORDER = [
    "base",
    "base_rl",
    "kimi_sft",
    "rejection_sft",
    "distill_sft",
    "rejection_rl",
    "distill_rl",
    "instruct",
    "instruct_rl",
]
LABEL = dict(
    zip(ORDER, ["B0", "B-RL", "B-Kimi", "B-RS", "B-Mix", "B-RS-RL", "B-Mix-RL", "I0", "I-RL"])
)
RL = ["base_rl", "rejection_rl", "distill_rl", "instruct_rl"]
COLORS = {
    "base": "#596579",
    "base_rl": "#2875B9",
    "kimi_sft": "#BD4F78",
    "rejection_sft": "#299A86",
    "distill_sft": "#D18A26",
    "rejection_rl": "#16816E",
    "distill_rl": "#C37C1C",
    "instruct": "#282F3B",
    "instruct_rl": "#7957A9",
}
PARENTS = {
    "base_rl": "base",
    "kimi_sft": "base",
    "rejection_sft": "base",
    "distill_sft": "base",
    "rejection_rl": "rejection_sft",
    "distill_rl": "distill_sft",
    "instruct_rl": "instruct",
}
RESULT_FILES = dict(
    zip(
        ORDER,
        [
            "qwen_base.json",
            "qwen_base_rl_step300.json",
            "sft_qwen25_3b_base_s1_kimi_epoch5.json",
            "sft_qwen25_3b_base_self_rft_epoch4.json",
            "sft_qwen25_3b_base_7b_distill_epoch5.json",
            "qwen_rejection_rl_step300.json",
            "qwen_distill_rl_step300.json",
            "qwen_instruct.json",
            "qwen_instruct_rl_step300.json",
        ],
    )
)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def audit_archive(path):
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None, "ZIP CRC failure"
        manifest = json.loads(z.read("collection_manifest.json"))
        assert manifest["status"] == "complete"
        for entry in manifest["files"]:
            raw = z.read(entry["archive_path"])
            assert len(raw) == entry["bytes"]
            assert hashlib.sha256(raw).hexdigest() == entry["sha256"]

        def read(name):
            return json.loads(z.read(name))

        def one(prefix, suffix):
            matches = [n for n in z.namelist() if n.startswith(prefix) and n.endswith(suffix)]
            assert len(matches) == 1, matches
            return matches[0]

        result = {
            "source_zip": Path(path).name,
            "zip_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "verified_files": len(manifest["files"]),
            "runs": {},
            "math": {},
            "epochs": {},
        }
        for run in manifest["runs"]:
            prefix = "drive/" + run["root"].split("/MyDrive/")[1] + "/logs/"
            step_path = one(prefix, "/steps.jsonl")
            rows = [json.loads(line) for line in z.read(step_path).splitlines() if line.strip()]
            assert [r["step"] for r in rows] == list(range(run["step_min"], run["step_max"] + 1))
            assert all(
                math.isfinite(r[k]) for r in rows for k in ["loss", "learning_rate", "grad_norm"]
            )
            mf = read(one(prefix, "/run_manifest.json"))
            report_path = step_path.rsplit("/", 1)[0] + "/data_report.json"
            report = read(report_path)
            report = {
                k: v
                for k, v in report.items()
                if k not in {"kept_ids", "dropped", "dataset", "data"}
            }
            if run["kind"] == "sft":
                rows = [
                    {k: r[k] for k in ["step", "epoch", "loss", "grad_norm", "learning_rate"]}
                    for r in rows
                ]
            result["runs"][run["key"]] = {
                "kind": run["kind"],
                "rows": rows,
                "manifest": mf,
                "data_report": report,
                "source_path": step_path,
            }
        for key, filename in RESULT_FILES.items():
            p = one("drive/", "/" + filename)
            item = read(p)
            assert item["mode"] == "full" and item["config"]["temperature"] == 0
            assert item["config"]["max_new_tokens"] == 20480
            result["math"][key] = {"source_path": p, "metrics": item["metrics"]}
            for name, count in [("amc23", 40), ("math_500", 500)]:
                assert item["metrics"][name]["problems"] == count
                assert item["metrics"][name]["responses"] == count
        for key in ["rejection_sft", "distill_sft", "distill_sft_epochs6_7"]:
            prefix = result["runs"][key]["source_path"].split("/logs/")[0]
            result["epochs"][key] = read(one(prefix + "/", "/epoch_summary.json"))["rows"]
        dump(DATA / "analysis_snapshot.json", result)
    return result


def merge_rl(data):
    series = {key: data["runs"][key]["rows"] for key in RL}
    assert data["runs"]["instruct_rl"]["manifest"]["recovery"]["from_step"] == 40
    series["instruct_rl"] = [
        r for r in data["runs"]["instruct_rl_original"]["rows"] if r["step"] <= 40
    ] + series["instruct_rl"]
    for rows in series.values():
        assert [r["step"] for r in rows] == list(range(1, 301))
        assert len([r for r in rows if not r["cached_rollout"]]) == 150
        for i, r in enumerate(rows):
            if r["cached_rollout"]:
                assert r["rollout_first_update"] == rows[i - 1]["rollout_first_update"]
                assert r["generated_responses"] == rows[i - 1]["generated_responses"]
    return series


def moving(values, width):
    return [stats.mean(values[max(0, i - width + 1) : i + 1]) for i in range(len(values))]


def save(fig, name):
    for ext in ["png", "svg"]:
        fig.savefig(FIG / f"{name}.{ext}", dpi=190, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def style(ax, title, ylabel, xlabel="Optimizer update"):
    ax.set_title(title, loc="left", fontweight="bold", pad=12)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.grid(alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)


def box(ax, xy, text, color="#EAF1F7", size=(0.25, 0.15)):
    x, y = xy
    w, h = size
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.012",
            facecolor=color,
            edgecolor="#BBC8D5",
            lw=1,
        )
    )
    ax.text(x, y, text, ha="center", va="center", fontsize=10)


def arrow(ax, start, end, dashed=False):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            lw=1.3,
            color="#77869A",
            linestyle="--" if dashed else "-",
        )
    )


def diagrams():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    box(ax, (0.16, 0.8), "MATH train + NuminaMath\n+ math-only s1K")
    box(ax, (0.5, 0.8), "Filter • deduplicate\nremove eval / RL overlap")
    box(ax, (0.84, 0.8), "27,001-problem pool*")
    arrow(ax, (0.3, 0.8), (0.36, 0.8))
    arrow(ax, (0.64, 0.8), (0.7, 0.8))
    box(ax, (0.25, 0.45), "Qwen2.5-3B base\n8 candidate solutions / problem")
    box(ax, (0.73, 0.45), "Qwen2.5-Math-7B-Instruct\n3B-hard problems (0–2 / 8)")
    arrow(ax, (0.84, 0.71), (0.75, 0.54))
    arrow(ax, (0.78, 0.71), (0.25, 0.54))
    box(
        ax,
        (0.25, 0.15),
        "Correctness + cleanup + caps\n28,267 pairs / 12,863 problems\n3B rejection SFT",
        "#E8F4EF",
        (0.34, 0.18),
    )
    box(
        ax,
        (0.73, 0.15),
        "3B-first + 7B supplementation\n37,691 pairs / 14,853 problems\nMixed 3B/7B SFT",
        "#FCF1DB",
        (0.34, 0.18),
    )
    arrow(ax, (0.25, 0.36), (0.25, 0.25))
    arrow(ax, (0.73, 0.36), (0.73, 0.25))
    arrow(ax, (0.39, 0.43), (0.57, 0.26))
    fig.suptitle(
        "1  |  Construction of the two larger SFT datasets", x=0.05, ha="left", fontweight="bold"
    )
    fig.text(
        0.05,
        -0.02,
        "*Pool size and generation recipe: project history. Final training counts: saved data reports.\nKimi-style s1 targets form a separate branch; see the text.",
        fontsize=9,
        color="#526074",
    )
    save(fig, "01_data_pipeline")
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    box(ax, (0.12, 0.6), "B0\nQwen2.5-3B", size=(0.18, 0.13))
    for y, text, color in [
        (0.88, "B-RL\n300 RL updates", "#EAF1F7"),
        (0.67, "B-Kimi\nSFT epoch 5", "#F7EAF0"),
        (0.46, "B-RS\nSFT epoch 4", "#E8F4EF"),
        (0.25, "B-Mix\nSFT epoch 5", "#FCF1DB"),
    ]:
        box(ax, (0.48, y), text, color, (0.22, 0.12))
        arrow(ax, (0.22, 0.6), (0.355, y))
    box(ax, (0.82, 0.46), "B-RS-RL\n+ 300 RL updates", "#E8F4EF", (0.23, 0.12))
    arrow(ax, (0.60, 0.46), (0.69, 0.46))
    box(ax, (0.82, 0.25), "B-Mix-RL\n+ 300 RL updates", "#FCF1DB", (0.23, 0.12))
    arrow(ax, (0.60, 0.25), (0.69, 0.25))
    box(
        ax,
        (0.8, 0.08),
        "Epochs 6–7: diagnostic extension\nnot the RL starting checkpoint",
        "#F3F4F6",
        (0.34, 0.09),
    )
    arrow(ax, (0.58, 0.20), (0.63, 0.1), dashed=True)
    box(ax, (0.12, 0.07), "I0\nQwen Instruct", size=(0.18, 0.10))
    box(ax, (0.40, 0.07), "I-RL\n300 RL updates", "#EEEAF7", (0.17, 0.10))
    arrow(ax, (0.22, 0.07), (0.30, 0.07))
    fig.suptitle(
        "2  |  Nine evaluated variants and their immediate parents",
        x=0.05,
        ha="left",
        fontweight="bold",
    )
    save(fig, "02_experiment_tree")


def sft_fig(data):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    groups = {
        "rejection_sft": data["runs"]["rejection_sft"]["rows"],
        "distill_sft": data["runs"]["distill_sft"]["rows"]
        + data["runs"]["distill_sft_epochs6_7"]["rows"],
    }
    for key, rows in groups.items():
        epochs = sorted({math.ceil(r["epoch"] - 1e-8) for r in rows})
        losses = [
            stats.mean(r["loss"] for r in rows if math.ceil(r["epoch"] - 1e-8) == e) for e in epochs
        ]
        axs[0, 0].plot(epochs, losses, "-o", color=COLORS[key], label=LABEL[key])
        summaries = data["epochs"][key] + (
            data["epochs"]["distill_sft_epochs6_7"] if key == "distill_sft" else []
        )
        for ax, bench in [(axs[1, 0], "AMC 2023"), (axs[1, 1], "MATH-500")]:
            vals = [r for r in summaries if r["Benchmark"] == bench]
            ax.plot(
                [r["Epoch"] for r in vals],
                [r["Accuracy (%)"] for r in vals],
                "-o",
                color=COLORS[key],
                label=LABEL[key],
            )
            chosen = next(r for r in vals if r["Epoch"] == (4 if key == "rejection_sft" else 5))
            ax.scatter(
                chosen["Epoch"],
                chosen["Accuracy (%)"],
                marker="*",
                s=210,
                color=COLORS[key],
                edgecolor="white",
                zorder=5,
            )
    kr = data["runs"]["kimi_sft"]["rows"]
    axs[0, 1].plot(
        range(1, 6),
        [
            stats.mean(r["loss"] for r in kr if math.ceil(r["epoch"] - 1e-8) == e)
            for e in range(1, 6)
        ],
        "-o",
        color=COLORS["kimi_sft"],
    )
    for ax, title, ylabel in zip(
        axs.flat,
        [
            "A  Rejection / mixed-data SFT",
            "B  Kimi-style SFT (separate scale)",
            "C  AMC 2023 by checkpoint",
            "D  MATH-500 by checkpoint",
        ],
        ["Mean logged training loss", "Mean logged training loss", "Accuracy (%)", "Accuracy (%)"],
    ):
        style(ax, title, ylabel, "Epoch")
        ax.set_xticks(range(1, 8) if ax is not axs[0, 1] else range(1, 6))
        if ax is not axs[0, 1]:
            ax.axvline(5.5, ls="--", color="#888", alpha=0.7)
            ax.legend(frameon=False)
    fig.suptitle(
        "4  |  Loss decreases; checkpoint accuracy is not monotonic\nDashed line: lower-LR continuation. Stars: selected checkpoints.",
        fontweight="bold",
        fontsize=14,
    )
    save(fig, "04_sft_trajectories")


def rl_fig(series):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for key, rows in series.items():
        fresh = [r for r in rows if not r["cached_rollout"]]
        for ax, metric in [(axs[0, 0], "reward"), (axs[0, 1], "completions/mean_length")]:
            values = [r["policy_metrics"][metric] for r in fresh]
            x = [r["step"] for r in fresh]
            ax.plot(x, values, color=COLORS[key], alpha=0.12, lw=0.7)
            ax.plot(x, moving(values, 10), color=COLORS[key], label=LABEL[key], lw=2)
        x = [r["step"] for r in rows]
        values = [r["policy_metrics"]["entropy"] for r in rows]
        axs[1, 0].plot(x, values, color=COLORS[key], alpha=0.12, lw=0.7)
        axs[1, 0].plot(x, moving(values, 20), color=COLORS[key], label=LABEL[key], lw=2)
        x = [r["step"] for r in fresh]
        raw = [100 * r["correct_responses"] / r["generated_responses"] for r in fresh]
        smooth = [
            100
            * sum(r["correct_responses"] for r in fresh[max(0, i - 9) : i + 1])
            / sum(r["generated_responses"] for r in fresh[max(0, i - 9) : i + 1])
            for i in range(len(fresh))
        ]
        axs[1, 1].plot(x, raw, color=COLORS[key], alpha=0.12, lw=0.7)
        axs[1, 1].plot(x, smooth, color=COLORS[key], label=LABEL[key], lw=2)
    for ax, title, ylabel in zip(
        axs.flat,
        [
            "A  Logged reward: retained mixed groups",
            "B  Logged completion length: retained groups",
            "C  Token entropy",
            "D  Correctness across generated candidates",
        ],
        ["Mean shaped reward", "Tokens", "Entropy (nats)", "Correct candidate responses (%)"],
    ):
        style(ax, title, ylabel)
        ax.set_xlim(1, 300)
        ax.axvline(40, ls=":", color=COLORS["instruct_rl"], alpha=0.55)
    axs[0, 0].legend(frameon=False, ncol=2)
    fig.suptitle(
        "5  |  DAPO dynamics across four starting checkpoints\nFaint: raw logs. Solid: trailing 10 rollouts / 20 updates. Dotted: I-RL recovery.",
        fontsize=13,
        fontweight="bold",
    )
    save(fig, "05_rl_dynamics")


def metrics(data):
    transfer = json.loads((DATA / "reported_transfer.json").read_text())
    out = {}
    for key in ORDER:
        out[key] = dict(zip(transfer["columns"], transfer["rows"][key]))
        for bench, label in [("amc23", "amc"), ("math_500", "math")]:
            metric = data["math"][key]["metrics"][bench]
            out[key][label] = 100 * metric["avg@1"]
            out[key][label + "_tokens"] = metric["response_length_tokens"]["all"]
    return out


def result_figs(values, series):
    transitions = list(PARENTS)
    columns = ["amc", "math", "arc", "mmlu_nonmath", "bbh", "ifeval_prompt_strict"]
    array = np.array([[values[k][c] - values[PARENTS[k]][c] for c in columns] for k in transitions])
    fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
    img = ax.imshow(array, cmap="RdBu", vmin=-20, vmax=20, aspect="auto")
    ax.set_xticks(
        range(6),
        [
            "AMC\n2023",
            "MATH-500",
            "ARC-C",
            "MMLU-Pro*\nnon-math",
            "BBH*\nnon-math",
            "IFEval\nprompt strict",
        ],
    )
    ax.set_yticks(range(7), [LABEL[PARENTS[k]] + " → " + LABEL[k] for k in transitions])
    for i in range(7):
        for j in range(6):
            ax.text(
                j,
                i,
                f"{array[i, j]:+.2f}",
                ha="center",
                va="center",
                color="white" if abs(array[i, j]) > 12 else "#1B2638",
                fontsize=11,
            )
    fig.colorbar(img, ax=ax, label="Change from immediate parent (percentage points)", shrink=0.85)
    ax.set_title(
        "3  |  Training gains depend on both method and benchmark",
        loc="left",
        fontweight="bold",
        pad=18,
    )
    save(fig, "03_transfer_deltas")
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.7), layout="constrained")
    offsets = {
        "amc": {
            "base": (-32, 8),
            "base_rl": (6, 9),
            "kimi_sft": (-38, 10),
            "rejection_sft": (8, -16),
            "distill_sft": (-55, 9),
            "rejection_rl": (7, -15),
            "distill_rl": (7, 10),
            "instruct": (7, 8),
            "instruct_rl": (8, -14),
        },
        "math": {
            "base": (8, -5),
            "base_rl": (-47, -14),
            "kimi_sft": (-43, 10),
            "rejection_sft": (-44, -16),
            "distill_sft": (8, -5),
            "rejection_rl": (8, 10),
            "distill_rl": (7, 10),
            "instruct": (8, 9),
            "instruct_rl": (-43, -15),
        },
    }
    for ax, bench, title in zip(axs, ["amc", "math"], ["AMC 2023", "MATH-500"]):
        for child, parent in PARENTS.items():
            ax.annotate(
                "",
                xy=(values[child][bench + "_tokens"], values[child][bench]),
                xytext=(values[parent][bench + "_tokens"], values[parent][bench]),
                arrowprops={"arrowstyle": "->", "color": "#B5BDC6", "lw": 1.1},
                zorder=1,
            )
        for k in ORDER:
            ax.scatter(
                values[k][bench + "_tokens"],
                values[k][bench],
                s=65,
                color=COLORS[k],
                marker="s" if k in ["base", "instruct"] else "o",
                zorder=3,
            )
            ax.annotate(
                LABEL[k],
                (values[k][bench + "_tokens"], values[k][bench]),
                xytext=offsets[bench][k],
                textcoords="offset points",
                fontsize=9,
            )
        ax.set_xscale("log")
        style(ax, title, "Accuracy (%)", "Mean generated tokens (log scale)")
        ax.margins(x=0.20, y=0.15)
    fig.suptitle(
        "7  |  Longer outputs do not consistently give higher accuracy\nArrows connect each starting model to its trained variant.",
        fontweight="bold",
        fontsize=13,
    )
    save(fig, "07_accuracy_length")
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.7), layout="constrained")
    y = np.arange(4)
    for i, k in enumerate(RL):
        rows = series[k]
        ends = [
            stats.mean(
                r["policy_metrics"]["reward"] for r in win if "reward" in r["policy_metrics"]
            )
            for win in [rows[:20], rows[-20:]]
        ]
        axs[0].plot(ends, [i, i], color=COLORS[k], lw=3)
        axs[0].scatter(ends, [i, i], color=COLORS[k], s=[25, 65])
    axs[0].set_yticks(y, [LABEL[k] for k in RL])
    style(axs[0], "A  Early → late training", "", "Logged reward (first / last 20 updates)")
    for ax, cols, names in [
        (axs[1], ["amc", "math"], ["AMC", "MATH"]),
        (
            axs[2],
            ["arc", "mmlu_nonmath", "bbh", "ifeval_prompt_strict"],
            ["ARC", "MMLU*", "BBH*", "IFEval"],
        ),
    ]:
        for j, (col, name) in enumerate(zip(cols, names)):
            vals = [values[k][col] - values[PARENTS[k]][col] for k in RL]
            ax.scatter(
                vals,
                y + (j - (len(cols) - 1) / 2) * 0.12,
                label=name,
                s=45,
                marker=["o", "s", "^", "D"][j],
            )
        ax.set_yticks(y, [])
        ax.axvline(0, color="#999", lw=1)
        ax.legend(
            frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.20)
        )
        style(
            ax,
            "B  Math evaluation" if ax is axs[1] else "C  Transfer evaluation",
            "",
            "Change from parent (pp)",
        )
    for ax in axs:
        ax.set_ylim(3.5, -0.5)
    fig.suptitle(
        "6  |  Improved training reward is not a guarantee of evaluation gains", fontweight="bold"
    )
    save(fig, "06_reward_vs_results")
    fig, axs = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    for k, rows in series.items():
        x = [r["step"] for r in rows]
        for ax, field in [
            (axs[0, 0], "loss"),
            (axs[0, 1], "grad_norm"),
            (axs[1, 0], "learning_rate"),
        ]:
            vals = [r[field] for r in rows]
            ax.plot(x, vals, color=COLORS[k], alpha=0.14, lw=0.6)
            ax.plot(x, moving(vals, 20), color=COLORS[k], label=LABEL[k])
        axs[1, 1].plot(
            x,
            moving([r["policy_metrics"]["clip_ratio/region_mean"] for r in rows], 20),
            color=COLORS[k],
        )
    for ax, title, label in zip(
        axs.flat,
        [
            "Logged policy loss",
            "Gradient norm (logged)",
            "Learning rate",
            "Policy-ratio clipping fraction",
        ],
        ["Loss", "Norm", "Learning rate", "Fraction"],
    ):
        style(ax, title, label)
    axs[0, 0].legend(frameon=False, ncol=2)
    fig.suptitle(
        "Appendix  |  RL optimizer diagnostics (20-update trailing means)", fontweight="bold"
    )
    save(fig, "08_rl_diagnostics")


def table(headers, rows):
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        + ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    )


def tables(data, values, series):
    result = {}
    result["MATH"] = table(
        [
            "Model",
            "AMC correct / 40",
            "AMC %",
            "AMC tokens",
            "MATH correct / 500",
            "MATH %",
            "MATH tokens",
        ],
        [
            [
                LABEL[k],
                round(values[k]["amc"] * 0.4),
                f"{values[k]['amc']:.1f}",
                f"{values[k]['amc_tokens']:.1f}",
                round(values[k]["math"] * 5),
                f"{values[k]['math']:.1f}",
                f"{values[k]['math_tokens']:.1f}",
            ]
            for k in ORDER
        ],
    )
    result["TRANSFER"] = table(
        [
            "Model",
            "ARC %",
            "MMLU overall %",
            "MMLU non-math %",
            "BBH overall %",
            "BBH macro %",
            "IFEval strict prompt %",
        ],
        [
            [LABEL[k]]
            + [
                f"{values[k][c]:.2f}"
                for c in ["arc", "mmlu", "mmlu_nonmath", "bbh", "bbh_macro", "ifeval_prompt_strict"]
            ]
            for k in ORDER
        ],
    )
    result["IFEVAL"] = table(
        [
            "Model",
            "Prompt strict %",
            "Instruction strict %",
            "Prompt loose %",
            "Instruction loose %",
        ],
        [
            [LABEL[k]]
            + [
                f"{values[k][c]:.2f}"
                for c in [
                    "ifeval_prompt_strict",
                    "ifeval_instruction_strict",
                    "ifeval_prompt_loose",
                    "ifeval_instruction_loose",
                ]
            ]
            for k in ORDER
        ],
    )
    rows = []
    for k in ["rejection_sft", "distill_sft"]:
        steps = data["runs"][k]["rows"] + (
            data["runs"]["distill_sft_epochs6_7"]["rows"] if k == "distill_sft" else []
        )
        ev = data["epochs"][k] + (
            data["epochs"]["distill_sft_epochs6_7"] if k == "distill_sft" else []
        )
        for epoch in sorted({r["Epoch"] for r in ev}):

            def get(benchmark):
                return next(
                    r["Accuracy (%)"]
                    for r in ev
                    if r["Epoch"] == epoch and r["Benchmark"] == benchmark
                )

            loss = stats.mean(r["loss"] for r in steps if math.ceil(r["epoch"] - 1e-8) == epoch)
            rows.append(
                [LABEL[k], epoch, f"{loss:.4f}", f"{get('AMC 2023'):.1f}", f"{get('MATH-500'):.1f}"]
            )
    result["EPOCHS"] = table(["Run", "Epoch", "Mean logged loss", "AMC %", "MATH %"], rows)
    rows = []
    for k, steps in series.items():
        means = []
        for window in [steps[:20], steps[-20:]]:
            fresh = [r for r in window if not r["cached_rollout"]]
            means.append(
                [
                    stats.mean(r["policy_metrics"]["reward"] for r in fresh),
                    stats.mean(r["policy_metrics"]["entropy"] for r in window),
                    stats.mean(r["policy_metrics"]["completions/mean_length"] for r in fresh),
                    100
                    * sum(r["correct_responses"] for r in fresh)
                    / sum(r["generated_responses"] for r in fresh),
                ]
            )
        rows.append(
            [LABEL[k]]
            + [
                f"{means[0][i]:.{3 if i < 2 else 1}f} → {means[1][i]:.{3 if i < 2 else 1}f}"
                for i in range(4)
            ]
        )
    result["RL_WINDOWS"] = table(
        ["Run", "Reward", "Entropy (nats)", "Retained completion tokens", "Candidate correct %"],
        rows,
    )
    dump(
        DATA / "derived_metrics.json",
        {
            "final_scores": values,
            "source_zip_sha256": data["zip_sha256"],
            "sft_loss_aggregation": "unweighted arithmetic mean of logged optimizer-step losses",
            "rl_recovery": "original steps <=40 plus MemoryFix steps >=41",
            "rl_plot_smoothing": "trailing 10 fresh rollouts; entropy 20 updates; candidate ratio pooled over window",
        },
    )
    return result


def render_report(generated):
    text = (DEST / "write-up.template.md").read_text()
    for key, value in generated.items():
        text = text.replace("{{" + key + "}}", value)
    assert not re.search(r"\{\{[A-Z_]+\}\}", text)
    (ROOT / "write-up.md").write_text(text)
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "toc", "attr_list"])

    def inline(match):
        path = ROOT / match[1]
        return 'src="data:image/png;base64,' + base64.b64encode(path.read_bytes()).decode() + '"'

    body = re.sub(r'src="(docs/final-report/figures/[^\"]+\.png)"', inline, body)
    css = """body{font:17px/1.65 Georgia,serif;color:#243044;background:#f1f3f6;margin:0}main{max-width:1120px;margin:32px auto;background:white;padding:64px 70px}h1,h2,h3{font-family:system-ui,sans-serif;line-height:1.25;color:#14263e}h1{font-size:40px}h2{margin-top:2.2em;border-top:1px solid #dae0e8;padding-top:24px}h3{margin-top:1.7em}img{width:100%;height:auto}table{border-collapse:collapse;font:13px/1.55 system-ui,sans-serif;width:100%;margin:24px 0}th,td{padding:8px 10px;border-bottom:1px solid #dce2e8;text-align:left}th{background:#eaf0f7}tr:nth-child(even){background:#f8fafc}a{color:#216898}code{font-size:.83em;overflow-wrap:anywhere}pre{padding:20px;background:#f5f7fa;white-space:pre-wrap}blockquote{border-left:4px solid #4485ac;margin-left:0;padding-left:20px;color:#42566c}.toc{font:14px/1.8 system-ui,sans-serif;background:#f4f7fa;padding:20px 30px}.toc ul{padding-left:20px}.reference{font-size:15px;line-height:1.6;padding-left:2.7em;text-indent:-2.7em;margin:1em 0;overflow-wrap:anywhere}@media(max-width:800px){main{margin:0;padding:24px}table{display:block;overflow-x:auto}}@media print{body,main{background:white;margin:0;padding:0}h2,h3{break-after:avoid}img,table{break-inside:avoid}.toc{display:none}}"""
    (ROOT / "write-up.html").write_text(
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Math Post-Training and Transfer in Qwen2.5-3B</title><style>'
        + css
        + "</style></head><body><main>"
        + body
        + "</main></body></html>"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    data = (
        audit_archive(args.archive)
        if args.archive
        else json.loads((DATA / "analysis_snapshot.json").read_text())
    )
    series = merge_rl(data)
    values = metrics(data)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "bold",
            "svg.fonttype": "none",
            "axes.labelcolor": "#26374D",
            "text.color": "#26374D",
        }
    )
    diagrams()
    sft_fig(data)
    rl_fig(series)
    result_figs(values, series)
    render_report(tables(data, values, series))
    print("Built write-up.md, self-contained write-up.html, and 8 PNG/SVG figures.")


if __name__ == "__main__":
    main()
