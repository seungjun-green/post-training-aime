# Spec 0 — Project Overview

## About this spec series

This project will be delivered to you as **five spec files**. This file (Spec 0) gives the big picture and the shared rules. Specs 1–4 will follow, each covering one part of the project.

- Implement the specs **one at a time, in order**.
- Do not start a spec until it has been provided.
- If a spec is ambiguous or conflicts with another spec, stop and ask instead of guessing.

## Project

**Korean math reasoning post-training on EXAONE.**

We take a non-reasoning instruction-tuned model and post-train it for Korean math reasoning in three stages: SFT, then RL, then long2short. The model is evaluated before training and after every stage, so each gain can be attributed to exactly one step. The central experiment is a head-to-head comparison of three RL algorithms, all starting from the same SFT checkpoint.

## Fixed decisions

- **Base model:** `LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct`, full fine-tuning (no LoRA).
- **Hardware:** a single NVIDIA RTX PRO 6000 Blackwell GPU with 96GB VRAM. Everything must run on this one GPU.
- **Language:** all training and evaluation prompts are in Korean, translated from the English originals.
- **Reward:** rule-based final-answer checking. No learned reward model anywhere.

## Spec list

| Spec | Covers | Main outputs |
|---|---|---|
| 1 | Data: n-gram decontamination, Colab translation notebook, evaluation protocol, baseline evaluation | Decontaminated English datasets, Korean datasets, eval harness, baseline results |
| 2 | Stage 1: s1-style SFT | SFT checkpoint and its eval results |
| 3 | Stage 2: RL with naive GRPO, Dr. GRPO, and DAPO | Three RL checkpoints, training logs, eval results |
| 4 | Stage 3: long2short with DPO | Final checkpoint and its eval results |

## Datasets (details in Spec 1)

- **SFT:** s1K-1.1 (DeepSeek-R1 reasoning traces) — `simplescaling/s1K-1.1`
- **RL:** DAPO-Math-17K — `open-r1/DAPO-Math-17k-Processed` (`en` config)
- **Evaluation:** AIME 2024, AIME 2025, AIME 2026, AMC 2023, MATH-500

## Shared rules for all specs

1. **One evaluation protocol.** The evaluation is defined once, in Spec 1. Every later stage must reuse the same eval code and settings without modification, so results are directly comparable.
2. **One prompt format.** Use the same chat template and prompt format for training and evaluation in every stage and for the baseline.
3. **Config-driven runs.** Every run's hyperparameters live in a config file committed to the repo. No hardcoded hyperparameters in scripts.
4. **Reproducibility.** Fix and record random seeds for every run.
5. **Consistent output locations.**
   - Checkpoints: `checkpoints/<stage>/<run_name>/`
   - Eval results: `results/<stage>/<run_name>.json`
   - Training logs: `logs/<stage>/<run_name>/`
6. **Don't modify earlier stages.** When implementing a later spec, leave earlier stages' code and eval untouched unless the spec says otherwise.
