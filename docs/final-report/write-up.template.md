# Math Post-Training and Cross-Domain Transfer in Qwen2.5-3B

**A nine-variant study of structured reasoning SFT, rejection sampling, mixed 3B/7B distillation, and DAPO reinforcement learning**

Final experimental report · 14 October 2026

## Abstract

Can math-only post-training improve a small language model beyond mathematics, and how does that depend on the training recipe? We compare nine Qwen2.5-3B variants: pretrained base and instruct baselines; three supervised fine-tuning (SFT) recipes; and four DAPO reinforcement-learning (RL) runs from different starting checkpoints. The SFT recipes use 989 structured Kimi-inspired s1 examples, 28,267 correct self-generated solutions, or 37,691 mixed 3B/7B solutions. Each RL run performs 300 optimizer updates on a verified-answer math dataset. Greedy evaluation covers AMC 2023, MATH-500, ARC-Challenge, a fixed 1,400-question MMLU-Pro subset, a project-defined non-math BBH subset, and IFEval.

Base + RL improves MATH-500 from 54.8% to 64.2%; mixed-data SFT + RL reaches 67.2%, the best base-derived result. Rejection and mixed-data SFT also improve the base model on all four principal transfer metrics. Additional RL has inconsistent effects: mixed-data SFT + RL improves MATH-500 by 2.6 percentage points but lowers non-math MMLU-Pro by 5.69 points relative to its SFT parent. The original instruct baseline remains strongest on every main benchmark. Training logs show declining SFT loss, increasing RL reward, and decreasing RL token entropy, without corresponding monotonic evaluation improvements. These single-seed results support recipe-dependent transfer, rather than a general claim that math post-training improves all capabilities.

[TOC]

## 1. Motivation and experimental scope

The project asks three practical questions:

1. Which math post-training recipe most improves a 3B pretrained model?
2. Does mathematical supervision transfer to science, broader knowledge, logical tasks, or instruction following?
3. How well do training loss, reward, entropy, and response length explain the observed evaluation outcomes?

All nine final variants belong to the Qwen2.5-3B family. The pretrained and instruct checkpoints are distinct starting points: the instruct model already includes upstream post-training. It is therefore an external baseline for these experiments, not a compute-matched control for the project’s base-derived models. The architecture and pretrained checkpoint are described in the Qwen2.5 report [1](#ref-1) and [Qwen2.5-3B model card](https://huggingface.co/Qwen/Qwen2.5-3B).

Earlier EXAONE and Korean-translation experiments motivated the project but are outside this final comparison. Their previous report is preserved in [the historical write-up](docs/archive/write-up-exaone-2026-10-03.md).

### 1.1 Evidence and provenance

This report combines three evidence sources:

- **Inspected training artifacts:** the supplied `qwen_training_logs_20261013T145235973661Z.zip`. All **369 collected files** passed ZIP integrity and per-file SHA-256 verification. Every expected training history was present, including recovery and continuation runs. Logged losses, gradient norms, and learning rates were finite; step ranges were continuous within each run.
- **Saved math evaluation metrics:** the ZIP contains final AMC/MATH metrics for all nine selected checkpoints and per-epoch results for rejection and mixed-data SFT. Tables below read these files directly. Raw generated answers were excluded by the collector, so this audit verifies recorded metrics and counts, not the correctness of every individual grading decision.
- **Reported transfer results and dataset history:** the cross-domain scores were supplied in the project conversation. Their raw RunPod responses, subject breakdowns, and scoring manifests are not in this ZIP. Dataset-construction details are recorded project history unless corroborated by the supplied training reports.

The [analysis snapshot](docs/final-report/data/analysis_snapshot.json) preserves the chart inputs, source paths, run configurations, and archive hash. [Reported transfer scores](docs/final-report/data/reported_transfer.json) are explicitly labeled as such. This distinction prevents a recorded aggregate from being mistaken for an independently regraded result.

## 2. Data construction

### 2.1 Shared pool for rejection sampling and mixed-data SFT

The recorded pipeline combines MATH train [6](#ref-6), NuminaMath-1.5 [7](#ref-7), and math-only s1K problems [5](#ref-5). It removes proofs, multiple-choice and multi-part problems, and examples lacking a parseable reference answer. Exact normalized deduplication is followed by MinHash near-deduplication, with priority MATH > NuminaMath > s1K.

Decontamination removes a training problem if it covers at least 70% of an individual evaluation problem’s distinct 8-grams, measured relative to the evaluation question. This also catches an evaluation question embedded in a longer prompt. Questions shorter than eight tokens use exact normalized substring matching. The pipeline checks AIME 2024–2026, AMC 2023, and MATH-500, and uses the same method to remove DAPO-Math-17K overlap. Near misses at 50–70% coverage are subsequently removed.

The recorded intermediate pool contains 380,943 problems. Further selection removes invalid NuminaMath rows, multiple-choice and `other` question types, and the `orca_math`, `cn_k12`, and `metamath` sub-sources. Sampling retains all MATH and s1K examples, targets 10K olympiad and 8K synthetic problems, 3K each from AOPS and Chinese contests, and the smaller inequalities, number-theory, and olympiad-reference groups. Algebra is capped at 40% within each sampled sub-source. The reported final generation pool contains **27,001 problems**.

This collection does not include the pool-construction execution manifest. Some previously documented pre-generation cleaning steps—such as removing leftover translation instructions or correcting malformed gold answers—remain unconfirmed at the execution level. The report therefore treats the 27,001 count and upstream cleaning recipe as reported provenance, while treating final training counts below as verified from saved reports. It does not claim a new contamination audit against the four transfer benchmarks.

![Dataset construction](docs/final-report/figures/01_data_pipeline.png)

*Figure 1. Construction of the two larger SFT datasets. Pool-stage details follow project history; final row and unique-problem counts come from the training artifacts. Arrows show data dependencies, not a controlled isolation of teacher size.*

### 2.2 Rejection sampling: correct solutions from the 3B base model

The reported generation stage samples eight responses per problem from Qwen2.5-3B base at temperature 1.0. Math-Verify [13](#ref-13) grades extracted final answers. Only correct, normally terminated responses survive. Responses are cut after their final balanced boxed answer, with permitted closing punctuation retained. Further filters remove missing boxes, CJK text, code blocks, chat/role contamination, and very short responses.

Difficulty is estimated using the 3B’s count of correct responses out of eight. All problems solved 1–7 times are eligible; a seeded random third of the 8/8 group is kept; 0/8 problems have no self-generated correct targets. Solutions are processed shortest first and near-deduplicated using 4-gram Jaccard similarity at least 0.7. The cap is three solutions for problems solved 1–6 times and two for problems solved 7–8 times.

The final dataset, [3B self-RFT math](https://huggingface.co/datasets/Seungjun/qwen2.5-3b-self-rft-math), contains **28,267 problem–response pairs across 12,863 unique problems**. Multiple rows can share the same question; these counts refer to different units. The saved training report confirms that all rows were retained without additional length truncation or deduplication during SFT preparation.

### 2.3 Mixed 3B/7B solutions

The teacher is **Qwen2.5-Math-7B-Instruct** [2](#ref-2). The recorded generation recipe applies it to 18,520 problems that the 3B solved only 0–2 times. Problems solved 3–8 times use 3B solutions. For 1–2/8 problems, 3B solutions are preferred and 7B solutions fill remaining slots when available. Problems unsolved by the 3B are admitted only when the teacher solves them at least 2/8 times. Deduplication is applied jointly across solution sources, with at most three retained solutions per hard problem.

The final [mixed 3B/7B dataset](https://huggingface.co/datasets/Seungjun/qwen2.5-3b-self-rft-7b-distill-math) contains **37,691 pairs across 14,853 unique problems**, all retained during SFT preparation. We call this model **B-Mix** below. The earlier experiment label “7B knowledge distillation SFT” refers to this mixture, not exclusively teacher-generated responses or soft-logit distillation.

This comparison changes teacher participation, problem coverage, example count, and training tokens simultaneously. It cannot identify the isolated causal benefit of a 7B teacher. Exact generation-time caps and the teacher’s executed sampling settings are not established by the training-log ZIP: the written generation history and repository templates contain differing caps. They should not be inferred from the SFT run manifest. The final dataset revisions and training targets are pinned and recorded.

### 2.4 Kimi-inspired structured s1 targets

The separate s1 branch uses `Seungjun/dp_removed_s1K-1.1`, with `question` as input and the complete `kimi-style-reasoning-answer` column as the supervised response. Of 996 source rows, **989** have usable outputs; seven missing outputs are excluded. No mathematical-correctness filter is recorded for these targets.

The responses were generated through DeepSeek and use Planning, Solution and Evaluation, Reflection, Exploration, and a final answer. The intended behaviors are to state an approach, check intermediate work, refine the solution, and consider alternatives. These behaviors are inspired by the long-CoT warmup discussion in [Kimi k1.5 §2.2](https://arxiv.org/html/2501.12599v1#S2.SS2). The fixed headings are this project’s prompt design, not a reproduced official Kimi template [4](#ref-4). The full final response is supervised; the separate API reasoning field is not the training target, and no additional `<think>` wrapper is added.

### 2.5 What the student actually sees

The project’s training dataset releases are collected in reference [14](#ref-14).

| SFT dataset | Training pairs | Unique problems | Mean supervised tokens, including EOS | Updates per epoch |
| --- | ---: | ---: | ---: | ---: |
| Kimi-style s1 | 989 | Not separately counted in supplied report | 1,732.8 | 62 |
| 3B rejection sampling | 28,267 | 12,863 | 429.9 | 1,767 |
| Mixed 3B/7B | 37,691 | 14,853 | 468.0 | 2,356 |

These lengths use the actual Qwen training tokenizer. They are not interchangeable with earlier character counts or token estimates from other tokenizers. Kimi targets are roughly four times as long as rejection-sampling targets, but the dataset contains far fewer examples and receives many fewer updates.

The RL source is `Seungjun/dp_removed_DAPO-Math-17k-Processed`, English train split, with **14,068 rows**. Its resolved revision in the four saved RL manifests is `e3929adfc2916140ce6900ff0fb37de3b30f0f37`. The larger SFT pool was designed to exclude RL overlap; this is not evidence of a separately verified disjointness guarantee for the historical Kimi branch.

## 3. Training design

![Experiment tree](docs/final-report/figures/02_experiment_tree.png)

*Figure 2. The nine final variants. Arrows identify immediate parents used for relative comparisons. The mixed-data epoch-6/7 extension is diagnostic and does not change the epoch-5 parent chosen for B-Mix-RL.*

| Label in this report | Experiment | Selected checkpoint |
| --- | --- | --- |
| B0 | Qwen2.5-3B pretrained base | Original HF revision |
| B-RL | Base + DAPO | RL update 300 |
| B-Kimi | Base + Kimi-style s1 SFT | SFT epoch 5 |
| B-RS | Base + 3B rejection SFT | SFT epoch 4 |
| B-Mix | Base + mixed 3B/7B SFT | SFT epoch 5 |
| B-RS-RL | B-RS + DAPO | RL update 300 |
| B-Mix-RL | B-Mix + DAPO | RL update 300 |
| I0 | Qwen2.5-3B-Instruct | Original HF revision |
| I-RL | Instruct + DAPO | RL update 300 |

### 3.1 Supervised fine-tuning

SFT updates all model parameters. The input uses the saved Qwen chat template and the project’s English math instruction. The user prompt is masked from the loss, and the response plus end token is supervised through explicitly constructed labels. The low-level Trainer flags alone do not describe this custom masking.

| Setting | Original SFT runs | Mixed-data continuation |
| --- | --- | --- |
| Epochs | 5 | 6–7, starting from epoch 5 |
| Microbatch / accumulation | 1 / 16 | 1 / 16 |
| Peak learning rate | 1e−5 | 1e−6 |
| Schedule | Cosine, 5% warmup | New cosine schedule, no warmup |
| Optimizer | AdamW; β₁=0.9, β₂=0.95, ε=1e−8 | Preserved optimizer state |
| Weight decay / gradient clipping | 1e−4 / 1.0 | Same |
| Precision | BF16 compute | Same |
| Maximum sequence length | 20,480 | Same |
| Packing / prompt truncation | Disabled / none | Same |
| Seed | 42 | 42 |

All three preparation reports record zero overlength removals. Checkpoints are saved each epoch. Rejection SFT completes **8,835 updates**; mixed-data SFT completes **11,780**; Kimi SFT completes **310**. The continuation adds **4,712 updates**, ending at global step **16,492**.

**Token-format distinction.** Rejection and mixed-data SFT preserve base Qwen’s `<|endoftext|>` EOS/PAD (151643), appending the original EOS to each completion. The historical Kimi run instead uses the existing `<|im_end|>` token as EOS (151645), with `<|endoftext|>` padding. Its vocabulary and chat template were not expanded, but its EOS setting differs from B0. Evaluations preserve each saved checkpoint’s settings. Consequently the Kimi comparison is not a pure target-style ablation.

### 3.2 DAPO reinforcement learning

The implementation follows the DAPO approach of asymmetric clipping, dynamic sampling, token-level loss, and length-aware reward shaping, adapted to a single-GPU TRL runner. This is a small-model project implementation, not a reproduction of the original large-scale results in the [DAPO paper](https://arxiv.org/abs/2503.14476) [3](#ref-3).

Each rollout samples eight answers per candidate problem. Groups with both correct and incorrect answers are retained; all-correct and all-wrong groups are excluded from the optimizer batch. Sampling refills until 16 groups are retained, giving 128 responses. These are split into two disjoint 64-response optimizer updates, with one pass over the rollout. Thus **300 updates correspond to 150 rollouts and 2,400 retained question selections**, not 2,400 proven unique questions and not an epoch over all 14,068 rows. Additional candidate questions are generated and rejected by dynamic sampling.

| Setting | All four RL runs |
| --- | --- |
| Optimizer updates / checkpoint interval | 300 / 20 |
| Generation temperature / top-p | 1.0 / 1.0 |
| Group size / retained groups | 8 / 16 |
| Minibatch responses / updates per rollout | 64 / 2 |
| Learning rate / warmup | 1e−6 / 20 updates; constant after warmup |
| AdamW β₁, β₂ / weight decay | 0.9, 0.95 / 0 |
| Correct / incorrect reward | +1 / −1 |
| Maximum response / soft-length threshold | 20,480 / 16,384 tokens |
| Lower / upper clipping epsilon | 0.20 / 0.28 |
| KL coefficient | 0 |
| Reward normalization | Within group |
| Truncated-completion masking | Disabled |
| Importance-sampling correction cap | 2.0 |
| Seed | 42 |

For response length L, the shaped reward is the correctness reward minus `max(0, L − 16,384) / 4,096`. Reward therefore combines final-answer correctness and a long-output penalty; it is not itself an accuracy percentage. The verifier checks the extracted final answer, not every reasoning step.

The saved runtime uses PyTorch 2.9.1, Transformers 4.57.6, TRL 0.24.0, vLLM 0.14.1, and Math-Verify 0.9.0. Training ran on an RTX PRO 6000 Blackwell Server Edition, nominally 96 GB. The RL implementation retains FP32 trainable/master weights and optimizer state with BF16 compute and rollout copies, uses gradient checkpointing, and chunks the vocabulary projection to reduce peak memory. Transfer evaluation later used two RTX PRO 6000 Workstation GPUs with one model per GPU.

### 3.3 Recovery and checkpoint selection

I-RL originally stopped after update 43 and recovered from the completed update-40 checkpoint into a separate MemoryFix run. The plotted history uses **original steps 1–40 plus recovered steps 41–300**. Original steps 41–43 are abandoned and excluded. The recovery manifest confirms the parent and boundary.

B-RS epoch 4 and B-Mix epoch 5 were selected after observing AMC/MATH results. B-Mix was also continued through epochs 6–7 with a lower learning rate, but epoch 5 remained the RL starting point. These are exploratory, benchmark-informed checkpoint choices. There is no independent checkpoint-selection validation set, so “best checkpoint” means best under the project’s observed selection process, not an unbiased estimate of an unseen-test optimum.

## 4. Evaluation protocol

All final evaluations use **temperature 0, top-p 1, one response per problem**, and the checkpoint’s saved chat template and EOS. The math and transfer suites use different output caps; they should be compared within their respective protocols.

| Benchmark | Problems per model | Prompting and scoring | New-token cap |
| --- | ---: | --- | ---: |
| AMC 2023 [8](#ref-8) | 40 | English math instruction; final-box extraction and mathematical equivalence | 20,480 |
| MATH-500 [6](#ref-6) | 500 | Same math protocol | 20,480 |
| ARC-Challenge [9](#ref-9) | 1,172 | Zero-shot generated reasoning and answer letter; answer accuracy | 4,096 |
| MMLU-Pro custom subset [10](#ref-10) | 1,400 | 100 questions in each of 14 subjects; five official same-subject validation demonstrations | 4,096 |
| Non-math BBH custom subset [11](#ref-11) | 4,615 | 19 task configurations; three official CoT demonstrations | 4,096 |
| IFEval [12](#ref-12) | 541 | Original prompts and official strict/loose checks | 4,096 |

The transfer suite totals **7,728 prompts per model and 69,552 across nine models**. It replaces an initially launched full MMLU-Pro run with a smaller, separate evaluation. Question selection uses seed 42 and deterministic SHA-256 ranking of question IDs within each subject, shared by all models. The **non-math MMLU aggregate contains 1,300 questions**, excluding math. Because all subjects have 100 questions, the overall accuracy equals the subject macro average. These scores are explicitly custom-subset results, not full MMLU-Pro leaderboard scores. Prompts follow the [official MMLU-Pro resources](https://github.com/TIGER-AI-Lab/MMLU-Pro).

BBH includes boolean expressions, causal judgement, disambiguation QA, Dyck languages, formal fallacies, hyperbaton, logical deduction with three/five/seven objects, movie recommendation, ruin names, salient translation-error detection, snarks, sports understanding, tracking shuffled objects with three/five/seven objects, web of lies, and word sorting. It excludes arithmetic, counting, geometric shapes, dates, navigation, penguin tables, colored-object reasoning, and temporal sequences. “Non-math” is a project-defined selection; it does not isolate a single pure reasoning faculty. The demonstrations come from [BIG-Bench Hard](https://github.com/suzgunmirac/BIG-Bench-Hard).

ARC reports generated-answer accuracy rather than likelihood-based `acc_norm`. MMLU/BBH extraction failures count as incorrect, without random guessing. IFEval uses its [official verifiable instruction checks](https://github.com/google-research/google-research/blob/master/instruction_following_eval/README.md), including strict and loose prompt- and instruction-level scores. These generative chat-formatted protocols are internal comparisons rather than exact reproductions of every benchmark’s public leaderboard setup.

## 5. Results

### 5.1 Mathematics: gains over base, but no improvement over instruct

{{MATH}}

*Table 1. Final selected checkpoints. Counts and mean token lengths are read from saved evaluation metrics. Tokens include all responses, correct and incorrect.*

B-RL raises AMC accuracy from **37.5% to 42.5%** and MATH-500 from **54.8% to 64.2%**. Among base-derived models, **B-Mix** gives the best AMC result, **47.5%**, and **B-Mix-RL** gives the best MATH result, **67.2%**. B-RS-RL improves MATH over its SFT parent by **4.8 points**, but loses **5.0 points** on AMC. B-Mix-RL gains **2.6 points** on MATH while losing **2.5 points** on AMC.

The unmodified instruct model remains best on both math benchmarks: **52.5% AMC and 70.0% MATH-500**. I-RL scores **50.0% and 65.8%**, respectively. Thus the experiments demonstrate substantial base-model improvement without surpassing the existing instruct checkpoint.

B-Kimi falls to **25.0% AMC and 51.4% MATH**, while producing much longer outputs. This is a negative result for the complete implemented recipe. It does not isolate the contribution of fixed section headings, target correctness, dataset size, training exposure, or EOS handling.

### 5.2 Transfer: SFT gains, heterogeneous RL effects

{{TRANSFER}}

*Table 2. User-reported transfer results. MMLU-Pro uses the fixed 100-per-subject subset; BBH uses the custom task selection. IFEval’s main measure here is strict prompt accuracy. MMLU overall and macro scores coincide by design.*

Relative to B0, both B-RS and B-Mix improve ARC, non-math MMLU-Pro, BBH, and strict prompt-level IFEval. B-Mix raises ARC by **4.61 points** and BBH by **8.60 points**. B-RL achieves the highest non-math MMLU-Pro score among base-derived variants, **24.08%**, a **7.54-point** increase over B0, while losing **3.07 points** on ARC.

Additional RL has different effects depending on the parent. B-RS-RL gains slightly on all four displayed transfer metrics. B-Mix-RL loses **2.82 points on ARC**, **5.69 on non-math MMLU-Pro**, **1.13 on BBH**, and **0.37 on strict prompt IFEval**, despite its MATH gain. I-RL also decreases on each main transfer metric, although several differences are small. I0 remains highest on every reported transfer metric.

B-Kimi’s **1.65% BBH** score is unusually low. The supplied aggregate cannot establish how much of this is task performance, output-format mismatch, truncation, or answer extraction. A response-level audit is needed before describing it as a collapse of logical reasoning.

![Changes relative to immediate parents](docs/final-report/figures/03_transfer_deltas.png)

*Figure 3. Percentage-point changes from each immediate parent, not from a shared baseline for every row. Positive cells indicate improvement. MMLU and BBH asterisks identify custom subsets. Small changes are descriptive; no statistical-significance claim is made.*

### 5.3 SFT trajectories: decreasing loss, fluctuating evaluation

![SFT trajectories](docs/final-report/figures/04_sft_trajectories.png)

*Figure 4. Arithmetic means of logged optimizer-step losses within each epoch; these are not end-of-epoch console loss values or a separately evaluated training-set NLL. Kimi loss has its own scale because its targets differ. Stars mark the selected rejection and mixed-data checkpoints. The dashed boundary introduces the lower-learning-rate mixed-data continuation.*

Rejection SFT’s mean logged loss decreases from **0.3483 at epoch 1 to 0.3231 at epoch 5**, but AMC peaks at epoch 4 and then falls. Mixed-data SFT decreases from **0.3053 to 0.2804** over epochs 1–5, remaining near **0.2803** in epochs 6–7. Its MATH score reaches **66.0% at epoch 7**, while AMC falls from **47.5% at epoch 5 to 37.5% at epoch 7**. More epochs therefore produce a trade-off rather than steady improvement on both benchmarks.

Kimi SFT loss also decreases, from **0.7811 to 0.6575**. Its final evaluation decline shows why fitting the supervised target distribution is insufficient evidence of improved benchmark performance. Absolute loss levels across different datasets are not model-quality rankings.

{{EPOCHS}}

*Table 3. Full rejection and mixed-data checkpoint trajectories. The epoch-6/7 continuation uses a new, lower-rate schedule while preserving optimizer state. Kimi has only its final evaluated checkpoint in this study.*

### 5.4 RL trajectories: reward, length, entropy, and candidate correctness

![RL training dynamics](docs/final-report/figures/05_rl_dynamics.png)

*Figure 5. Faint lines are raw observations; solid lines use trailing windows of 10 fresh rollouts for reward/length/candidate correctness and 20 optimizer updates for entropy. Early windows contain fewer observations. Candidate correctness is the pooled correct-response count divided by generated-response count over the window. The vertical dotted line marks I-RL’s recovery boundary. Reward and completion length use available fresh-rollout log entries for retained groups; entropy is logged on every update.*

{{RL_WINDOWS}}

*Table 4. First 20 versus last 20 optimizer updates, after recovery stitching. Reward and retained-group length average the ten fresh-rollout observations within each window; entropy averages all 20 updates. Candidate correctness pools generated responses before mixed-group filtering.*

All four runs increase logged reward between these windows and reduce token entropy. The strongest reward change is B-RL, **−0.543 → −0.215**; its entropy decreases **0.663 → 0.160 nats**. Retained-group completion length increases in all four early/late comparisons. Negative reward values are compatible with learning because incorrect answers receive −1 and the logged statistic concerns selected mixed groups.

Candidate correctness rises too, but this is a training diagnostic on changing sampled problems. It is neither fixed-set validation accuracy nor pass@8. The selection process changes which groups reach the optimizer, and one rollout feeds two updates. Repeated rollout-level fields must not be counted twice. Lower entropy is an observed behavioral change; these data do not establish it as the cause of transfer losses.

![Reward changes versus evaluation changes](docs/final-report/figures/06_reward_vs_results.png)

*Figure 6. Within each RL run, the small reward marker represents the early window and the large marker the late window. Evaluation panels show final-minus-parent percentage-point changes on separate axes. Reward is not numerically comparable to accuracy. Improving the training objective does not ensure gains on every held-out benchmark.*

### 5.5 Response length and efficiency

![Accuracy versus output length](docs/final-report/figures/07_accuracy_length.png)

*Figure 7. Greedy math evaluation: mean tokens versus accuracy, with logarithmic token axes. Arrows connect immediate parents to their trained variants. Square markers are the original base/instruct checkpoints. This is an outcome comparison, not a causal estimate of the value of an extra generated token.*

B-Kimi uses **8,556.8 mean tokens on AMC** and **6,709.7 on MATH**, roughly **4.9× and 4.6× B0**, while scoring lower. B-RL improves MATH while reducing mean length from **1,444.2 to 895.1 tokens**. B-Mix-RL improves its parent’s MATH score while reducing mean length from **1,161.3 to 776.0 tokens**. Long output is therefore not a reliable proxy for effective reasoning in this comparison.

The apparent increase in sampled training-rollout length and decrease in some greedy benchmark lengths is not a contradiction: the prompts, filtering, decoding temperature, and response populations differ. Output tokens are an inference-length proxy; the report does not equate them with measured latency or monetary cost.

## 6. Discussion

**Verified short solutions are a useful baseline.** Rejection SFT improves MATH and all four main transfer measures over B0 without imposing a long structured response format. Mixed-data SFT broadens coverage and improves several results further. Because its data size and composition change together, the experiment supports the combined recipe, not a teacher-size-only explanation.

**RL’s value depends on the starting model and target metric.** The same nominal 300-update recipe benefits base MATH strongly, yields smaller MATH gains after SFT, and lowers instruct MATH. Transfer is similarly heterogeneous. Neither “RL always helps” nor “RL always hurts general ability” fits these results.

**Optimization metrics and evaluation answer different questions.** SFT loss measures fit to the target responses. RL reward reflects shaped correctness on selected training rollouts. Neither includes the full set of constraints represented by science questions, unfamiliar answer formats, logical tasks, and instruction-following checks. Their improvement should be reported alongside evaluation, not substituted for it.

**The instruct baseline is an essential reference.** I0 outperforms every project variant on the main math and transfer benchmarks. The project’s strongest claim is that several recipes substantially improve a pretrained 3B checkpoint. It does not show that this amount of math post-training replaces upstream instruction tuning or produces a superior general assistant.

## 7. Limitations

1. **One seed and exploratory selection.** There are no repeated-seed estimates. SFT checkpoints and extra epochs were chosen with benchmark feedback, making final math scores selection-sensitive. No formal significance test is reported.
2. **Small and custom evaluation sets.** AMC has 40 problems: one answer changes accuracy by 2.5 points. Each MMLU subject has 100 questions: one answer changes its subject score by one point. The custom MMLU and BBH results cannot be substituted for full-benchmark scores.
3. **Unequal data and compute.** SFT datasets differ in size, difficulty, response length, source quality, and update count. Mixed-data SFT is not a controlled pure-distillation ablation. Even at 300 RL updates, dynamic sampling produces different candidate-generation workloads.
4. **Kimi recipe confounds.** Dataset size, lack of correctness filtering, target format, and EOS setting all differ. Its result cannot establish that planning/reflection/exploration themselves are harmful.
5. **Scoring and output-format uncertainty.** Math checks final answers rather than proof validity. Transfer scoring can be affected by extraction and truncation. The raw transfer records were not supplied for this write-up, so the unusually low Kimi BBH result remains unaudited at response level.
6. **Incomplete upstream generation provenance.** Final training rows and revisions are pinned, but the collected artifacts do not settle every pool-cleaning action or executed generation parameter. Reported decontamination is not proof of no semantic overlap, and transfer-benchmark overlap was not newly checked here.
7. **No causal attribution to entropy or length.** Their changes accompany training; the experiments do not independently manipulate them. Adaptive training-rollout correctness cannot establish improved held-out generalization.

## 8. Conclusion

Math-only post-training produced useful but uneven gains in Qwen2.5-3B. Rejection sampling and mixed 3B/7B SFT improved several capabilities beyond mathematics. DAPO substantially improved the pretrained base model’s MATH score, but its incremental benefit depended on the starting checkpoint and benchmark. The structured Kimi-style recipe increased generated length without improving final accuracy, while the original instruct model remained strongest overall.

The practical lesson is to evaluate the complete recipe: target data, response format, training exposure, checkpoint choice, and decoding/scoring protocol. Declining loss, increasing reward, and longer reasoning are not sufficient success criteria. In this study, the most informative evidence is their relationship to both math accuracy and cross-domain behavior.

## Appendix A. Full IFEval results

{{IFEVAL}}

*Strict/loose and prompt/instruction metrics measure different criteria; no single column represents all instruction-following behavior. These are the supplied aggregate results.*

## Appendix B. RL optimizer diagnostics

![RL optimizer diagnostics](docs/final-report/figures/08_rl_diagnostics.png)

*Faint raw traces and 20-update trailing means. Policy loss is not a supervised cross-entropy or accuracy measure; its sign and magnitude should not be read as a standalone success criterion. The learning-rate schedules overlap because the four runs share the same schedule. Logged clipping fractions remain small in these runs.*

## Appendix C. Reproducibility and artifact map

The report and charts can be regenerated with:

```bash
python -m pip install -r requirements-writeup.txt
python scripts/build_final_writeup.py
```

To re-audit and rebuild the analysis snapshot from the original supplied archive:

```bash
python scripts/build_final_writeup.py --archive /path/to/qwen_training_logs_20261013T145235973661Z.zip
```

The builder verifies the archive, selects the exact math checkpoint files, stitches I-RL at step 40, checks continuous step ranges, and produces the following artifacts:

| Artifact | Purpose |
| --- | --- |
| [write-up.md](write-up.md) | This report, generated from the report template |
| [write-up.html](write-up.html) | Standalone reading version with embedded chart images |
| [Report template](docs/final-report/write-up.template.md) | Editable narrative with generated-table placeholders |
| [Figure directory](docs/final-report/figures/) | All eight charts/diagrams as PNG and editable vector SVG |
| [Analysis snapshot](docs/final-report/data/analysis_snapshot.json) | Source-attributed chart data, configurations and source-archive checksum |
| [Derived metrics](docs/final-report/data/derived_metrics.json) | Final score inputs and aggregation definitions |
| [Reported transfer table](docs/final-report/data/reported_transfer.json) | Exact user-supplied percentages and provenance note |
| [Training-log collector](notebooks/collect_qwen_training_logs.ipynb) | CPU Colab collection of source logs and small metadata |
| [Experiment progress and paths](progress.md) | Notebook links and exact training/evaluation Drive paths |

**Pinned starting models:** Qwen2.5-3B at `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b`; Qwen2.5-3B-Instruct at `aa8e72537993ba99e69dfaafa59ed015b17504d1`.

**Pinned SFT datasets:** Kimi-style s1 at `636ecf409774771afb0bf10a436f4b1e608b5f29`; rejection SFT at `96c5f70e37098a9f41f5e44adc4e906ee3f5ef15`; mixed 3B/7B SFT at `d13f567cd42e7f7e182ea5125a288d86cbefbe67`. Full training identities, runtime versions, and recovered/continued parent paths are retained in the analysis snapshot.

**Evaluation settings:** [math comparison configuration](configs/qwen_compare_eval.yaml), [selected SFT + RL comparison](configs/qwen_sft_rl_eval.yaml), and [MMLU-100 transfer configuration](configs/qwen_transfer_eval_mmlu100.yaml). Cross-domain outputs were configured for `Seungjun/qwen25-3b-transfer-eval-results`; upload success or current repository visibility is not assumed by this report.

## References

<a id="ref-1"></a>
**[1]** Qwen et al. (2024). *Qwen2.5 Technical Report*. arXiv:2412.15115. [Paper](https://arxiv.org/abs/2412.15115). Model releases: [Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B) and [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct).
{: .reference}

<a id="ref-2"></a>
**[2]** An Yang et al. (2024). *Qwen2.5-Math Technical Report: Toward Mathematical Expert Model via Self-Improvement*. arXiv:2409.12122. [Paper](https://arxiv.org/abs/2409.12122). Teacher release: [Qwen2.5-Math-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct).
{: .reference}

<a id="ref-3"></a>
**[3]** Qiying Yu et al. (2025). *DAPO: An Open-Source LLM Reinforcement Learning System at Scale*. arXiv:2503.14476. [Paper](https://arxiv.org/abs/2503.14476).
{: .reference}

<a id="ref-4"></a>
**[4]** Kimi Team et al. (2025). *Kimi k1.5: Scaling Reinforcement Learning with LLMs*. arXiv:2501.12599. [Paper](https://arxiv.org/abs/2501.12599). [Version 1, §2.2](https://arxiv.org/html/2501.12599v1#S2.SS2).
{: .reference}

<a id="ref-5"></a>
**[5]** Niklas Muennighoff et al. (2025). *s1: Simple test-time scaling*. arXiv:2501.19393. [Paper](https://arxiv.org/abs/2501.19393); [official code and data resources](https://github.com/simplescaling/s1).
{: .reference}

<a id="ref-6"></a>
**[6]** Dan Hendrycks et al. (2021). *Measuring Mathematical Problem Solving With the MATH Dataset*. arXiv:2103.03874. [Paper](https://arxiv.org/abs/2103.03874). MATH-500 release used as the upstream evaluation source: [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500).
{: .reference}

<a id="ref-7"></a>
**[7]** Project Numina / AI-MO. *NuminaMath-1.5*. [Dataset and documentation](https://huggingface.co/datasets/AI-MO/NuminaMath-1.5).
{: .reference}

<a id="ref-8"></a>
**[8]** math-ai. *AMC 2023 dataset release*. [math-ai/amc23](https://huggingface.co/datasets/math-ai/amc23).
{: .reference}

<a id="ref-9"></a>
**[9]** Peter Clark et al. (2018). *Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge*. arXiv:1803.05457. [Paper](https://arxiv.org/abs/1803.05457); [dataset](https://huggingface.co/datasets/allenai/ai2_arc).
{: .reference}

<a id="ref-10"></a>
**[10]** Yubo Wang et al. (2024). *MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark*. arXiv:2406.01574. [Paper](https://arxiv.org/abs/2406.01574); [official repository](https://github.com/TIGER-AI-Lab/MMLU-Pro).
{: .reference}

<a id="ref-11"></a>
**[11]** Mirac Suzgun et al. (2022). *Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them*. arXiv:2210.09261. [Paper](https://arxiv.org/abs/2210.09261); [official BBH tasks and prompts](https://github.com/suzgunmirac/BIG-Bench-Hard).
{: .reference}

<a id="ref-12"></a>
**[12]** Jeffrey Zhou et al. (2023). *Instruction-Following Evaluation for Large Language Models*. arXiv:2311.07911. [Paper](https://arxiv.org/abs/2311.07911); [official IFEval implementation](https://github.com/google-research/google-research/tree/master/instruction_following_eval).
{: .reference}

<a id="ref-13"></a>
**[13]** Hugging Face. *Math-Verify*. [Source and documentation](https://github.com/huggingface/Math-Verify). Version 0.9.0.
{: .reference}

<a id="ref-14"></a>
**[14]** Seungjun. *Project training datasets*. Hugging Face dataset releases: [Kimi-style s1 targets](https://huggingface.co/datasets/Seungjun/dp_removed_s1K-1.1), [3B rejection-sampling targets](https://huggingface.co/datasets/Seungjun/qwen2.5-3b-self-rft-math), [mixed 3B/7B targets](https://huggingface.co/datasets/Seungjun/qwen2.5-3b-self-rft-7b-distill-math), and [processed DAPO training problems](https://huggingface.co/datasets/Seungjun/dp_removed_DAPO-Math-17k-Processed). Exact training revisions are recorded in Appendix C and the analysis snapshot.
{: .reference}
