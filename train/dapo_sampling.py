"""Rule rewards, length shaping, and bounded dynamic group sampling."""

import math

from eval.scoring import score_response
from train.dapo_data import candidate_order


def score_completion(text, gold, length, algorithm):
    if not 0 < length <= algorithm["max_completion_length"]:
        raise ValueError("Invalid rollout length")
    extracted, correct = score_response(text, gold, algorithm["verify_timeout_seconds"])
    interval = algorithm["max_completion_length"] - algorithm["soft_length_limit"]
    penalty = -algorithm["length_penalty_factor"] * max(0, length - algorithm["soft_length_limit"]) / interval
    reward = (algorithm["correct_reward"] if correct else algorithm["incorrect_reward"]) + penalty
    return {"extracted_answer": extracted, "correct": correct, "length_penalty": penalty, "reward": reward}


def collect_groups(rows, generate, config, step, record_batch=None, progress=None):
    a = config["algorithm"]
    order = candidate_order(rows, config["training"]["seed"], step)
    kept, cursor = [], 0
    stats = {"candidate_groups": 0, "mixed_groups": 0, "all_correct_groups": 0, "all_wrong_groups": 0,
             "generated_tokens": 0, "generated_responses": 0, "truncated_responses": 0,
             "correct_responses": 0, "generation_batches": 0}
    for batch_index in range(a["max_generation_batches"]):
        batch = order[cursor:cursor + a["candidate_groups_per_batch"]]
        cursor += len(batch)
        if not batch:
            break
        generated = generate(batch, step, batch_index)
        if len(generated) != len(batch):
            raise ValueError("Rollout engine returned the wrong number of groups")
        journal = []
        for question, completions in zip(batch, generated, strict=True):
            if len(completions) != a["group_size"]:
                raise ValueError("Incomplete DAPO group")
            group = []
            for index, completion in enumerate(completions):
                ids, logps = completion["token_ids"], completion["logprobs"]
                if len(ids) != len(logps) or not all(math.isfinite(p) for p in logps):
                    raise ValueError("Missing/nonfinite rollout token log probabilities")
                score = score_completion(completion["text"], question["gold"], len(ids), a)
                group.append({**question, **completion, **score, "sample_index": index})
                stats["generated_tokens"] += len(ids)
                stats["generated_responses"] += 1
                stats["truncated_responses"] += completion["finish_reason"] == "length"
                stats["correct_responses"] += score["correct"]
            correct = sum(r["correct"] for r in group)
            mixed = 0 < correct < a["group_size"]
            accepted = mixed and len(kept) < a["retained_groups"]
            stats["candidate_groups"] += 1
            stats["mixed_groups"] += mixed
            stats["all_correct_groups"] += correct == a["group_size"]
            stats["all_wrong_groups"] += correct == 0
            if accepted:
                kept.append(group)
            for r in group:
                # Text is enough for audit; avoid duplicating millions of token IDs on Drive.
                journal.append({k: v for k, v in r.items() if k not in {"token_ids", "logprobs", "prompt_token_ids"}} |
                               {"tokens": len(r["token_ids"]), "mixed_group": mixed, "retained": accepted})
        stats["generation_batches"] += 1
        stats["retained_groups"] = len(kept)
        stats["acceptance_rate"] = stats["mixed_groups"] / stats["candidate_groups"]
        if record_batch:
            record_batch(journal, dict(stats))
        if progress:
            progress(dict(stats))
        if len(kept) == a["retained_groups"]:
            return [r for group in kept for r in group], stats
    raise RuntimeError(
        f"Dynamic sampling stopped: retained {len(kept)}/{a['retained_groups']} mixed groups "
        f"after {stats['candidate_groups']} questions. No optimizer update was made for this batch. "
        "Inspect the saved rollout diagnostics before increasing the retry limit or G."
    )
