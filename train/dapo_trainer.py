"""Bounded dynamic sampling around TRL 0.24's accumulated-token DAPO objective."""

import importlib.util
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

# TRL 0.24 imports the old name even when its own generation path is disabled.
# vLLM 0.14 renamed this class. This path does not request structured generation.
if importlib.util.find_spec("vllm") is not None:
    import vllm.sampling_params as sampling_params
    if not hasattr(sampling_params, "GuidedDecodingParams"):
        sampling_params.GuidedDecodingParams = sampling_params.StructuredOutputsParams

from trl import GRPOConfig, GRPOTrainer

from common.io import append_jsonl, write_json, write_jsonl
from train.dapo_data import batch_schedule
from train.dapo_logps import chunked_lm_head_logps
from train.dapo_sampling import collect_groups


def make_arguments(config, checkpoint_dir, log_dir):
    a = config["algorithm"]
    schedule = batch_schedule(config)
    return GRPOConfig(
        output_dir=str(checkpoint_dir), logging_dir=str(log_dir), **config["training"],
        gradient_accumulation_steps=schedule["mini_batch_size"],
        steps_per_generation=schedule["rollout_batch_size"],
        num_generations=a["group_size"], num_iterations=a.get("num_iterations", 1),
        max_prompt_length=config["data"]["max_prompt_tokens"], max_completion_length=a["max_completion_length"],
        temperature=a["temperature"], top_p=a["top_p"], top_k=None, repetition_penalty=1.0,
        epsilon=a["epsilon"], epsilon_high=a["epsilon_high"], beta=a["beta"],
        loss_type=a["loss_type"], scale_rewards=a["scale_rewards"],
        mask_truncated_completions=a["mask_truncated_completions"],
        importance_sampling_level="token", vllm_importance_sampling_correction=True,
        vllm_importance_sampling_cap=a["importance_sampling_cap"],
        # Custom colocated engine: handles remote EXAONE code, pinning, sleep,
        # max_num_seqs and the public vLLM 0.14 API. Stock TRL's engine is bypassed.
        use_vllm=False, eval_strategy="no", save_strategy="steps", remove_unused_columns=False,
        disable_dropout=True, log_completions=False,
    )


def recorded_rewards(completions, dapo_reward, **kwargs):
    if len(completions) != len(dapo_reward):
        raise ValueError("Reward/completion alignment failed")
    return dapo_reward


def chunk_logps(logits, targets, temperature):
    z = logits.float() / temperature
    return z.gather(-1, targets.unsqueeze(-1)).squeeze(-1) - z.logsumexp(-1)


class DAPOTrainer(GRPOTrainer):
    def __init__(self, *args, dapo_config, prepared_rows, rollout, log_dir, run_identity, **kwargs):
        self.dapo_config = dapo_config
        self.prepared_rows = prepared_rows
        self.rollout = rollout
        self.log_dir = Path(log_dir)
        self.run_identity = run_identity
        self.attempt_dir = self.log_dir / "rollouts" / f"attempt_{time.time_ns()}"
        self.cached_rollout = None
        self.rollout_stats = {}
        self.update_started = None
        self.rollout_first_update = None
        self.schedule = batch_schedule(dapo_config)
        super().__init__(*args, reward_funcs=recorded_rewards, **kwargs)
        if self.accelerator.num_processes != 1:
            raise ValueError("This DAPO runner supports exactly one process/GPU")
        # Enable TRL's train-vs-rollout log-prob correction and old-policy tracking.
        # All generation is intercepted below; the stock vLLM initializer is unused.
        self.use_vllm = True

    def _generate_and_score_completions(self, inputs):
        expected = self.dapo_config["algorithm"]["retained_groups"] * self.num_generations
        if len(inputs) != expected:
            raise ValueError(f"Expected one full update's {expected} rollout slots; received {len(inputs)}")
        self.update_started = time.perf_counter()
        step = self.state.global_step
        self.rollout_first_update = step + 1
        self.rollout.sync(self.accelerator.unwrap_model(self.model))

        def record_batch(records, stats):
            directory = self.attempt_dir / f"update_{step + 1:06d}"
            write_jsonl(directory / f"batch_{stats['generation_batches']:03d}.jsonl", records)
            write_json(directory / "sampling.json", stats)

        def progress(stats):
            print(f"DAPO rollout for updates {step + 1}-{step + self.schedule['updates_per_rollout']}: "
                  f"{stats['retained_groups']}/{self.dapo_config['algorithm']['retained_groups']} "
                  f"mixed groups retained from {stats['candidate_groups']} questions", flush=True)

        try:
            # TRL's dataloader defines update boundaries. A separate deterministic
            # per-update question order supports refill, exact source IDs and resume.
            selected, self.rollout_stats = collect_groups(
                self.prepared_rows, self.rollout.generate, self.dapo_config, step,
                record_batch=record_batch, progress=progress,
            )
        finally:
            self.rollout.sleep()
        self.cached_rollout = selected
        selected_inputs = [{"prompt": r["prompt"], "dapo_reward": r["reward"]} for r in selected]
        try:
            output = super()._generate_and_score_completions(selected_inputs)
        finally:
            self.cached_rollout = None
        if self.dapo_config.get("generation"):
            # TRL 0.24 recognizes only tokenizer.eos_token_id in these metrics.
            # Llama also stops at end-of-text/end-of-message; trust vLLM's reason.
            lengths = [len(r["token_ids"]) for r in selected if r["finish_reason"] == "stop"]
            metrics = self._metrics["train"]
            metrics["completions/clipped_ratio"][-1] = sum(
                r["finish_reason"] == "length" for r in selected) / len(selected)
            for name, value in {"mean": sum(lengths) / len(lengths) if lengths else 0,
                                "min": min(lengths, default=0), "max": max(lengths, default=0)}.items():
                metrics[f"completions/{name}_terminated_length"][-1] = value
        # DAPO divides every microbatch by the SAME total active token count.
        if int(output["completion_mask"].sum()) != int(output["num_items_in_batch"]):
            raise ValueError("DAPO accumulated-token denominator does not match the active completions")
        return output

    def _prepare_inputs(self, generation_batch):
        inputs = super()._prepare_inputs(generation_batch)
        if self.model.training and self.schedule["minibatches_per_rollout"] > 1:
            # TRL shuffles and splits the generated batch before returning the first
            # microbatch. Normalize AFTER that shuffle, over exactly the responses
            # contributing to each optimizer update, not the entire rollout.
            if (self._step - 1) % self.args.steps_per_generation == 0:
                width = self.args.gradient_accumulation_steps
                for start in range(0, len(self._buffered_inputs), width):
                    minibatch = self._buffered_inputs[start:start + width]
                    total = sum(batch["completion_mask"].sum() for batch in minibatch)
                    for batch in minibatch:
                        batch["num_items_in_batch"] = total
            # inputs is an entry of _buffered_inputs; its denominator is now the
            # active minibatch count. Old log-probs/advantages stay unchanged.
        return inputs

    def _generate_single_turn(self, prompts, images):
        if images is not None or self.cached_rollout is None:
            raise ValueError("Generation must come from the checked text-only dynamic sampler")
        rows = self.cached_rollout
        if prompts != [r["prompt"] for r in rows]:
            raise ValueError("Selected prompts changed before policy optimization")
        return ([r["prompt_token_ids"] for r in rows], [r["token_ids"] for r in rows],
                [r["logprobs"] for r in rows], {})

    def _get_per_token_logps_and_entropies(self, model, input_ids, attention_mask, logits_to_keep,
                                          batch_size=None, compute_entropy=False, **kwargs):
        # Trim padding for each microbatch. This also fixes absolute position IDs
        # for EXAONE and avoids running short completions at a batch's 20K maximum.
        prompt_width = input_ids.shape[1] - logits_to_keep
        all_logps, all_entropies = [], []
        chunk = self.dapo_config["algorithm"]["logprob_chunk_tokens"]
        unwrapped = self.accelerator.unwrap_model(model)
        chunk_projection = unwrapped.config.model_type == "qwen2"
        for ids, mask in zip(input_ids, attention_mask, strict=True):
            p = int(mask[:prompt_width].sum())
            c = int(mask[prompt_width:].sum())
            if not p or not c:
                raise ValueError("Empty prompt or completion in DAPO loss")
            unpadded = ids[mask.bool()].unsqueeze(0)
            if chunk_projection:
                # Calling the decoder directly bypasses Accelerate's top-level
                # forward wrapper. Restore its autocast context explicitly. The
                # decoder's native gradient checkpointing remains enabled.
                with self.accelerator.autocast():
                    hidden = unwrapped.model(
                        input_ids=unpadded, attention_mask=torch.ones_like(unpadded),
                        use_cache=False, return_dict=True,
                    ).last_hidden_state[0, p - 1:p + c - 1]
                    logps, entropy = chunked_lm_head_logps(
                        hidden, unwrapped.get_output_embeddings(), unpadded[0, p:],
                        self.temperature, chunk, compute_entropy,
                    )
                all_logps.append(F.pad(logps, (0, logits_to_keep - c)))
                if compute_entropy:
                    all_entropies.append(F.pad(entropy, (0, logits_to_keep - c)))
                continue
            logits = model(input_ids=unpadded, attention_mask=torch.ones_like(unpadded),
                           use_cache=False).logits[0, p - 1:p + c - 1]
            targets = unpadded[0, p:]
            logps, entropies = [], []
            for offset in range(0, c, chunk):
                z, y = logits[offset:offset + chunk], targets[offset:offset + chunk]
                if z.requires_grad:
                    logps.append(checkpoint(chunk_logps, z, y, self.temperature, use_reentrant=False))
                else:
                    logps.append(chunk_logps(z, y, self.temperature))
                if compute_entropy:
                    with torch.no_grad():
                        scaled = z.float() / self.temperature
                        entropies.append(scaled.logsumexp(-1) - (scaled.softmax(-1) * scaled).sum(-1))
            all_logps.append(F.pad(torch.cat(logps), (0, logits_to_keep - c)))
            if compute_entropy:
                all_entropies.append(F.pad(torch.cat(entropies), (0, logits_to_keep - c)))
        return torch.stack(all_logps), torch.stack(all_entropies) if compute_entropy else None

    def log(self, logs, start_time=None):
        if "loss" in logs:
            for key in ["loss", "grad_norm", "learning_rate"]:
                if key not in logs or not math.isfinite(float(logs[key])):
                    raise FloatingPointError(f"Missing/nonfinite DAPO optimizer metric: {key}")
            seconds = time.perf_counter() - self.update_started
            update_in_rollout = self.state.global_step - self.rollout_first_update + 1
            minibatches = self.schedule["minibatches_per_rollout"]
            iteration = (update_in_rollout - 1) // minibatches + 1
            fresh_tokens = self.rollout_stats["generated_tokens"] if update_in_rollout == 1 else 0
            record = {**logs, **self.rollout_stats, "step": self.state.global_step,
                      "rollout_first_update": self.rollout_first_update,
                      "policy_iteration": iteration, "num_iterations": self.num_iterations,
                      "minibatch_index": (update_in_rollout - 1) % minibatches + 1,
                      "update_in_rollout": update_in_rollout, **self.schedule,
                      "cached_rollout": update_in_rollout > 1,
                      "reused_rollout": iteration > 1, "new_generated_tokens": fresh_tokens,
                      "update_seconds": seconds, "generated_tokens_per_second": fresh_tokens / seconds}
            # Capture TRL entropy, clipping, shaped rewards and sampling mismatch metrics.
            record["policy_metrics"] = {k: sum(v) / len(v) for k, v in self._metrics["train"].items() if v}
            append_jsonl(self.log_dir / "steps.jsonl", record)
            self.update_started = time.perf_counter()
        super().log(logs, start_time)

    def _save_checkpoint(self, model, trial):
        # TRL does not serialize its cached rollouts/old log-probs. Save only after
        # every configured pass; resume can then safely generate a fresh batch.
        if self.state.global_step % self.schedule["updates_per_rollout"]:
            raise ValueError("Save checkpoints only after a complete rollout cycle")
        destination = Path(self.args.output_dir) / f"checkpoint-{self.state.global_step}"
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite checkpoint: {destination}")
        super()._save_checkpoint(model, trial)
        write_json(destination / "dapo_checkpoint.json", {
            "run_identity": self.run_identity, "global_step": self.state.global_step,
            "model_kind": self.dapo_config["model_kind"],
        })
