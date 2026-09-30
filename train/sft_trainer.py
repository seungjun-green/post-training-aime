"""Small SFTTrainer extensions for step telemetry and durable epoch exports."""

import math
import time
from pathlib import Path

from transformers import Trainer
from trl import SFTTrainer

from common.io import append_jsonl, write_json


class Stage1Trainer(SFTTrainer):
    def __init__(self, *args, log_dir, run_identity, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_dir = Path(log_dir)
        self.run_identity = run_identity
        self.step_started = None
        self.step_tokens = 0
        # The pinned EXAONE loader computes its own mean CE and does not consume
        # num_items_in_batch. Trainer must scale each microbatch during accumulation.
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs["use_cache"] = False
        # Use the same native CE as SFTTrainer, without TRL's optional full-vocabulary
        # entropy/accuracy diagnostics (large extra allocations on 20K sequences).
        return Trainer.compute_loss(
            self, model, inputs, return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )

    def training_step(self, model, inputs, num_items_in_batch=None):
        if self.step_started is None:
            self.step_started = time.perf_counter()
        self.step_tokens += int(inputs["attention_mask"].sum())
        return super().training_step(model, inputs, num_items_in_batch)

    def log(self, logs, start_time=None):
        if "loss" in logs:
            elapsed = time.perf_counter() - self.step_started
            logs = dict(logs, tokens_per_second=self.step_tokens / elapsed,
                        step_tokens=self.step_tokens, step_seconds=elapsed)
            for key in ["loss", "learning_rate", "grad_norm", "tokens_per_second"]:
                if key not in logs or not math.isfinite(float(logs[key])):
                    raise FloatingPointError(f"Missing/nonfinite optimizer-step metric: {key}")
            append_jsonl(self.log_dir / "steps.jsonl", {
                **logs, "step": self.state.global_step, "epoch": self.state.epoch,
            })
            self.step_tokens = 0
            self.step_started = None
        super().log(logs, start_time)

    def _save_checkpoint(self, model, trial):
        epoch = round(self.state.epoch)
        if not math.isclose(self.state.epoch, epoch, abs_tol=1e-8):
            raise ValueError("Stage 1 saves only complete epochs")
        destination = Path(self.args.output_dir) / f"epoch_{epoch}"
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite completed checkpoint: {destination}")
        super()._save_checkpoint(model, trial)
        temporary = Path(self.args.output_dir) / f"checkpoint-{self.state.global_step}"
        write_json(temporary / "stage1_checkpoint.json", {
            "run_identity": self.run_identity, "epoch": epoch,
            "global_step": self.state.global_step,
        })
        # save_pretrained includes the remote EXAONE Python modules and tokenizer.
        # Rename only after model, optimizer, scheduler, RNG and trainer state finish.
        temporary.rename(destination)
