"""Restore training state but restart cosine decay only when leaving parent epoch 5."""

from pathlib import Path

from transformers import get_cosine_schedule_with_warmup

from train.qwen_sft_trainer import QwenSFTTrainer


class QwenContinuationTrainer(QwenSFTTrainer):
    def __init__(self, *args, parent_checkpoint, additional_steps, **kwargs):
        self.parent_checkpoint = Path(parent_checkpoint).resolve()
        self.additional_steps = additional_steps
        if additional_steps < 1:
            raise ValueError('Continuation needs a positive update budget')
        super().__init__(*args, **kwargs)

    def create_scheduler(self, num_training_steps, optimizer=None):
        if self.lr_scheduler is None:
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer or self.optimizer, num_warmup_steps=0,
                num_training_steps=self.additional_steps)
            self._created_lr_scheduler = True
        return self.lr_scheduler

    def _load_optimizer_and_scheduler(self, checkpoint):
        if checkpoint is None:
            raise ValueError('Continuation requires a complete parent or continuation checkpoint')
        super()._load_optimizer_and_scheduler(checkpoint)
        if Path(checkpoint).resolve() == self.parent_checkpoint:
            # Optimizer moments/step counts survive. Only LR metadata and scheduler reset.
            for group in self.optimizer.param_groups:
                group['lr'] = group['initial_lr'] = self.args.learning_rate
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                self.optimizer, num_warmup_steps=0, num_training_steps=self.additional_steps)
        # On epoch-6 resume, retain the saved local scheduler step and current LR.
