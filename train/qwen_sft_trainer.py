"""Full-parameter SFT with checkpointed vocabulary projection for long Qwen answers."""

from train.dapo_logps import chunked_lm_head_logps
from train.sft_trainer import Stage1Trainer


class QwenSFTTrainer(Stage1Trainer):
    def __init__(self, *args, lm_head_chunk_tokens=128, **kwargs):
        super().__init__(*args, **kwargs)
        self.lm_head_chunk_tokens = lm_head_chunk_tokens
        if self.accelerator.unwrap_model(self.model).config.model_type != "qwen2":
            raise ValueError("Chunked SFT requires the native Qwen2 model")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Same mean causal CE as the shared trainer; mask before projecting so
        # prompt/padding logits are never allocated. Trainer scales accumulation.
        qwen = self.accelerator.unwrap_model(model)
        labels = inputs["labels"][:, 1:]
        mask = labels != -100
        if not mask.any():
            raise ValueError("No supervised assistant tokens in this batch")
        with self.accelerator.autocast():
            hidden = qwen.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                                use_cache=False, return_dict=True).last_hidden_state
            logps, _ = chunked_lm_head_logps(
                hidden[:, :-1][mask], qwen.lm_head, labels[mask], 1.0,
                self.lm_head_chunk_tokens,
            )
            loss = -logps.mean()
        return (loss, {"loss": loss}) if return_outputs else loss
