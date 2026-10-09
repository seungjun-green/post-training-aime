"""Epoch boundaries and compact notebook telemetry for self-RFT SFT."""

import json
from pathlib import Path

from transformers import TrainerCallback
from transformers.trainer_callback import PrinterCallback, ProgressCallback

PROGRESS_PREFIX = 'SELF_RFT_PROGRESS '


class EpochProgressCallback(TrainerCallback):
    def __init__(self, stop_after_epoch=None):
        self.stop_after_epoch = stop_after_epoch

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'loss' in logs:
            print(PROGRESS_PREFIX + json.dumps({
                'step': state.global_step, 'total': state.max_steps,
                'epoch': state.epoch, 'loss': logs['loss'],
                'learning_rate': logs.get('learning_rate'),
            }), flush=True)

    def on_save(self, args, state, control, **kwargs):
        if self.stop_after_epoch is not None and state.epoch >= self.stop_after_epoch:
            checkpoint = Path(args.output_dir) / f'epoch_{self.stop_after_epoch}'
            if not (checkpoint / 'stage1_checkpoint.json').is_file():
                raise ValueError('Cannot stop before the epoch checkpoint is complete')
            control.should_training_stop = True
        return control


def install_progress(trainer, stop_after_epoch=None):
    trainer.remove_callback(ProgressCallback)
    trainer.remove_callback(PrinterCallback)
    trainer.add_callback(EpochProgressCallback(stop_after_epoch))
