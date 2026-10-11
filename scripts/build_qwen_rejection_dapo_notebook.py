"""Build 300 DAPO updates from the selected 3B rejection-sampling SFT epoch 4."""

from build_notebooks import ROOT, payload, write_notebook
from build_qwen_base_dapo_notebook import BUNDLE_FILES as BASE_FILES
from build_qwen_distill_dapo_notebook import cells as distill_cells

BUNDLE_FILES = BASE_FILES + [
    'train/run_qwen_rejection_dapo.py', 'configs/dapo_qwen25_3b_rejection.yaml',
]


def cells():
    result = distill_cells()
    replacements = {
        'B-7B knowledge distillation SFT + RL': 'B-rejection sampling SFT + RL',
        '7B-distillation': '3B rejection-sampling',
        'sft_qwen25_3b_base_7b_distill': 'sft_qwen25_3b_base_self_rft',
        'dapo_qwen25_3b_7b_distill_epoch5': 'dapo_qwen25_3b_rejection_epoch4',
        'base-7b-distillation-sft': 'base-rejection-sampling-sft',
        'lg-aime-dapo-qwen25-distill': 'lg-aime-dapo-qwen25-rejection',
        'lg-dapo-distill-env': 'lg-dapo-rejection-env',
        'qwen25-distill-dapo-work': 'qwen25-rejection-dapo-work',
        'configs/dapo_qwen25_3b_7b_distill.yaml': 'configs/dapo_qwen25_3b_rejection.yaml',
        'train.run_qwen_distill_dapo': 'train.run_qwen_rejection_dapo',
        'select_distill_model': 'select_rejection_model',
        'epoch_5': 'epoch_4', 'epoch-5': 'epoch-4', 'epoch 5': 'epoch 4',
        'AMC 47.5%, MATH-500 64.6%': 'AMC 37.5%, MATH-500 61.4%',
        'Epochs 6–7 from the SFT continuation are separate experiments.':
            'Epoch 4 is the user-selected best checkpoint from the completed five-epoch SFT run.',
        'original five-epoch distillation run': 'five-epoch rejection-sampling run',
    }
    encoded = payload([ROOT / name for name in BUNDLE_FILES])
    for cell in result:
        if cell.cell_type == 'code' and 'BUNDLE =' in cell.source:
            cell.source = '\n'.join(
                f'BUNDLE = {encoded!r}' if line.startswith('BUNDLE = ') else line
                for line in cell.source.splitlines())
        else:
            for before, after in replacements.items():
                cell.source = cell.source.replace(before, after)
    return result


if __name__ == '__main__':
    write_notebook('train_dapo_qwen25_3b_rejection_epoch4.ipynb', cells())
