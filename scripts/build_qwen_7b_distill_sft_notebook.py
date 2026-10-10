"""Build five-epoch Qwen 3B SFT with per-epoch evaluation on the 7B distillation dataset."""

from build_notebooks import ROOT, bootstrap, payload, write_notebook
from build_qwen_self_rft_sft_notebook import cells as self_rft_cells
from train.qwen_self_rft_data import BUNDLE_FILES as SHARED_FILES

BUNDLE_FILES = SHARED_FILES + ['configs/sft_qwen25_3b_7b_distill.yaml']


def cells():
    result = self_rft_cells()
    replacements = {
        'Qwen2.5-3B BASE + self-RFT SFT': 'Qwen2.5-3B BASE + 7B distillation SFT',
        'B-rejection sampling SFT': 'B-7B knowledge distillation SFT',
        'Seungjun/qwen2.5-3b-self-rft-math': 'Seungjun/qwen2.5-3b-self-rft-7b-distill-math',
        '96c5f70e37098a9f41f5e44adc4e906ee3f5ef15': 'd13f567cd42e7f7e182ea5125a288d86cbefbe67',
        '28,267': '37,691', '12,863': '14,853',
        '1,767': '2,356', '8,835': '11,780',
        'base-rejection-sampling-sft': 'base-7b-distillation-sft',
        'sft_qwen25_3b_base_self_rft': 'sft_qwen25_3b_base_7b_distill',
        'configs/sft_qwen25_3b_self_rft.yaml': 'configs/sft_qwen25_3b_7b_distill.yaml',
        '/content/lg-qwen-self-rft-sft': '/content/lg-qwen-7b-distill-sft',
        '/content/lg-qwen-self-rft-eval-env': '/content/lg-qwen-7b-distill-eval-env',
    }
    for index, cell in enumerate(result):
        if cell.cell_type == 'code' and 'BUNDLE =' in cell.source:
            boot = bootstrap(payload([ROOT / path for path in BUNDLE_FILES]))
            boot.source = boot.source.replace('/content/lg-korean-aime', '/content/lg-qwen-7b-distill-sft')
            result[index] = boot
        else:
            for before, after in replacements.items():
                cell.source = cell.source.replace(before, after)
    intro = result[0].source
    intro = intro.replace('Input: **`problem`**. Assistant target: the **entire `response`**.',
        'Input: **`problem`**. Assistant target: the **entire `response`**.\n\n'
        'The published dataset combines **28,415 3B responses** and **9,276 7B responses**.\n'
        'Its manifest identifies the teacher as **`Qwen/Qwen2.5-Math-7B-Instruct`**.\n'
        'The trained student remains **Qwen2.5-3B base**. Both response sources are retained.')
    result[0].source = intro
    return result


if __name__ == '__main__':
    write_notebook('train_qwen25_3b_base_7b_distill.ipynb', cells())
