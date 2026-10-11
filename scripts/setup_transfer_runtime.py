"""Install the existing GPU lock plus pinned IFEval dependencies and verify offline scoring."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--venv', required=True)
    args = parser.parse_args()
    subprocess.check_call([sys.executable, str(ROOT / 'scripts/setup_eval_runtime.py'), '--venv', args.venv])
    python = str(Path(args.venv) / 'bin/python')
    subprocess.check_call([sys.executable, '-m', 'uv', 'pip', 'install', '--python', python,
                           '-c', str(ROOT / 'requirements-eval.lock'),
                           '-r', str(ROOT / 'requirements-transfer-extra.txt')])
    subprocess.check_call([python, '-c',
        "from eval.transfer_scoring import ifeval_scores; "
        "d={'key':0,'prompt':'Say hello','instruction_id_list':['keywords:existence'],"
        "'kwargs':[{'keywords':['hello']}]}; "
        "assert ifeval_scores(d,'hello')['prompt_strict']; "
        "from instruction_following_eval import instructions_util; "
        "assert instructions_util.count_sentences('Hello. Goodbye.')==2; "
        "from vllm import ModelRegistry; "
        "assert 'Qwen2ForCausalLM' in ModelRegistry.get_supported_archs(); "
        "print('Transfer runtime and official IFEval scorer ready')"], cwd=ROOT)


if __name__ == '__main__':
    main()
