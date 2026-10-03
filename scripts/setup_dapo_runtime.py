"""Install the isolated, pinned EXAONE DAPO training/generation runtime."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", default="/content/lg-dapo-env")
    args = parser.parse_args()
    python = Path(args.venv) / "bin/python"
    if not python.exists():
        subprocess.check_call([sys.executable, "-m", "uv", "venv", "--python", "3.12", args.venv])
    subprocess.check_call([sys.executable, "-m", "uv", "pip", "sync", "--python", str(python),
                           str(ROOT / "requirements-dapo.lock")])
    subprocess.check_call([str(python), "-c",
        "from train.dapo_trainer import DAPOTrainer; import torch, transformers, trl, vllm; "
        "print('torch', torch.__version__, 'transformers', transformers.__version__, "
        "'trl', trl.__version__, 'vllm', vllm.__version__); "
        "assert torch.cuda.is_available(), 'Connect the GPU runtime'; "
        "assert 'ExaoneForCausalLM' in vllm.ModelRegistry.get_supported_archs(); "
        "print(torch.cuda.get_device_name(0))"], cwd=ROOT)


if __name__ == "__main__":
    main()
