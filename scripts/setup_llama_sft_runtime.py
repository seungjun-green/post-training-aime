"""Create a separate locked Python 3.12 LoRA environment on the GPU host."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", default="/content/llama-lora-env")
    args = parser.parse_args()
    python = Path(args.venv) / "bin/python"
    if not python.exists():
        subprocess.check_call([sys.executable, "-m", "uv", "venv", "--python", "3.12", args.venv])
    subprocess.check_call([
        sys.executable, "-m", "uv", "pip", "sync", "--python", str(python),
        str(ROOT / "requirements-stage1-lora.lock"),
    ])
    subprocess.check_call([
        str(python), "-c",
        "import torch, transformers, trl, peft; "
        "print('torch', torch.__version__, 'transformers', transformers.__version__, "
        "'trl', trl.__version__, 'peft', peft.__version__); "
        "assert torch.cuda.is_available(), 'Connect the CUDA GPU runtime'; "
        "print(torch.cuda.get_device_name(0))",
    ])
    print("LoRA training Python:", python)


if __name__ == "__main__":
    main()
