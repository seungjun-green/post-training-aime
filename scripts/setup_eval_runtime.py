"""Install the locked Linux GPU evaluation environment outside Colab's Python."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", default="/content/lg-eval-env")
    args = parser.parse_args()
    target = Path(args.venv)
    python = target / "bin/python"
    if not python.exists():
        subprocess.check_call([sys.executable, "-m", "uv", "venv", "--python", "3.12", str(target)])
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "uv",
            "pip",
            "sync",
            "--python",
            str(python),
            str(ROOT / "requirements-eval.lock"),
        ]
    )
    subprocess.check_call(
        [
            str(python),
            "-c",
            "import torch, transformers, vllm; "
            "print('torch', torch.__version__, 'CUDA', torch.version.cuda, "
            "'transformers', transformers.__version__, 'vllm', vllm.__version__); "
            "assert torch.cuda.is_available(), 'Connect the CUDA GPU runtime'; "
            "x=torch.ones(1, device='cuda'); print(torch.cuda.get_device_name(0), x.item())",
        ]
    )
    print("Evaluation Python:", python)


if __name__ == "__main__":
    main()
