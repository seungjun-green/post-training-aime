"""Stream subprocess diagnostics into notebooks and a local log."""

import os
import signal
import subprocess
from collections import deque
from datetime import datetime, timezone
from pathlib import Path


def run_logged(command, *, cwd, log_path):
    """Keep stderr visible in Colab; include the actual failure in its traceback."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    secrets = [
        os.environ[k]
        for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "GITHUB_TOKEN")
        if os.environ.get(k)
    ]
    tail = deque(maxlen=50)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n--- Run started {datetime.now(timezone.utc).isoformat()} ---\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            start_new_session=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        try:
            for line in process.stdout:
                for secret in secrets:
                    line = line.replace(secret, "[REDACTED]")
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
                tail.append(line)
            code = process.wait()
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise
        finally:
            process.stdout.close()
    if code:
        raise RuntimeError(
            f"Evaluation process exited with status {code}. Log: {log_path}\n" + "".join(tail)
        )
    return log_path
