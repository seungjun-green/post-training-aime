"""Stream subprocess diagnostics into notebooks and a local log."""

import os
import re
import signal
import subprocess
from collections import deque
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path


def run_logged(command, *, cwd, log_path, progress_totals=None):
    """Keep stderr visible in Colab; include the actual failure in its traceback."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    secrets = [
        os.environ[k]
        for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "GITHUB_TOKEN")
        if os.environ.get(k)
    ]
    tail = deque(maxlen=50)
    progress = nullcontext(None)
    offsets = {}
    if progress_totals:
        from tqdm.auto import tqdm

        total = 0
        for name, count in progress_totals.items():
            offsets[name] = total
            total += count
        progress = tqdm(total=total, desc="Loading evaluation", unit="problem")
    with progress as bar, log_path.open("a", encoding="utf-8") as log:
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
                match = re.fullmatch(r"([\w-]+): (\d+)/(\d+) problems\s*", line)
                if bar is not None and match and match[1] in offsets:
                    name, done, expected = match[1], int(match[2]), int(match[3])
                    if expected == progress_totals[name] and 0 <= done <= expected:
                        bar.set_description(name, refresh=False)
                        bar.update(max(0, offsets[name] + done - bar.n))
                    else:
                        print(line, end="", flush=True)
                else:
                    print(line, end="", flush=True)
                log.write(line)
                log.flush()
                tail.append(line)
            code = process.wait()
            if code == 0 and bar is not None:
                # Fully resumed datasets emit no per-problem log lines.
                bar.update(bar.total - bar.n)
                bar.set_description("Evaluation complete")
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
