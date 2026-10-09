"""One notebook tqdm bar per stage; retain complete subprocess logs on Drive."""

import json
import os
import re
import signal
import subprocess
from collections import deque
from datetime import datetime, timezone
from pathlib import Path


def run_progress(command, *, cwd, log_path, description, total, mode, step_offset=0):
    from tqdm.auto import tqdm

    if mode not in ('training', 'evaluation'):
        raise ValueError('Unknown progress mode')
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    secrets = [os.environ[k] for k in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN') if os.environ.get(k)]
    tail = deque(maxlen=40)
    with path.open('a', encoding='utf-8') as log, tqdm(
            total=total, desc=description, unit='update' if mode == 'training' else 'problem') as bar:
        log.write(f'\n--- Run started {datetime.now(timezone.utc).isoformat()} ---\n')
        process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace',
            bufsize=1, start_new_session=True, env={**os.environ, 'PYTHONUNBUFFERED': '1'})
        try:
            for line in process.stdout:
                for secret in secrets:
                    line = line.replace(secret, '[REDACTED]')
                log.write(line)
                log.flush()
                tail.append(line)
                if mode == 'training' and line.startswith('SELF_RFT_PROGRESS '):
                    event = json.loads(line.removeprefix('SELF_RFT_PROGRESS '))
                    done = min(total, max(0, event['step'] - step_offset))
                    bar.update(max(0, done - bar.n))
                    bar.set_postfix(loss=f"{event['loss']:.4f}",
                                    lr=f"{event['learning_rate']:.2g}", refresh=False)
                elif mode == 'evaluation':
                    match = re.fullmatch(r'(amc23|math_500): (\d+)/(\d+) problems\s*', line)
                    if match:
                        name, done, expected = match[1], int(match[2]), int(match[3])
                        if expected != {'amc23': 40, 'math_500': 500}[name] or done > expected:
                            raise ValueError('Unexpected evaluation progress counts')
                        done += 40 if name == 'math_500' else 0
                        bar.update(max(0, done - bar.n))
            result = process.wait()
            if result:
                raise RuntimeError(f'{description} failed (exit {result}). Log: {path}\n' + ''.join(tail))
            bar.update(max(0, total - bar.n))
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
    return path
