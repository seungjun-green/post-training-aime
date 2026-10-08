"""Presentation-only progress for pool notebooks; raw diagnostics stay in run_logged files."""
import ast
from contextlib import redirect_stdout
import re
import sys
import time

from common.process import run_logged


class PoolProgress:
    def __init__(self, log_path, title='Sampling', clock=time.monotonic):
        self.log_path, self.title, self.clock = str(log_path), title, clock
        self.started = clock()
        self.last_render = -float('inf')
        self.stream = sys.stdout
        self.buffer = ''
        self.handle = None
        self.phase = 'Preparing / loading'
        self.rows = {}
        self.baseline = None
        self.warning_count = 0
        self.warning = ''
        try:
            from IPython import get_ipython
            self.notebook = getattr(get_ipython(), 'kernel', None) is not None
        except ImportError:
            self.notebook = False
        self.render(force=True)

    @staticmethod
    def duration(seconds):
        minutes, seconds = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)
        return f'{hours:02d}:{minutes:02d}:{seconds:02d}'

    def write(self, text):
        self.buffer += text
        while '\n' in self.buffer:
            line, self.buffer = self.buffer.split('\n', 1)
            self.consume(line)
        return len(text)

    def flush(self):
        self.stream.flush()

    def consume(self, line):
        line = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', line).strip()
        match = re.match(r'(full|smoke): (\d+)/(\d+) rows', line)
        if match:
            self.rows['All problems'] = (int(match[2]), int(match[3]))
            self.phase = 'Loading model' if 'already complete' in line else 'Generating and grading'
        elif line.startswith('GPU ') and ': ' in line:
            for item in line.split(' | '):
                label, value = item.split(': ', 1)
                if value.startswith('{'):
                    try:
                        record = ast.literal_eval(value)
                        self.rows[label] = (int(record['done']), int(record['total']))
                    except (ValueError, SyntaxError, KeyError, TypeError):
                        self.rows[label] = 'See worker log'
                elif value == 'finished' and isinstance(self.rows.get(label), tuple):
                    total = self.rows[label][1]
                    self.rows[label] = (total, total)
                else:
                    self.rows[label] = value
            self.phase = 'Parallel generation and grading'
        elif 'Generation active:' in line:
            self.phase = 'Generating and grading'
        elif 'Starting to load model' in line or 'Loading safetensors' in line:
            self.phase = 'Loading model weights'
        elif 'Capturing CUDA graphs' in line:
            self.phase = 'Warming up GPU / capturing graphs'
        elif 'torch.compile' in line or 'Compiling a graph' in line:
            self.phase = 'Compiling model'
        elif 'Saved result files:' in line:
            self.phase = 'Results exported'
        elif 'Verified upload:' in line:
            self.phase = 'Upload verified'
        if re.search(r'\b(WARNING|WARN|ERROR|Error)\b', line):
            self.warning_count += 1
            self.warning = line[:180]
        counts = [v for v in self.rows.values() if isinstance(v, tuple)]
        if counts and len(counts) == len(self.rows) and self.baseline is None:
            self.baseline = (sum(v[0] for v in counts), self.clock())
        self.render()

    def render(self, force=False):
        now = self.clock()
        if not force and now - self.last_render < (1 if self.notebook else 30):
            return
        self.last_render = now
        lines = [f'{self.title} — {self.phase}', f'Elapsed: {self.duration(now-self.started)}']
        for label, value in self.rows.items():
            if isinstance(value, tuple):
                done, total = value
                fraction = done / total if total else 1
                bars = min(20, int(fraction * 20))
                lines.append(f'{label}: [{"█" * bars}{"·" * (20-bars)}] {done:,}/{total:,} ({fraction:.1%})')
            else:
                lines.append(f'{label}: {value}')
        counts = [v for v in self.rows.values() if isinstance(v, tuple)]
        if self.baseline and counts and len(counts) == len(self.rows):
            done, total = sum(v[0] for v in counts), sum(v[1] for v in counts)
            new = done - self.baseline[0]
            elapsed = now - self.baseline[1]
            if new > 0 and elapsed > 0:
                rate = new / elapsed
                lines.append(f'{rate*60:.1f} problems/min · estimated generation remaining: {self.duration(max(0, total-done)/rate)}')
        if self.warning_count:
            lines.append(f'Diagnostics: {self.warning_count} warning/error lines (full details in log)')
            lines.append(self.warning)
        lines.append(f'Full log: {self.log_path}')
        data = {'text/plain': '\n'.join(lines)}
        if self.notebook:
            from IPython.display import display
            if self.handle is None:
                self.handle = display(data, raw=True, display_id=True)
            else:
                self.handle.update(data)
        else:
            self.stream.write(data['text/plain'] + '\n')
            self.stream.flush()

    def finish(self, phase):
        if self.buffer:
            self.consume(self.buffer)
            self.buffer = ''
        self.phase = phase
        self.render(force=True)


def run_pool_logged(command, *, cwd, log_path, title='Sampling'):
    """Keep existing logging, redaction and interruption behavior; collapse console output."""
    progress = PoolProgress(log_path, title)
    try:
        with redirect_stdout(progress):
            result = run_logged(command, cwd=cwd, log_path=log_path)
    except BaseException:
        progress.finish('Stopped / failed — see error below and full log')
        raise
    progress.finish('Complete')
    return result
