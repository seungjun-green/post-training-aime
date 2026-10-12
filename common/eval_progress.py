"""Notebook-only evaluation display; deliberately outside the inference code digest."""

import re
import time
from contextlib import redirect_stdout

from common.process import run_logged

BENCHMARK_LABELS = {'arc_challenge': 'ARC-Challenge', 'mmlu_pro': 'MMLU-Pro',
                    'bbh_nonmath': 'BBH', 'ifeval': 'IFEval'}


class EvaluationDisplay:
    """Turn subprocess output into one bar while run_logged retains diagnostics."""

    def __init__(self, bar, totals, label):
        self.bar, self.totals, self.label = bar, totals, label
        self.offsets, offset = {}, 0
        for name, count in totals.items():
            self.offsets[name] = offset
            offset += count
        self.buffer = ''
        self.started = False
        self.last_refresh = 0
        self.stage = None
        self.bar.set_postfix_str(f'{self.bar.total:,} remaining', refresh=False)

    def baseline(self, completed):
        # Cached responses must not count as newly generated work in tqdm's rate/ETA.
        # reset() clears its moving average and restarts the wall clock.
        self.bar.reset()
        self.bar.n = self.bar.initial = self.bar.last_print_n = completed
        self.bar.set_postfix_str(f'{self.bar.total - completed:,} remaining', refresh=False)
        self.bar.refresh()

    def write(self, text):
        self.buffer += text
        while '\n' in self.buffer:
            line, self.buffer = self.buffer.split('\n', 1)
            self.consume(line)
        return len(text)

    def flush(self):
        # print(..., flush=True) also flushes after pieces without a newline.
        # Retain these until the entire line arrives or the subprocess finishes.
        pass

    def finish(self):
        if self.buffer:
            self.consume(self.buffer)
            self.buffer = ''

    def consume(self, line):
        clean = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', line).strip()
        cached = re.fullmatch(r'.+: (\d+)/(\d+) responses already saved', clean)
        if cached and not self.started and int(cached[2]) == self.bar.total:
            saved = int(cached[1])
            if 0 <= saved <= self.bar.total:
                self.baseline(saved)
        match = re.fullmatch(r'([\w-]+): (\d+)/(\d+) problems', clean)
        if match and match[1] in self.totals:
            name, done, expected = match[1], int(match[2]), int(match[3])
            if expected == self.totals[name] and 0 <= done <= expected:
                if not self.started:
                    # The first benchmark event follows model loading/compilation.
                    self.baseline(self.bar.n)
                    self.started = True
                target = max(self.bar.n, self.offsets[name] + done)
                self.bar.set_description(f'{self.label} · {BENCHMARK_LABELS.get(name, name)}', refresh=False)
                self.bar.set_postfix_str(f'{self.bar.total - target:,} remaining', refresh=False)
                self.bar.update(target - self.bar.n)
                if self.stage != name:
                    self.stage = name
                    self.bar.refresh()
        # Refresh elapsed/ETA on worker heartbeats without printing the heartbeat.
        # Startup messages, warnings, score lines and child tqdm bars stay in the log.
        now = time.monotonic()
        if now - self.last_refresh >= 1:
            self.bar.refresh()
            self.last_refresh = now


def run_eval_logged(command, *, cwd, log_path, progress_totals, label='Evaluation'):
    """Keep a single tqdm display; preserve full logs, errors and cancellation behavior."""
    from tqdm.auto import tqdm

    if not progress_totals or any(not isinstance(n, int) or n < 0 for n in progress_totals.values()):
        raise ValueError('Expected benchmark problem counts')
    print('Detailed log:', log_path, flush=True)
    with tqdm(total=sum(progress_totals.values()), desc=f'{label} · Loading', unit='problem',
              mininterval=1, dynamic_ncols=True, leave=True,
              bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} completed '
                         '[{elapsed}<{remaining}, {rate_fmt}{postfix}]') as bar:
        display = EvaluationDisplay(bar, progress_totals, label)
        try:
            # The existing runner logs/redacts every line and cleans up child processes
            # on failure or interruption. Redirect only its notebook console output.
            with redirect_stdout(display):
                result = run_logged(command, cwd=cwd, log_path=log_path)
            display.finish()
        except BaseException as error:
            status = 'Interrupted' if isinstance(error, KeyboardInterrupt) else 'Failed'
            bar.set_description(f'{label} · {status}')
            raise  # run_logged includes the actual diagnostic tail and full log path.
        if bar.n != bar.total:
            bar.set_description(f'{label} · Incomplete progress')
            raise RuntimeError(f'Worker exited before reporting all problems ({bar.n}/{bar.total}). See {log_path}')
        bar.set_postfix_str('0 remaining', refresh=False)
        bar.set_description(f'{label} · Complete')
        return result
