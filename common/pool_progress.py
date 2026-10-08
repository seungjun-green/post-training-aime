"""tqdm progress for pool notebooks; raw diagnostics remain in log files."""
import ast
from contextlib import redirect_stdout
import re
import sys
import time

from common.process import run_logged


class PoolProgress:
    def __init__(self, log_path, title='Sampling'):
        from tqdm.auto import tqdm
        self.tqdm = tqdm
        self.log_path, self.title = str(log_path), title
        self.stream = sys.stdout
        self.buffer = ''
        self.phase = 'Preparing / loading'
        self.rows, self.bars = {}, {}
        self.generating = False
        self.last_refresh = 0
        self.warning_count = 0
        self.latest_grade = None
        self.startup = self.new_bar(desc=f'{title}: {self.phase}', total=None,
                                    bar_format='{desc} | elapsed {elapsed}', leave=False)
        self.stream.write(f'Full log: {self.log_path}\n')

    def new_bar(self, **kwargs):
        # Widget creation must not feed its repr back into the captured subprocess stream.
        with redirect_stdout(self.stream):
            return self.tqdm(file=self.stream, mininterval=1, **kwargs)

    def write(self, text):
        self.buffer += text
        while '\n' in self.buffer:
            line, self.buffer = self.buffer.split('\n', 1)
            self.consume(line)
        return len(text)

    def flush(self):
        self.stream.flush()

    def start_generation(self):
        if not self.generating:
            self.generating = True
            self.startup.close()
            # Start ETA timing after model loading and compilation, with resumed rows initialised.
            self.render()

    def consume(self, line):
        line = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', line).strip()
        match = re.match(r'(full|smoke): (\d+)/(\d+) rows', line)
        if match:
            if 'already complete' not in line:
                self.start_generation()
            self.rows['All problems'] = (int(match[2]), int(match[3]))
            self.phase = 'Loading model' if 'already complete' in line else 'Generating and grading'
            grade = re.search(r'correct=(\d+)/8', line)
            if grade:
                self.latest_grade = grade[1] + '/8'
        elif line.startswith('GPU ') and ': ' in line:
            self.start_generation()
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
            self.start_generation()
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
        self.render()

    def render(self, force=False):
        now = time.monotonic()
        force = force or now - self.last_refresh >= 1
        if force:
            self.last_refresh = now
        if not self.generating:
            self.startup.set_description_str(f'{self.title}: {self.phase}', refresh=force)
            self.startup.update(0)
            return
        for label, value in self.rows.items():
            known = isinstance(value, tuple)
            bar = self.bars.get(label)
            if bar is None or (known and bar.total is None):
                if bar is not None:
                    bar.close()
                done, total = value if known else (0, None)
                bar = self.new_bar(total=total, initial=done, desc=label if known else f'{label}: {value}',
                    unit='problem', smoothing=0, dynamic_ncols=True, leave=True,
                    bar_format='{desc}: {percentage:6.2f}%|{bar}| {n_fmt}/{total_fmt} problems '
                               '[elapsed {elapsed} | remaining {remaining} | {rate_fmt}{postfix}]')
                self.bars[label] = bar
            if known:
                bar.update(max(0, value[0] - bar.n))
            else:
                bar.set_description_str(f'{label}: {value}', refresh=False)
            postfix = {}
            if self.latest_grade is not None and label == 'All problems':
                postfix['last correct'] = self.latest_grade
            if self.warning_count:
                postfix['log warnings'] = self.warning_count
            bar.set_postfix(postfix, refresh=False)
            if force:
                bar.refresh()

    def finish(self, phase):
        if self.buffer:
            self.consume(self.buffer)
            self.buffer = ''
        self.phase = phase
        if phase == 'Complete' and self.rows and not self.generating:
            self.start_generation()
        self.render(force=True)
        self.startup.close()
        for bar in self.bars.values():
            bar.close()
        self.stream.write(f'{self.title}: {phase}. Full log: {self.log_path}\n')
        self.stream.flush()


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


def check_resume(config_path, run_dir):
    """Read-only diagnostics: never rewrite a manifest or bypass its signature guard."""
    import json
    from pathlib import Path
    import yaml
    from pipeline.sample_sft_pool import code_hash
    from common.io import digest
    manifest_path = Path(run_dir) / 'manifest.json'
    if not manifest_path.exists():
        return
    saved = json.loads(manifest_path.read_text())
    requested = yaml.safe_load(Path(config_path).read_text())
    changes = []
    def compare(old, new, prefix=''):
        if isinstance(old, dict) and isinstance(new, dict):
            for key in sorted(set(old) | set(new)):
                name = f'{prefix}.{key}' if prefix else key
                if key not in old:
                    changes.append(f'{name}: added {new[key]!r}')
                elif key not in new:
                    changes.append(f'{name}: removed (saved {old[key]!r})')
                else:
                    compare(old[key], new[key], name)
        elif old != new:
            changes.append(f'{prefix}: saved={old!r}; requested={new!r}')
    compare(saved['config'], requested)
    current_hash = code_hash(Path(__file__).resolve().parents[1])
    if saved.get('code_hash') != current_hash:
        changes.append('Bundled experiment code differs from the saved run. Restore the original experiment bundle to resume.')
    if saved['signature'] == digest({'config': requested, 'code_hash': current_hash}):
        return
    if not changes:
        changes.append('Saved signature is inconsistent with its config/code hash; inspect the manifest without editing it.')
    details = '\n'.join('- ' + change for change in changes)
    raise ValueError('Resume blocked; existing checkpoints are untouched.\n' + details +
                     '\nRestore the saved settings/code to continue this run. Do not delete or edit the manifest to bypass this check.')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-resume', nargs=2, metavar=('CONFIG', 'RUN_DIR'), required=True)
    args = parser.parse_args()
    check_resume(*args.check_resume)
