import io
import sys
from types import SimpleNamespace

import pytest

from common.eval_progress import EvaluationDisplay, run_eval_logged


class Bar:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.total = kwargs['total']
        self.n = self.initial = self.last_print_n = 0
        self.deltas, self.descriptions, self.postfixes = [], [], []
        self.reset_count = 0
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def reset(self):
        self.n = 0
        self.reset_count += 1

    def refresh(self):
        pass

    def update(self, delta):
        self.deltas.append(delta)
        self.n += delta

    def set_description(self, text, **kwargs):
        self.descriptions.append(text)

    def set_postfix_str(self, text, **kwargs):
        self.postfixes.append(text)


@pytest.fixture
def bars(monkeypatch):
    created = []

    def factory(**kwargs):
        bar = Bar(**kwargs)
        created.append(bar)
        return bar

    monkeypatch.setitem(sys.modules, 'tqdm.auto', SimpleNamespace(tqdm=factory))
    return created


def test_noisy_child_is_one_bar_with_resume_and_complete_log(tmp_path, bars, capsys, monkeypatch):
    monkeypatch.setenv('HF_TOKEN', 'hidden-token')
    output = ('B0: 3/5 responses already saved\n'
              'INFO vLLM startup\nWARNING engine setting hidden-token\n'
              '\rLoading shards: 50%|####| 1/2\rLoading shards: 100%|####| 2/2\n'
              'arc_challenge: 2/2 problems\nmmlu_pro: 1/3 problems\n'
              'Generation active: 32 problems pending; GPU response limit 32\n'
              'mmlu_pro: 2/3 problems\nmmlu_pro: 3/3 problems\n'
              'mmlu_pro generated_answer_accuracy: 66.67%\n')
    command = [sys.executable, '-c', f'import sys; sys.stdout.write({output!r})']
    log = tmp_path / 'eval_console.log'
    assert run_eval_logged(command, cwd=tmp_path, log_path=log,
                           progress_totals={'arc_challenge': 2, 'mmlu_pro': 3}, label='B0') == log
    assert len(bars) == 1
    bar = bars[0]
    assert bar.n == 5 and bar.initial == 3 and bar.closed
    assert sum(bar.deltas) == 2  # Only two new problems contribute to throughput.
    assert bar.reset_count == 2  # Restore cache count, then exclude engine loading time.
    assert 'B0 · MMLU-Pro' in bar.descriptions and bar.descriptions[-1] == 'B0 · Complete'
    assert bar.postfixes[-1] == '0 remaining' and '2 remaining' in bar.postfixes
    for field in ['{n_fmt}', '{total_fmt}', '{elapsed}', '{remaining}', '{rate_fmt}']:
        assert field in bar.kwargs['bar_format']
    visible = capsys.readouterr().out
    assert 'Detailed log:' in visible
    assert 'Generation active' not in visible and 'vLLM' not in visible and 'Loading shards' not in visible
    saved = log.read_text()
    assert 'Generation active' in saved and 'WARNING' in saved and 'generated_answer_accuracy' in saved
    assert 'hidden-token' not in saved and '[REDACTED]' in saved


def test_failures_keep_counts_and_diagnostic_tail(tmp_path, bars, monkeypatch):
    monkeypatch.setenv('HF_TOKEN', 'secret-token')
    command = [sys.executable, '-c', "import sys; print('arc_challenge: 1/2 problems'); "
               "print('CUDA out of memory secret-token', file=sys.stderr); sys.exit(1)"]
    log = tmp_path / 'eval.log'
    with pytest.raises(RuntimeError, match='CUDA out of memory') as failure:
        run_eval_logged(command, cwd=tmp_path, log_path=log,
                        progress_totals={'arc_challenge': 2}, label='B0')
    assert bars[0].n == 1 and bars[0].closed and bars[0].descriptions[-1] == 'B0 · Failed'
    assert 'secret-token' not in str(failure.value) + log.read_text()
    assert str(log) in str(failure.value)


def test_cancellation_leaves_incomplete_bar(tmp_path, bars, monkeypatch):
    import common.eval_progress as progress

    def interrupted(*args, **kwargs):
        print('arc_challenge: 1/2 problems')
        raise KeyboardInterrupt

    monkeypatch.setattr(progress, 'run_logged', interrupted)
    with pytest.raises(KeyboardInterrupt):
        run_eval_logged([], cwd=tmp_path, log_path=tmp_path / 'log',
                        progress_totals={'arc_challenge': 2}, label='B0')
    assert bars[0].n == 1 and bars[0].closed and bars[0].descriptions[-1] == 'B0 · Interrupted'


def test_split_lines_invalid_events_and_no_fake_success(tmp_path, bars):
    bar = Bar(total=2)
    display = EvaluationDisplay(bar, {'arc_challenge': 2}, 'B0')
    display.write('arc_challenge: 1/2 prob')
    display.flush()
    assert bar.n == 0
    display.write('lems\narc_challenge: 2/99 problems\narc_challenge: 9/2 problems\n')
    assert bar.n == 1
    display.write('arc_challenge: 2/2 problems')
    display.finish()
    assert bar.n == 2
    with pytest.raises(RuntimeError, match='before reporting all problems'):
        run_eval_logged([sys.executable, '-c', "print('arc_challenge: 1/2 problems')"],
                        cwd=tmp_path, log_path=tmp_path / 'log', progress_totals={'arc_challenge': 2})
    assert bars[0].n == 1


def test_fully_resumed_run_and_multiple_attempts(tmp_path, bars):
    log = tmp_path / 'log'
    command = [sys.executable, '-c', "print('B0: 2/2 responses already saved'); print('arc_challenge: 2/2 problems')"]
    for _ in range(2):
        run_eval_logged(command, cwd=tmp_path, log_path=log, progress_totals={'arc_challenge': 2})
    assert len(bars) == 2 and all(bar.n == 2 and sum(bar.deltas) == 0 for bar in bars)
    assert log.read_text().count('responses already saved') == 2


def test_real_tqdm_renders_counts_remaining_and_eta(tmp_path, monkeypatch):
    from tqdm.std import tqdm

    output, created = io.StringIO(), []

    def factory(**kwargs):
        bar = tqdm(**{**kwargs, 'file': output, 'mininterval': 0, 'disable': False})
        created.append(bar)
        return bar

    monkeypatch.setitem(sys.modules, 'tqdm.auto', SimpleNamespace(tqdm=factory))
    command = [sys.executable, '-c', "print('B0: 1/2 responses already saved'); "
               "print('arc_challenge: 1/2 problems'); print('arc_challenge: 2/2 problems')"]
    run_eval_logged(command, cwd=tmp_path, log_path=tmp_path / 'log',
                    progress_totals={'arc_challenge': 2}, label='B0')
    assert created[0].n == 2 and created[0].initial == 1
    text = output.getvalue()
    assert '2/2 completed' in text and '0 remaining' in text and 'B0 · Complete' in text
    assert '<00:00' in text and 'problem/s' in text
