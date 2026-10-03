import sys
from types import SimpleNamespace

import pytest

from common.process import run_logged


def test_subprocess_stdout_stderr_and_actual_error_are_visible(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-sensitive-token")
    path = tmp_path / "console.log"
    command = [
        sys.executable,
        "-c",
        "import sys,os; print('starting'); "
        "print('actual CUDA failure ' + os.environ['HF_TOKEN'], file=sys.stderr); sys.exit(1)",
    ]
    with pytest.raises(RuntimeError, match="actual CUDA failure") as error:
        run_logged(command, cwd=tmp_path, log_path=path)
    assert "actual CUDA failure" in capsys.readouterr().out
    assert "starting" in path.read_text()
    assert "test-sensitive-token" not in path.read_text() + str(error.value)
    assert "[REDACTED]" in path.read_text()


def test_success_returns_log_and_keeps_prior_attempt(tmp_path):
    path = tmp_path / "console.log"
    command = [sys.executable, "-c", "print('completed')"]
    assert run_logged(command, cwd=tmp_path, log_path=path) == path
    run_logged(command, cwd=tmp_path, log_path=path)
    assert path.read_text().count("completed") == 2


def test_notebook_progress_updates_in_place_and_preserves_error_log(tmp_path, monkeypatch, capsys):
    created, updates = [], []
    handle = SimpleNamespace(update=lambda data, **kw: updates.append(data["text/plain"]))

    def display(data, **kwargs):
        assert kwargs == {"raw": True, "display_id": True}
        created.append(data["text/plain"])
        return handle

    monkeypatch.setitem(sys.modules, "IPython", SimpleNamespace(get_ipython=lambda: SimpleNamespace(kernel=True)))
    monkeypatch.setitem(sys.modules, "IPython.display", SimpleNamespace(display=display))
    monkeypatch.setenv("HF_TOKEN", "private-token")
    lines = ("\rProcessed prompts: 0%|          | 0/128 [00:00<?, ?it/s]\n\x1b[A"
             "\rProcessed prompts: 1%|          | 1/128 [00:02<05:57, 2.81s/it]\n\x1b[A"
             "\rProcessed prompts: 2%|          | 2/128 [00:02<02:34, 1.23s/it]\n"
             "DAPO update 1: 2/16 mixed groups retained\nactual failure private-token\n")
    command = [sys.executable, "-c", f"import sys; sys.stdout.write({lines!r}); sys.exit(1)"]
    path = tmp_path / "console.log"
    with pytest.raises(RuntimeError, match="actual failure") as error:
        run_logged(command, cwd=tmp_path, log_path=path, compact_progress=True)
    assert len(created) == 1 and len(updates) == 2
    assert "2/128" in updates[-1]
    visible = capsys.readouterr().out
    assert "Processed prompts" not in visible and "DAPO update 1" in visible
    assert "actual failure [REDACTED]" in visible
    assert path.read_text().count("Processed prompts") == 3
    assert "private-token" not in path.read_text() + str(error.value)


@pytest.mark.parametrize("exit_code,completed", [(0, 5), (1, 4)])
def test_progress_tracks_problem_counts_and_does_not_complete_failures(
    tmp_path, monkeypatch, exit_code, completed
):
    class Bar:
        def __init__(self, total, **kwargs):
            self.total, self.n, self.closed = total, 0, False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.closed = True

        def update(self, delta):
            self.n += delta

        def set_description(self, *args, **kwargs):
            pass

    bar = Bar(5)
    monkeypatch.setitem(sys.modules, "tqdm.auto", SimpleNamespace(tqdm=lambda **kw: bar))
    # The first dataset was resumed; the second reports one completed problem.
    command = [sys.executable, "-c", f"print('second: 1/2 problems'); exit({exit_code})"]
    kwargs = dict(
        cwd=tmp_path, log_path=tmp_path / "log", progress_totals={"first": 3, "second": 2}
    )
    if exit_code:
        with pytest.raises(RuntimeError):
            run_logged(command, **kwargs)
    else:
        run_logged(command, **kwargs)
    assert bar.n == completed and bar.closed
    assert "second: 1/2 problems" in (tmp_path / "log").read_text()
