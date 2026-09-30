import sys

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
