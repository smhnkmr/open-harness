from __future__ import annotations

import time
from pathlib import Path

import pytest

from open_harness.backend.local import LocalBackend


def _echo_cmd(shell: str) -> str:
    return "echo hi"


@pytest.fixture
def backend(tmp_path: Path) -> LocalBackend:
    return LocalBackend(root=tmp_path)


def test_execute_echo(backend: LocalBackend) -> None:
    result = backend.execute("echo hi", timeout=10)
    assert result.exit_code == 0
    assert "hi" in result.stdout
    assert result.timed_out is False


def test_execute_captures_stderr(backend: LocalBackend) -> None:
    if backend.shell not in ("bash", "sh"):
        pytest.skip("stderr redirection syntax assumed posix-shell here")
    result = backend.execute("echo oops 1>&2", timeout=10)
    assert "oops" in result.stderr


def test_execute_nonzero_exit(backend: LocalBackend) -> None:
    if backend.shell not in ("bash", "sh"):
        pytest.skip("exit builtin assumed posix-shell here")
    result = backend.execute("exit 3", timeout=10)
    assert result.exit_code == 3
    assert result.timed_out is False


def test_execute_timeout_kills_process(backend: LocalBackend) -> None:
    if backend.shell not in ("bash", "sh"):
        pytest.skip("sleep assumed posix-shell here")
    start = time.time()
    result = backend.execute("sleep 5", timeout=0.5)
    elapsed = time.time() - start
    assert result.timed_out is True
    assert result.exit_code == 124
    # Should not have waited anywhere near the full sleep duration.
    assert elapsed < 4


def test_execute_uses_cwd(backend: LocalBackend, tmp_path: Path) -> None:
    # Redirection is understood by bash, cmd, and PowerShell alike, and lets
    # the OS (not a shell builtin's own path-translation, e.g. MSYS bash's
    # posix-style `pwd`) tell us where the process actually ran.
    sub = tmp_path / "sub"
    sub.mkdir()
    backend.execute("echo marker > marker.txt", timeout=10, cwd=sub)
    assert (sub / "marker.txt").exists()
    assert not (tmp_path / "marker.txt").exists()


def test_execute_env_minimal_plus_extra(backend: LocalBackend) -> None:
    if backend.shell not in ("bash", "sh"):
        pytest.skip("env var expansion assumed posix-shell here")
    result = backend.execute("echo $MY_TEST_VAR", timeout=10, env={"MY_TEST_VAR": "hello"})
    assert "hello" in result.stdout


def test_upload_and_download_roundtrip(backend: LocalBackend) -> None:
    results = backend.upload({"a/b.txt": b"hello world"})
    assert all(r.ok for r in results)
    out = backend.download(["a/b.txt"])
    assert out["a/b.txt"] == b"hello world"


def test_upload_rejects_path_escaping_root(backend: LocalBackend) -> None:
    results = backend.upload({"../escape.txt": b"nope"})
    assert len(results) == 1
    assert results[0].ok is False
    assert results[0].error is not None


def test_download_rejects_path_escaping_root(backend: LocalBackend, tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside_secret.txt"
    outside.write_text("secret", encoding="utf-8")
    try:
        out = backend.download([str(outside)])
        result = out[str(outside)]
        assert not isinstance(result, bytes)
        assert result.ok is False
    finally:
        outside.unlink(missing_ok=True)


def test_download_missing_file_returns_error_result(backend: LocalBackend) -> None:
    out = backend.download(["does/not/exist.txt"])
    result = out["does/not/exist.txt"]
    assert not isinstance(result, bytes)
    assert result.ok is False


def test_shell_detection_prefers_bash_when_present() -> None:
    import shutil

    from open_harness.backend.local import detect_shell

    shell = detect_shell()
    if shutil.which("bash"):
        assert shell == "bash"
    else:
        assert shell in ("powershell", "pwsh", "cmd", "sh")


def test_path_prepend_puts_project_interpreter_first(tmp_path):
    from open_harness.backend.local import LocalBackend

    b = LocalBackend(root=tmp_path, path_prepend=[str(tmp_path / "venvbin")])
    r = b.execute("echo $PATH" if b.shell == "bash" else "echo %PATH%", timeout=10)
    # bash on Windows rewrites C:\x\y as /c/x/y or /tmp/...; compare on the unique leaf
    separator = ":" if b.shell == "bash" else ";"
    first_entry = r.stdout.strip().split(separator)[0]
    assert first_entry.replace("\\", "/").endswith("/venvbin")
