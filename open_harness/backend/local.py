"""Local filesystem/subprocess backend.

Spec: open-harness-spec.md section 6.5.

`LocalBackend` runs commands via a real shell (bash preferred, else
PowerShell, else cmd) and confines file I/O to a root directory. It does not
enforce permissions -- the policy engine does that before a tool call reaches
here.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
from pathlib import Path

from open_harness.backend.base import Backend, ExecResult, FileOpResult

# Environment variables inherited by default into every subprocess, on top of
# whatever the caller passes in `env`. Kept minimal and deliberate.
_INHERITED_ENV_KEYS = (
    "PATH",
    "HOME",
    "USERPROFILE",
    "SYSTEMROOT",
    "SystemRoot",
    "TEMP",
    "TMP",
    "COMSPEC",
)


def detect_shell() -> str:
    """Pick a shell: bash on PATH, else PowerShell, else cmd."""
    for candidate in ("bash", "powershell", "pwsh", "cmd"):
        found = shutil.which(candidate)
        if found:
            return candidate
    # Last resort: cmd always exists on Windows; sh always exists on POSIX.
    return "cmd" if os.name == "nt" else "sh"


def _shell_argv(shell: str, cmd: str) -> list[str]:
    """Build the argv used to invoke `cmd` under `shell`."""
    resolved = shutil.which(shell) or shell
    name = Path(resolved).name.lower()
    if name.startswith(("bash", "sh")):
        # Non-login, non-interactive: no profile sourcing. This keeps
        # startup latency low and deterministic, and lets bash exec()
        # straight into a single command instead of forking through a
        # profile-script chain first -- important for prompt, reliable
        # kill-on-timeout.
        return [resolved, "-c", cmd]
    if name.startswith(("pwsh", "powershell")):
        return [resolved, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", cmd]
    if name.startswith("cmd"):
        return [resolved, "/d", "/s", "/c", cmd]
    # Unknown shell: assume POSIX-sh-like invocation.
    return [resolved, "-c", cmd]


def _minimal_env(extra: dict[str, str] | None) -> dict[str, str]:
    env: dict[str, str] = {}
    for key in _INHERITED_ENV_KEYS:
        value = os.environ.get(key)
        if value is not None:
            env[key] = value
    if extra:
        env.update(extra)
    return env


class LocalBackend(Backend):
    """Executes commands on the local machine and reads/writes local files
    under `root`."""

    def __init__(self, root: Path, shell: str | None = None, path_prepend: list[str] | None = None) -> None:
        self.root = Path(root).resolve()
        self.shell = shell or detect_shell()
        # Directories placed ahead of PATH for every command, so `python`,
        # `pytest` and `ruff` resolve to the project's interpreter rather than
        # whatever the system has first.
        self.path_prepend = [str(p) for p in (path_prepend or []) if p]

    # ------------------------------------------------------------------ exec

    def execute(
        self,
        cmd: str,
        *,
        timeout: float,
        env: dict[str, str] | None = None,
        cwd: Path | None = None,
    ) -> ExecResult:
        argv = _shell_argv(self.shell, cmd)
        run_env = _minimal_env(env)
        if self.path_prepend:
            run_env["PATH"] = os.pathsep.join([*self.path_prepend, run_env.get("PATH", "")])
        work_dir = str(Path(cwd).resolve()) if cwd is not None else str(self.root)

        popen_kwargs: dict[str, object] = {}
        if os.name == "nt":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True

        proc = subprocess.Popen(
            argv,
            cwd=work_dir,
            env=run_env,
            stdin=subprocess.DEVNULL,   # commands never read the harness's stdin (it is the client pipe)
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **popen_kwargs,
        )

        timed_out = False
        try:
            stdout_b, stderr_b = proc.communicate(timeout=timeout)
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            self._kill_tree(proc)
            stdout_b, stderr_b = proc.communicate()
            timed_out = True
            exit_code = 124

        stdout = stdout_b.decode("utf-8", errors="replace")
        stderr = stderr_b.decode("utf-8", errors="replace")
        return ExecResult(stdout=stdout, stderr=stderr, exit_code=exit_code, timed_out=timed_out)

    @staticmethod
    def _kill_tree(proc: subprocess.Popen) -> None:
        """Kill the process and any children it spawned."""
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
        try:
            proc.kill()
        except OSError:
            pass

    # -------------------------------------------------------------- fs paths

    def _resolve_under_root(self, raw: str) -> Path | None:
        """Resolve `raw` (absolute, or relative to root) and reject anything
        that escapes `root` (via `..`, symlinks, or an unrelated absolute
        path). Returns None on escape or an unresolvable path."""
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        try:
            resolved = candidate.resolve()
        except OSError:
            return None
        try:
            resolved.relative_to(self.root)
        except ValueError:
            return None
        return resolved

    def upload(self, files: dict[str, bytes]) -> list[FileOpResult]:
        results: list[FileOpResult] = []
        for raw_path, data in files.items():
            resolved = self._resolve_under_root(raw_path)
            if resolved is None:
                results.append(FileOpResult(path=raw_path, ok=False, error="path escapes backend root"))
                continue
            try:
                resolved.parent.mkdir(parents=True, exist_ok=True)
                resolved.write_bytes(data)
                results.append(FileOpResult(path=raw_path, ok=True))
            except OSError as exc:
                results.append(FileOpResult(path=raw_path, ok=False, error=str(exc)))
        return results

    def download(self, paths: list[str]) -> dict[str, bytes | FileOpResult]:
        out: dict[str, bytes | FileOpResult] = {}
        for raw_path in paths:
            resolved = self._resolve_under_root(raw_path)
            if resolved is None:
                out[raw_path] = FileOpResult(path=raw_path, ok=False, error="path escapes backend root")
                continue
            try:
                out[raw_path] = resolved.read_bytes()
            except OSError as exc:
                out[raw_path] = FileOpResult(path=raw_path, ok=False, error=str(exc))
        return out
