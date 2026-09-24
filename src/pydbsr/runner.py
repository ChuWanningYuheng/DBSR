"""Locating and running the DBSR Fortran executables."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

_PKG_BIN = Path(__file__).resolve().parent / "bin"


class DBSRError(RuntimeError):
    """A DBSR program stopped with an error."""

    def __init__(self, program, cwd, message, output_tail=""):
        self.program, self.cwd, self.message, self.output_tail = program, cwd, message, output_tail
        super().__init__(f"{program} failed in {cwd}: {message}\n--- last output ---\n{output_tail}")


def bin_dir() -> Path | None:
    """Directory with the executables: $DBSR_BIN, the installed package, or None (use $PATH)."""
    env = os.environ.get("DBSR_BIN")
    if env:
        return Path(env).expanduser().resolve()
    if _PKG_BIN.is_dir() and any(_PKG_BIN.iterdir()):
        return _PKG_BIN
    return None


def which(program: str) -> str:
    d = bin_dir()
    if d is not None and (d / program).exists():
        return str(d / program)
    p = shutil.which(program)
    if p:
        return p
    raise FileNotFoundError(
        f"DBSR executable {program!r} not found (looked in {d or '$PATH'}). "
        "Install pydbsr with its compiled programs or set DBSR_BIN.")


def available_programs() -> list[str]:
    d = bin_dir()
    return sorted(p.name for p in d.iterdir() if os.access(p, os.X_OK)) if d else []


# Normal termination in gfortran prints nothing or "STOP" / "STOP 0" / "STOP ' '".
_ERROR_PATTERNS = [
    re.compile(r"^\s*STOP\s+(?!0\s*$)\S.*", re.I),
    re.compile(r"^\s*ERROR STOP.*", re.I),
    re.compile(r"Fortran runtime error.*", re.I),
    re.compile(r"Program received signal.*", re.I),
    re.compile(r"segmentation fault", re.I),
]
# messages printed with STOP by upstream codes that are not errors
_BENIGN_STOPS = [re.compile(p, re.I) for p in (r"^\s*STOP\s*$", r"^\s*STOP\s+0\s*$")]


@dataclass
class RunResult:
    program: str
    args: list[str]
    cwd: Path
    returncode: int
    output: str
    seconds: float


def run(program: str, args: Sequence[str] = (), cwd: str | Path = ".", *,
        threads: int | None = None, env: dict | None = None, echo: bool = False,
        log: str | Path | None = None, check: bool = True, timeout: float | None = None,
        mpi: int | None = None, mpirun: str = "mpirun") -> RunResult:
    """Run a DBSR program in ``cwd``.

    Arguments are passed as separate argv entries (no shell), so values with
    blanks (``core=1s 2s 2p``) need no quoting.  ``threads`` sets
    OMP/MKL/OpenBLAS thread counts; ``mpi=n`` launches ``mpirun -np n``.
    Errors (``STOP <message>``, Fortran runtime errors, non-zero exit) raise
    :class:`DBSRError` when ``check`` is true.
    """
    cwd = Path(cwd)
    exe = which(program)
    cmd = [exe, *map(str, args)]
    if mpi:
        cmd = [mpirun, "-np", str(mpi), *cmd]
    e = dict(os.environ if env is None else env)
    if threads is not None:
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            e[k] = str(threads)
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, errors="replace", env=e)
    lines = []
    try:
        for line in proc.stdout:
            lines.append(line)
            if echo:
                sys.stdout.write(line)
                sys.stdout.flush()
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise
    out = "".join(lines)
    if log is not None:
        Path(cwd, log).write_text(f"$ {' '.join(cmd)}\n{out}")
    res = RunResult(program, list(map(str, args)), cwd, proc.returncode, out, time.time() - t0)
    if check:
        msg = error_message(out)
        if proc.returncode != 0 or msg:
            raise DBSRError(program, cwd, msg or f"exit code {proc.returncode}", "".join(lines[-25:]))
    return res


def error_message(output: str) -> str | None:
    for line in output.splitlines():
        if any(p.match(line) for p in _BENIGN_STOPS):
            continue
        for p in _ERROR_PATTERNS:
            m = p.search(line)
            if m:
                return m.group(0).strip()
    return None


# ---------------------------------------------------------------------------
# parallel execution over partial waves
# ---------------------------------------------------------------------------
_COPY_LIMIT = 4 << 20      # files below 4 MB are copied into sandboxes, larger ones linked


def _sandbox(workdir: Path, tag: str) -> tuple[Path, set]:
    """Private directory for one job.

    Small files are copied (programs rewrite some inputs in place, e.g. every
    program using the DBS library rewrites ``knot.dat``); large ones
    (bsw, int_bnk, matrices) are symlinked.  Returns the directory and the
    names of the copied inputs (not moved back afterwards).
    """
    sb = workdir / "_parallel" / tag
    if sb.exists():
        shutil.rmtree(sb)
    sb.mkdir(parents=True)
    copied = set()
    for f in workdir.iterdir():
        # shared-name logs are written by every job: never share them
        if not f.is_file() or f.name.endswith(".log") or ".out." in f.name:
            continue
        if f.stat().st_size < _COPY_LIMIT:
            shutil.copy2(f, sb / f.name)
            copied.add(f.name)
        else:
            (sb / f.name).symlink_to(f.resolve())
    return sb, copied


def _collect(sb: Path, workdir: Path, suffix: str, copied: set):
    """Move files created or changed in the sandbox back; shared-name logs get ``.suffix``."""
    for f in sb.iterdir():
        if f.is_symlink():
            continue
        if f.name in copied:
            src = workdir / f.name
            if src.exists() and src.stat().st_mtime >= f.stat().st_mtime:
                continue                 # unchanged input
            if re.search(r"\.\d{3,}$", f.name) is None:
                continue                 # shared input rewritten by the program (e.g. knot.dat)
        dest = workdir / f.name
        if re.search(r"\.\d{3,}$", f.name) is None and dest.exists():
            dest = workdir / f"{f.name}.{suffix}"
        shutil.move(str(f), dest)
    shutil.rmtree(sb)


class ResourcePool:
    """Blocking budget of CPU cores and memory (GB) shared by concurrent jobs."""

    def __init__(self, cores: int, mem_gb: float):
        import threading
        self.cores, self.mem = cores, mem_gb
        self.free_cores, self.free_mem = cores, mem_gb
        self.cv = threading.Condition()

    def acquire(self, cores: int, mem_gb: float):
        cores = min(cores, self.cores)
        mem_gb = min(mem_gb, self.mem)          # a job larger than the budget runs alone
        with self.cv:
            self.cv.wait_for(lambda: self.free_cores >= cores and self.free_mem >= mem_gb)
            self.free_cores -= cores
            self.free_mem -= mem_gb
        return cores, mem_gb

    def release(self, cores: int, mem_gb: float):
        with self.cv:
            self.free_cores += cores
            self.free_mem += mem_gb
            self.cv.notify_all()


def run_partial_waves(program: str, klsps: Iterable[int], workdir: str | Path, *,
                      args: Sequence[str] = (), jobs: int = 1, threads: int = 1,
                      echo: bool = False, progress: Callable[[str], None] | None = print,
                      mpi: int | None = None) -> list[RunResult]:
    """Run ``program klsp=k`` for every partial wave, ``jobs`` of them at a time.

    Each job runs in its own sandbox directory (so that programs writing
    fixed-name scratch/log files do not collide) and its outputs, which are
    indexed by the partial-wave number (``*.nnn``), are moved back.
    """
    workdir = Path(workdir).resolve()
    klsps = list(klsps)
    if jobs <= 1:
        results = []
        for k in klsps:
            t = time.time()
            results.append(run(program, [f"klsp1={k}", f"klsp2={k}", *args], workdir,
                               threads=threads, echo=echo, log=f"{program}.out.{k:03d}", mpi=mpi))
            if progress:
                progress(f"  {program}: partial wave {k} done ({time.time() - t:.1f} s)")
        return results

    def job(k):
        sb, copied = _sandbox(workdir, f"{program}_{k:03d}")
        try:
            r = run(program, [f"klsp1={k}", f"klsp2={k}", *args], sb, threads=threads,
                    log=f"{program}.out.{k:03d}", mpi=mpi)
        finally:
            _collect(sb, workdir, f"{k:03d}", copied)
        return r

    results = {}
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futs = {pool.submit(job, k): k for k in klsps}
        for fut in as_completed(futs):
            k = futs[fut]
            results[k] = fut.result()
            if progress:
                progress(f"  {program}: partial wave {k} done ({results[k].seconds:.1f} s)")
    try:
        (workdir / "_parallel").rmdir()          # only if empty
    except OSError:
        pass
    return [results[k] for k in klsps]
