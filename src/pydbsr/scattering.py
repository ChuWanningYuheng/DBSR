"""Inner-region R-matrix calculation: dbsr_prep3 -> conf3 -> breit3 -> mat3 -> hd3."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Callable, Sequence

from . import io
from .atoms import Ion
from .runner import run, run_partial_waves
from .structure import State, Target

__all__ = ["Scattering"]

STEPS = ("prep", "conf", "breit", "mat", "hd")


class Scattering:
    """Close-coupling e + target calculation with DBSR.

    Parameters
    ----------
    target : Target or list of State
        The target calculation, or the states to include.
    states : list of State, optional
        Subset of ``target.states`` to include (e.g. ``tg.select(max_states=20)``).
    workdir : path
        Directory of the scattering run.  Target files are copied there.
    jmax : float
        Maximal total angular momentum J of the (N+1)-electron system.
        Partial waves J = Jmin..Jmax of both parities are generated.
    partial_waves : list of (2J, parity), optional
        Explicit list instead of ``jmax``.
    The B-spline grid (R-matrix radius ``rmax``, step ``hmax``) is the one of
    the target calculation: set it with ``Target(..., grid={'rmax': .., 'hmax': ..})``
    (DBSR needs the target orbitals on the same grid).
    params : dict
        Extra ``name = value`` entries for ``dbsr_par`` (read by all programs;
        e.g. ``{'mbreit': 1}``; see the DBSR manual).
    exp_energies : bool
        Use experimental target energies (``State.exp_energy_cm``, e.g. from
        :func:`pydbsr.nist.assign`) for the thresholds (``dbsr_hd iexp=1``).
    jobs, threads : int
        Partial waves run ``jobs`` at a time, each with ``threads`` OpenMP/BLAS
        threads.  ``mpi=n`` uses the MPI programs where available.
    """

    def __init__(self, target: Target | Sequence[State], workdir: str | Path, *,
                 states: Sequence[State] | None = None, ion: Ion | None = None, jmax: float | None = None, jmin: float | None = None,
                 partial_waves: Sequence[tuple[int, int]] | None = None,
                 params: dict | None = None, prep_args: dict | None = None,
                 perturbers: Sequence[tuple[int, str]] = (),
                 exp_energies: bool = True, jobs: int = 1, threads: int = 1,
                 mpi: int | None = None, echo: bool = False,
                 progress: Callable[[str], None] | None = print):
        if isinstance(target, Target):
            self.target_dir = target.workdir
            self.ion = target.ion
            self.states = list(target.states if states is None else states)
            self.ref_knot = target.workdir / f"{target.specs[0].name}.knot" if target.specs else None
        else:
            self.states = list(target if states is None else states)
            if ion is None:
                raise ValueError("ion= is required when a list of states is given")
            self.ion = ion
            self.target_dir = None
            self.ref_knot = None
        if ion is not None:
            self.ion = ion
        if not self.states:
            raise ValueError("no target states")
        self.workdir = Path(workdir)
        if partial_waves is None:
            if jmax is None:
                raise ValueError("give jmax= or partial_waves=")
            partial_waves = io.partial_waves(self.ion.nelc, jmax, jmin)
        self.partial_waves = [tuple(pw) for pw in partial_waves]
        self.params = dict(params or {})
        self.prep_args = dict(prep_args or {})
        self.perturbers = list(perturbers)
        self.exp_energies = exp_energies
        self.jobs, self.threads, self.mpi = jobs, threads, mpi
        self.echo = echo
        self.progress = progress or (lambda s: None)

    # ----------------------------------------------------------- helpers
    @property
    def nlsp(self) -> int:
        return len(self.partial_waves)

    def _log(self, msg):
        self.progress(msg)

    def _prog(self, name: str) -> str:
        """MPI variant if requested and installed."""
        if self.mpi:
            from .runner import bin_dir
            d = bin_dir()
            if d is not None and (d / f"{name}_mpi").exists():
                return f"{name}_mpi"
        return name

    def _find(self, name: str, suffix: str) -> Path:
        for d in [self.target_dir, self.workdir]:
            if d is not None and (Path(d) / f"{name}{suffix}").exists():
                return Path(d) / f"{name}{suffix}"
        raise FileNotFoundError(f"{name}{suffix}")

    # ----------------------------------------------------------- set-up
    def prepare(self):
        """Copy target files and write knot.dat, target_jj and dbsr_par."""
        wd = self.workdir
        wd.mkdir(parents=True, exist_ok=True)
        for s in self.states:
            for suf in (".c", ".bsw"):
                src = self._find(s.name, suf)
                if src.resolve() != (wd / src.name).resolve():
                    shutil.copy(src, wd / src.name)
        for _, pert in self.perturbers:
            for suf in (".c", ".bsw"):
                src = Path(pert + suf) if Path(pert + suf).exists() else self._find(Path(pert).name, suf)
                shutil.copy(src, wd / src.name)
        # B-spline grid
        knot = wd / "knot.dat"
        src = self.ref_knot if self.ref_knot and self.ref_knot.exists() else None
        if src is None and knot.exists():
            src = knot
        if src is None:
            raise FileNotFoundError("no knot file: run the target calculation first or put knot.dat "
                                    f"into {wd}")
        if src.resolve() != knot.resolve():
            shutil.copy(src, knot)
        # target_jj with names only; dbsr_prep3 fills in the rest
        io.TargetJJ(title=f"e + {self.ion.name} scattering (pydbsr)", nz=self.ion.z, nelc=self.ion.nelc,
                    states=[io.TargetEntry(s.name) for s in self.states],
                    partial_waves=[(tj, p, None) for tj, p in self.partial_waves],
                    perturbers=[(k, Path(n).name) for k, n in self.perturbers]).write(wd / "target_jj")
        io.write_par(wd / "dbsr_par", self.params)
        self._save()

    def _save(self):
        data = {"ion": [self.ion.element, self.ion.charge], "partial_waves": self.partial_waves,
                "states": [s.__dict__ for s in self.states], "params": self.params}
        (self.workdir / "pydbsr_scattering.json").write_text(json.dumps(data, indent=1))

    @classmethod
    def load(cls, workdir, **kw) -> "Scattering":
        workdir = Path(workdir)
        data = json.loads((workdir / "pydbsr_scattering.json").read_text())
        states = [State(**s) for s in data["states"]]
        return cls(states, workdir, ion=Ion(*data["ion"]),
                   partial_waves=[tuple(p) for p in data["partial_waves"]],
                   params=data.get("params"), **kw)

    # ----------------------------------------------------------- steps
    def run_prep(self):
        args = [f"{k}={v}" for k, v in self.prep_args.items()]
        self._log("dbsr_prep3")
        run("dbsr_prep3", args, self.workdir, echo=self.echo, log="dbsr_prep3.out")
        self._write_thresholds()

    def run_conf(self):
        self._log("dbsr_conf3")
        if self.mpi and self._prog("dbsr_conf3") != "dbsr_conf3":
            run("dbsr_conf3_mpi", [], self.workdir, echo=self.echo, log="dbsr_conf3.out", mpi=self.mpi)
        else:
            run("dbsr_conf3", [], self.workdir, echo=self.echo, log="dbsr_conf3.out")

    def _pw(self, program, args=()):
        klsps = range(1, self.nlsp + 1)
        self._log(f"{program}: {self.nlsp} partial waves, jobs={self.jobs}, threads={self.threads}")
        prog = self._prog(program)
        if prog != program:     # MPI: one partial wave at a time, all ranks
            return run_partial_waves(prog, klsps, self.workdir, args=args, jobs=1,
                                     threads=self.threads, echo=self.echo, progress=self._log, mpi=self.mpi)
        return run_partial_waves(program, klsps, self.workdir, args=args, jobs=self.jobs,
                                 threads=self.threads, echo=self.echo, progress=self._log)

    def run_breit(self, **kw):
        return self._pw("dbsr_breit3", [f"{k}={v}" for k, v in kw.items()])

    def run_mat(self, **kw):
        return self._pw("dbsr_mat3", [f"{k}={v}" for k, v in kw.items()])

    def run_hd(self, itype: int = 0, **kw):
        args = [f"itype={itype}"]
        if self.exp_energies and (self.workdir / "thresholds").exists():
            args.append("iexp=1")
        args += [f"{k}={v}" for k, v in kw.items()]
        return self._pw("dbsr_hd3", args)

    def run(self, steps: Sequence[str] = STEPS, **hd_args):
        """Run the inner-region chain (default: all steps)."""
        if "prep" in steps:
            self.prepare()
            self.run_prep()
        if "conf" in steps:
            self.run_conf()
        if "breit" in steps:
            self.run_breit()
        if "mat" in steps:
            self.run_mat()
        if "hd" in steps:
            self.run_hd(**hd_args)
        return self

    # ----------------------------------------------------------- thresholds
    def target_table(self) -> io.TargetJJ:
        """target_jj as filled by dbsr_prep3 (states sorted by energy)."""
        return io.TargetJJ.read(self.workdir / "target_jj")

    def _write_thresholds(self):
        path = self.workdir / "thresholds"
        if path.exists():
            path.unlink()
        if not self.exp_energies:
            return
        by_name = {s.name: s for s in self.states}
        tj = self.target_table()
        exp = [by_name[t.name].exp_energy_cm if t.name in by_name else None for t in tj.states]
        if all(e is None for e in exp):
            return
        if any(e is None for e in exp):
            missing = [t.name for t, e in zip(tj.states, exp) if e is None]
            raise ValueError(f"experimental energies missing for {missing}; "
                             "assign all of them or use exp_energies=False")
        e0 = exp[0]
        lines = [f"{e - e0:.4f}" for e in exp] + ["unit = cm", "it = 1"]
        path.write_text("\n".join(lines) + "\n")
        self._log(f"thresholds: experimental energies for {len(exp)} target states (dbsr_hd iexp=1)")

    def call(self, program: str, *args, **kw):
        """Run any DBSR program/utility in the work directory, e.g.
        ``sc.call('dbsr_mult3', 'cfg.001', 'cfg.004', 'E1')`` or
        ``sc.call('dbsr_dmat3', 'cfg.001', 'cfg.004', 'b', 'b')``."""
        kw.setdefault("echo", self.echo)
        kw.setdefault("log", f"{program}.out")
        return run(program, [str(a) for a in args], self.workdir, **kw)

    # ----------------------------------------------------------- results
    def h_files(self) -> list[Path]:
        return [self.workdir / f"h.{k:03d}" for k in range(1, self.nlsp + 1)
                if (self.workdir / f"h.{k:03d}").exists()]

    def h_target_names(self) -> list[str]:
        """Target-state names in the order of the h-files (as ``dbsr_hd`` writes them:
        target_jj order, re-sorted by the experimental energies when ``iexp=1``)."""
        names = [t.name for t in self.target_table().states]
        thr = self.workdir / "thresholds"
        if self.exp_energies and thr.exists():
            e = [float(x) for x in thr.read_text().split()[:len(names)]]
            names = [n for _, _, n in sorted(zip(e, range(len(names)), names))]
        return names

    def outer(self, **kw):
        """Outer region: collision strengths, see :class:`pydbsr.outer.OuterRegion`."""
        from .outer import OuterRegion
        kw.setdefault("names", self.h_target_names())
        return OuterRegion.from_files(self.h_files(), **kw)

    def bound_states(self, msol: int = 100, **kw) -> list[dict]:
        """Bound (N+1)-electron states: dbsr_hd itype=-1 + dbound_tab."""
        self.run_hd(itype=-1, msol=msol, **kw)
        run("dbound_tab", [], self.workdir, log="dbound_tab.out")
        return read_dbound_tab(self.workdir / "dbound_tab")


def read_dbound_tab(path) -> list[dict]:
    out = []
    for line in Path(path).read_text().splitlines():
        f = line.split()
        if len(f) >= 12 and f[0].isdigit() and f[1].isdigit():
            out.append(dict(klsp=int(f[0]), sol=int(f[1]), label=f[2], two_j=int(f[3]), parity=int(f[4]),
                            E_Ry=float(f[5]), E_eV=float(f[6]), E_cm=float(f[7]), E_au=float(f[8]),
                            E_bind=float(f[9]), target=int(f[10]), channel=int(f[11])))
    return out
