"""Target structure: Dirac-Hartree-Fock (dbsr_hf) runs and target states."""
from __future__ import annotations

import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

from . import io
from .atoms import Ion, expand_core
from .constants import AU_EV, AU_CM
from .runner import run

__all__ = ["State", "ConfigSpec", "Target"]


@dataclass
class State:
    """One jj-coupled target state (a pair of files ``name.c`` / ``name.bsw``)."""
    name: str
    config: str               # e.g. '5s2 5p4 5d'
    label: str                # jj label from dbsr_hf / dbsr_ci
    two_j: int
    parity: int
    energy: float             # total energy, a.u.
    source: str = ""          # HF run it came from
    exp_energy_cm: float | None = None   # experimental excitation energy (cm-1), e.g. NIST
    nist_label: str | None = None

    @property
    def J(self) -> float:
        return self.two_j / 2

    @property
    def g(self) -> int:
        return self.two_j + 1

    @property
    def Jstr(self) -> str:
        return f"{self.two_j // 2}" if self.two_j % 2 == 0 else f"{self.two_j}/2"

    def __str__(self):
        p = "e" if self.parity > 0 else "o"
        return f"{self.name:<18s} {self.config:<16s} J={self.Jstr:<4s} {p}  E={self.energy:.8f}"


@dataclass
class ConfigSpec:
    conf: str                          # '5s2 5p4 5d'
    name: str
    varied: str = "all"                # dbsr_hf varied=
    term: str = "LS"                   # LS | AV | jj
    jj_varied: str = "none"            # varied= for the second (jj) step
    inp: str | None = None             # input .bsw
    ci: bool = False                   # run dbsr_breit3 + dbsr_ci3 afterwards
    extra: dict = field(default_factory=dict)


def _default_name(conf: str) -> str:
    """'5s2 5p4 5d' -> '5p4_5d1' (closed shells dropped when not alone)."""
    shells = io.parse_config(conf)
    parts = [f"{io.shell_name(n, l)}{q}" for n, l, q in shells
             if not (q == 2 * (2 * l + 1) and len(shells) > 1)]
    return "_".join(parts) or "conf"


class Target:
    """Builder for the target (N-electron) states.

    Example
    -------
    >>> tg = Target(Ion("Xe", 1), core="[Kr]4d10", workdir="xe_target")
    >>> tg.add("5s2 5p5")                  # reference: all orbitals optimised
    >>> tg.add("5s 5p6")                   # uses the reference orbitals
    >>> tg.add("5s2 5p4 5d")               # only 5d is optimised
    >>> tg.add("5s2 5p4 6s")
    >>> tg.compute(jobs=4)
    >>> tg.states                          # all J-states of all configurations

    ``grid={'rmax': 60.0, 'hmax': 0.5}`` sets the B-spline grid (``knot.dat``):
    ``rmax`` becomes the R-matrix radius of the scattering calculation and
    ``hmax`` (max. step at large r) limits the highest electron energy that
    the continuum basis can describe (roughly E_max ~ (pi/hmax)^2/2 a.u.
    / a few).  The target orbitals must vanish at ``rmax``.
    """

    def __init__(self, ion: Ion, core: str, workdir: str | Path = "target",
                 grid: dict | None = None, max_it: int = 75, hf_args: dict | None = None,
                 echo: bool = False):
        self.ion = ion
        self.core_shells = expand_core(core)
        self.workdir = Path(workdir)
        self.grid = grid or {}
        self.max_it = max_it
        self.hf_args = dict(hf_args or {})
        self.echo = echo
        self.specs: list[ConfigSpec] = []
        self.states: list[State] = []

    # ------------------------------------------------------------------ setup
    @property
    def core(self) -> str:
        return " ".join(self.core_shells)

    def add(self, conf: str, name: str | None = None, varied: str | Sequence[str] | None = None,
            term: str = "LS", jj_varied: str = "none", ci: bool = False, **hf_args) -> ConfigSpec:
        """Add a configuration (only peel shells, e.g. ``'5s2 5p4 6p'``).

        The first configuration is the reference: all its orbitals are
        optimised (``varied='all'``).  For the others, by default only the
        orbitals absent from the reference are optimised and the reference
        orbitals are used as input, which keeps the orbital set compact and
        consistent (as in the DBSR examples).
        """
        n_core = sum(2 * (2 * io.L_SYMBOLS.index(s[-1]) + 1) for s in self.core_shells)
        n = n_core + io.config_nelectrons(conf)
        if n != self.ion.nelc:
            raise ValueError(f"{conf!r}: {n} electrons with the core, {self.ion} has {self.ion.nelc}")
        name = name or _default_name(conf)
        if any(s.name == name for s in self.specs):
            raise ValueError(f"duplicate name {name!r}")
        ref = self.specs[0] if self.specs else None
        if varied is None:
            if ref is None:
                varied = "all"
            else:
                ref_sh = {(n_, l) for n_, l, _ in io.parse_config(ref.conf)}
                new = [io.shell_name(n_, l) for n_, l, _ in io.parse_config(conf) if (n_, l) not in ref_sh]
                varied = ",".join(new) if new else "none"
        elif not isinstance(varied, str):
            varied = ",".join(varied)
        spec = ConfigSpec(conf=io.pretty_conf(conf), name=name, varied=varied, term=term,
                          jj_varied=jj_varied, inp=None if ref is None else f"{ref.name}.bsw",
                          ci=ci, extra=hf_args)
        self.specs.append(spec)
        return spec

    # ------------------------------------------------------------------ runs
    def _hf_args(self, spec: ConfigSpec) -> list[str]:
        peel = " ".join(io.shell_name(n, l) for n, l, _ in io.parse_config(spec.conf))
        args = [spec.name, f"z={self.ion.z}", f"core={self.core}", f"peel={peel}",
                f"conf={io.to_dbsr_conf(spec.conf)}", f"term={spec.term}",
                f"varied={spec.varied}", f"max_it={self.max_it}"]
        if spec.inp:
            args.append(f"inp={spec.inp}")
        args += self._grid_args()
        args += [f"{k}={v}" for k, v in {**self.hf_args, **spec.extra}.items()]
        return args

    def _grid_args(self) -> list[str]:
        # dbsr_hf reads the grid from knot.dat or from the command line (not from name.knot)
        return [f"{k}={v}" for k, v in self.grid.items()]

    def _run_spec(self, spec: ConfigSpec):
        """dbsr_hf (LS, then jj) in a private subdirectory: dbsr_hf uses fixed-name
        scratch files, so parallel runs must not share a directory."""
        wd = self.workdir
        sb = wd / "_hf" / spec.name
        if sb.exists():
            shutil.rmtree(sb)
        sb.mkdir(parents=True)
        if spec.inp:
            shutil.copy(wd / spec.inp, sb / spec.inp)
        run("dbsr_hf", self._hf_args(spec), sb, echo=self.echo, log=f"{spec.name}.hf1.out")
        jj = [spec.name, "term=jj", f"varied={spec.jj_varied}", f"max_it={self.max_it}", *self._grid_args()]
        run("dbsr_hf", jj, sb, echo=self.echo, log=f"{spec.name}.hf2.out")
        if spec.ci:
            run("dbsr_breit3", [f"{spec.name}.c"], sb, echo=self.echo, log=f"{spec.name}.breit.out")
            run("dbsr_ci3", [spec.name], sb, echo=self.echo, log=f"{spec.name}.ci.out")
        for f in sb.iterdir():
            if f.name.startswith(spec.name + "."):
                shutil.move(str(f), wd / f.name)
        shutil.rmtree(sb)

    def compute(self, jobs: int = 1, force: bool = False,
                progress: Callable[[str], None] | None = print) -> list[State]:
        """Run dbsr_hf for all configurations and split the solutions into states."""
        if not self.specs:
            raise ValueError("no configurations: use Target.add()")
        wd = self.workdir
        wd.mkdir(parents=True, exist_ok=True)
        ref = self.specs[0]

        def need(spec):
            return force or not (wd / f"{spec.name}.j").exists()

        if need(ref):
            if progress:
                progress(f"dbsr_hf: {ref.name} ({ref.conf}), reference")
            self._run_spec(ref)
        others = [s for s in self.specs[1:] if need(s)]
        if others:
            if progress:
                progress("dbsr_hf: " + ", ".join(f"{s.name} ({s.conf})" for s in others))
            with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
                list(pool.map(self._run_spec, others))
        self.states = self._split_states()
        self.save()
        return self.states

    def _split_states(self) -> list[State]:
        states = []
        for spec in self.specs:
            jfile, cfile = self.workdir / f"{spec.name}.j", self.workdir / f"{spec.name}.c"
            sols = io.read_j(jfile)
            cf = io.CFile.read(cfile)
            count: dict[int, int] = {}
            for sol in sorted(sols, key=lambda s: s.energy):
                # 2J/parity from the CSFs themselves (the J column of .j files is not reliable)
                two_j, parity = io.csf_jp(cf.csfs[sol.ic1 - 1][2])
                sol.two_j = two_j
                count[two_j] = count.get(two_j, 0) + 1
                name = f"{spec.name}_j{two_j}_{count[two_j]}"
                io.write_state(spec.name, sol, cf, self.workdir / f"{spec.name}.bsw", self.workdir / name)
                states.append(State(name, spec.conf, sol.label, two_j, parity, sol.energy, spec.name))
        states.sort(key=lambda s: s.energy)
        return states

    # --------------------------------------------------------------- queries
    @property
    def ground(self) -> State:
        return min(self.states, key=lambda s: s.energy)

    def excitation_ev(self, s: State) -> float:
        return (s.energy - self.ground.energy) * AU_EV

    def select(self, max_states: int | None = None, emax_ev: float | None = None,
               configs: Iterable[str] | None = None,
               where: Callable[[State], bool] | None = None) -> list[State]:
        """Subset of states ordered by energy."""
        out = list(self.states)
        if configs is not None:
            keep = {io.pretty_conf(c) for c in configs}
            out = [s for s in out if s.config in keep]
        if emax_ev is not None:
            out = [s for s in out if self.excitation_ev(s) <= emax_ev]
        if where is not None:
            out = [s for s in out if where(s)]
        return out[:max_states] if max_states else out

    def table(self, states: Sequence[State] | None = None) -> str:
        states = self.states if states is None else states
        e0 = self.ground.energy
        rows = [f"{'#':>3} {'name':<18} {'config':<16} {'J':<4} {'P':1} {'E (a.u.)':>16} {'dE (eV)':>9}"
                f" {'NIST (eV)':>9}"]
        for i, s in enumerate(states, 1):
            nist = f"{s.exp_energy_cm / AU_CM * AU_EV:9.4f}" if s.exp_energy_cm is not None else ""
            rows.append(f"{i:3d} {s.name:<18} {s.config:<16} {s.Jstr:<4} {'e' if s.parity > 0 else 'o'}"
                        f" {s.energy:16.8f} {(s.energy - e0) * AU_EV:9.4f} {nist:>9}")
        return "\n".join(rows)

    # ----------------------------------------------------------- persistence
    def save(self):
        data = {"ion": [self.ion.element, self.ion.charge], "core": self.core, "grid": self.grid,
                "specs": [asdict(s) for s in self.specs], "states": [asdict(s) for s in self.states]}
        (self.workdir / "pydbsr_target.json").write_text(json.dumps(data, indent=1))

    @classmethod
    def load(cls, workdir: str | Path) -> "Target":
        workdir = Path(workdir)
        data = json.loads((workdir / "pydbsr_target.json").read_text())
        tg = cls(Ion(*data["ion"]), data["core"], workdir, grid=data.get("grid"))
        tg.specs = [ConfigSpec(**s) for s in data["specs"]]
        tg.states = [State(**s) for s in data["states"]]
        return tg

    @classmethod
    def from_files(cls, ion: Ion, workdir: str | Path, names: Sequence[str] | None = None) -> "Target":
        """Wrap existing state files ``name.c``/``name.bsw`` (e.g. from MCHF/CI runs)."""
        workdir = Path(workdir)
        names = names or sorted(p.stem for p in workdir.glob("*.c") if (workdir / f"{p.stem}.bsw").exists())
        tg = cls(ion, "", workdir)
        for nm in names:
            cf = io.CFile.read(workdir / f"{nm}.c")
            two_j, parity = _jp_from_cfile(workdir / f"{nm}.c")
            tg.states.append(State(nm, "", "", two_j, parity, cf.energy or 0.0, "file"))
        tg.states.sort(key=lambda s: s.energy)
        return tg


def _jp_from_cfile(path) -> tuple[int, int]:
    """2J and parity of the first CSF of a .c file."""
    return io.csf_jp(io.CFile.read(path).csfs[0][2])
