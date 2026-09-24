"""Target structure: Dirac-Hartree-Fock (dbsr_hf) runs and target states."""
from __future__ import annotations

import json
import re
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
    nist_no: int | None = None           # NIST level number (increasing energy)
    configs: list | None = None          # all configurations of its (CI) calculation

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
    correlation: str = ""              # correlation configurations (' + '-separated)
    mchf_varied: str = ""              # orbitals re-optimised by dbsr_mchf
    mchf_max_it: int = 100


def _confs(spec: ConfigSpec) -> list[str]:
    return [c.strip() for c in spec.conf.split(" + ")]


def _all_confs(spec: ConfigSpec) -> list[str]:
    """Physical + correlation configurations."""
    return _confs(spec) + ([c.strip() for c in spec.correlation.split(" + ")] if spec.correlation else [])


def _n_physical(spec: ConfigSpec, cf) -> dict:
    """Number of CSFs of the physical configurations in every J block {block: n}."""
    phys = [tuple(sorted(((n, l), q) for n, l, q in io.parse_config(c))) for c in _confs(spec)]
    shells = {sh for key in phys for sh, _ in key}
    out = {}
    for ib, (a, b) in enumerate(_j_blocks(cf), 1):
        n = 0
        for k in range(a - 1, b):
            occ = _csf_config(cf.csfs[k][0])
            if tuple(sorted((sh, q) for sh, q in occ if sh in shells)) in phys and \
                    all(sh in shells for sh, q in occ if q):
                n += 1
        out[ib] = n
    return out


def _csf_config(line: str) -> tuple:
    """Non-relativistic occupations of a jj CSF line '  5s ( 2)  5p-( 2)  5p ( 4)'."""
    occ: dict = {}
    for n, l, q in re.findall(r"(\d+)([a-z])-?\s*\(\s*(\d+)\)", line):
        key = (int(n), io.L_SYMBOLS.index(l))
        occ[key] = occ.get(key, 0) + int(q)
    return tuple(sorted((k, q) for k, q in occ.items() if q))


def _dominant_config(sol, cf, confs: list[str]) -> str:
    """Configuration (from ``confs``) with the largest weight in a solution."""
    if len(confs) == 1:
        return confs[0]
    keys = {}
    for c in confs:
        keys[tuple(sorted(((n, l), q) for n, l, q in io.parse_config(c)))] = c
    w: dict = {}
    for k, c in enumerate(sol.coefs):
        occ = _csf_config(cf.csfs[sol.ic1 - 1 + k][0])
        # compare only the shells that appear in the candidate configurations
        for key, name in keys.items():
            shells = {sh for sh, _ in key}
            if tuple(sorted((sh, q) for sh, q in occ if sh in shells)) == key:
                w[name] = w.get(name, 0.0) + c * c
    return max(w, key=w.get) if w else confs[0]


def _core_from_label(s) -> str | None:
    """Parent core from a NIST label: '5p4(1D2)5d 2[2] J=5/2' -> '(1D2)',
    '5s2.5p3.(2P*).6p 1D' -> '(2P*)'; None if there is none."""
    m = re.search(r"\(([^()]*)\)", s.nist_label or "")
    return f"({m.group(1)})" if m else None


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", name) or "g"


def _j_blocks(cf) -> list[tuple[int, int]]:
    """J blocks of a c-file in file order: [(first, last)] CSF indices (1-based)."""
    js = [io.csf_jp(c[2]) for c in cf.csfs]
    out, start = [], 0
    for i in range(1, len(js) + 1):
        if i == len(js) or js[i] != js[start]:
            out.append((start + 1, i))
            start = i
    return out


def jj_subshells(n: int, l: int) -> str:
    """'6p' -> '6p-,6p': in DBSR notation 'nl' is j = l+1/2 only and 'nl-' is j = l-1/2."""
    nl = io.shell_name(n, l)
    return nl if l == 0 else f"{nl}-,{nl}"


def expand_varied(varied: str) -> str:
    """Expand non-relativistic shell names in a dbsr_hf ``varied`` list to both jj
    subshells ('6p' -> '6p-,6p'); 'all', 'none', 'nl-' and numbers are kept."""
    if varied.strip().lower() in ("all", "none") or not varied.strip():
        return varied
    out = []
    for item in varied.replace(" ", ",").split(","):
        item = item.strip()
        m = re.fullmatch(r"(\d+)([a-z])", item)
        if m and m.group(2) in io.L_SYMBOLS[1:]:
            full = jj_subshells(int(m.group(1)), io.L_SYMBOLS.index(m.group(2)))
            out += [x for x in full.split(",") if x not in out]
        elif item and item not in out:
            out.append(item)
    return ",".join(out)


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

    def add(self, conf: str | Sequence[str], name: str | None = None,
            varied: str | Sequence[str] | None = None,
            term: str = "LS", jj_varied: str = "none", ci: bool = False,
            correlation: Sequence[str] | str | None = None, mchf_varied: str | None = None,
            mchf_max_it: int = 100, **hf_args) -> ConfigSpec:
        """Add a configuration (only peel shells, e.g. ``'5s2 5p4 6p'``), or a list of
        configurations of the same parity that are treated together, with
        configuration interaction between them (e.g. ``['5s 5p6', '5s2 5p4 5d']``:
        the strong 5s5p^6 - 5p^4 5d mixing in Xe II).  Each resulting state is
        labelled with its dominant configuration.

        ``correlation``: extra configurations containing correlation orbitals,
        e.g. ``['5s2 5p4 6d']`` for 5s2 5p4 5d.  After dbsr_hf, dbsr_mchf
        re-optimises ``mchf_varied`` (default: the new orbitals of all
        configurations, e.g. '5d-,5d,6d-,6d') for the *physical* states only
        (the lowest ones in each J, as many as the physical configurations
        have CSFs).  The correlation orbital then describes the term
        dependence of the physical orbital (a different 5d for different
        parent cores) while all states stay orthogonal (one orbital set, one
        diagonalisation), as required by the scattering programs.  Only the
        physical states are kept.

        The first configuration is the reference: all its orbitals are
        optimised (``varied='all'``).  For the others, by default only the
        orbitals absent from the reference are optimised and the reference
        orbitals are used as input, which keeps the orbital set compact and
        consistent (as in the DBSR examples).

        Shell names in ``varied`` are non-relativistic ('6p', '5d') and are
        expanded to both jj subshells ('6p-,6p'): in DBSR notation '6p' alone
        is only the p3/2 spinor, and a non-varied 6p1/2 stays at its crude
        initial estimate (states with a 6p1/2 electron then come out several
        eV too high).  Use 'nl-' explicitly to vary a single subshell.
        """
        confs = [conf] if isinstance(conf, str) else list(conf)
        corr = [correlation] if isinstance(correlation, str) else list(correlation or [])
        n_core = sum(2 * (2 * io.L_SYMBOLS.index(s[-1]) + 1) for s in self.core_shells)
        for c in confs + corr:
            n = n_core + io.config_nelectrons(c)
            if n != self.ion.nelc:
                raise ValueError(f"{c!r}: {n} electrons with the core, {self.ion} has {self.ion.nelc}")
        if len({io.config_parity(c) for c in confs + corr}) > 1:
            raise ValueError(f"configurations {confs + corr} have different parities")
        name = name or "__".join(_default_name(c) for c in confs)
        if any(s.name == name for s in self.specs):
            raise ValueError(f"duplicate name {name!r}")
        ref = self.specs[0] if self.specs else None
        if varied is None:
            if ref is None:
                varied = "all"
            else:
                ref_sh = {(n_, l) for c in _confs(ref) for n_, l, _ in io.parse_config(c)}
                new = []
                for c in confs + corr:
                    new += [(n_, l) for n_, l, _ in io.parse_config(c) if (n_, l) not in ref_sh and (n_, l) not in new]
                varied = ",".join(jj_subshells(n_, l) for n_, l in new) if new else "none"
        else:
            if not isinstance(varied, str):
                varied = ",".join(varied)
            varied = expand_varied(varied)
        if corr and mchf_varied is None:
            mchf_varied = varied if varied not in ("all", "none") else ""
        spec = ConfigSpec(conf=" + ".join(io.pretty_conf(c) for c in confs), name=name,
                          correlation=" + ".join(io.pretty_conf(c) for c in corr),
                          mchf_varied=expand_varied(mchf_varied or ""), mchf_max_it=mchf_max_it, varied=varied, term=term,
                          jj_varied=jj_varied, inp=None if ref is None else f"{ref.name}.bsw",
                          ci=ci, extra=hf_args)
        self.specs.append(spec)
        return spec

    # ------------------------------------------------------------------ runs
    def _hf_args(self, spec: ConfigSpec) -> list[str]:
        shells = []
        for c in _all_confs(spec):
            shells += [io.shell_name(n, l) for n, l, _ in io.parse_config(c) if io.shell_name(n, l) not in shells]
        args = [spec.name, f"z={self.ion.z}", f"core={self.core}", f"peel={' '.join(shells)}",
                f"term={spec.term}", f"varied={spec.varied}", f"max_it={self.max_it}"]
        if len(_all_confs(spec)) == 1:
            args.append(f"conf={io.to_dbsr_conf(spec.conf)}")
        if spec.inp:
            args.append(f"inp={spec.inp}")
        args += self._grid_args()
        args += [f"{k}={v}" for k, v in {**self.hf_args, **spec.extra}.items()]
        return args

    def _run_mchf(self, spec: ConfigSpec, sb: Path):
        """Optimise the correlation orbitals for the physical states (see add())."""
        cf = io.CFile.read(sb / f"{spec.name}.c")
        nphys = _n_physical(spec, cf)
        levels = ";".join(f"{ib}," + ",".join(str(k) for k in range(1, nphys[ib] + 1))
                          for ib in sorted(nphys) if nphys[ib] > 0)
        shutil.copy(sb / f"{spec.name}.knot", sb / "knot.dat")
        run("dbsr_breit3", [f"{spec.name}.c"], sb, echo=self.echo, log=f"{spec.name}.breit.out")
        varied = spec.mchf_varied or spec.varied
        run("dbsr_mchf", [spec.name, f"varied={varied}", "eol=5", f"levels={levels}",
                          f"max_it={spec.mchf_max_it}"], sb, echo=self.echo, log=f"{spec.name}.mchf.out")
        # dbsr_mchf writes only the optimised levels to name.j; recompute all
        # eigenvectors with the new orbitals (dbsr_hf jj step, orbitals fixed)
        run("dbsr_hf", [spec.name, "term=jj", "varied=none", f"inp={spec.name}.bsw", "max_it=1",
                        *self._grid_args()], sb, echo=self.echo, log=f"{spec.name}.hf3.out")

    def _write_ls(self, spec: ConfigSpec, path: Path):
        """dbsr_hf input with several LS configurations (name.LS): atom, core in
        4-character columns, one configuration per line, '*'."""
        lines = [self.ion.symbol, "".join(f"{c:>4s}" for c in self.core_shells)]
        for c in _all_confs(spec):
            lines.append("".join(f"{io.shell_name(n, l):>4s}({q:2d})" for n, l, q in io.parse_config(c)))
        lines.append("*")
        path.write_text("\n".join(lines) + "\n")

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
        if len(_all_confs(spec)) > 1:
            self._write_ls(spec, sb / f"{spec.name}.LS")
        run("dbsr_hf", self._hf_args(spec), sb, echo=self.echo, log=f"{spec.name}.hf1.out")
        jj = [spec.name, "term=jj", f"varied={spec.jj_varied}", f"max_it={self.max_it}", *self._grid_args()]
        run("dbsr_hf", jj, sb, echo=self.echo, log=f"{spec.name}.hf2.out")
        if spec.correlation:
            self._run_mchf(spec, sb)
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
            keep = None
            if spec.correlation:
                nphys = _n_physical(spec, cf)
                blocks = _j_blocks(cf)
                keep = {blocks[ib - 1][0]: n for ib, n in nphys.items()}
                taken: dict = {}
            for sol in sorted(sols, key=lambda s: s.energy):
                if keep is not None:
                    taken[sol.ic1] = taken.get(sol.ic1, 0) + 1
                    if taken[sol.ic1] > keep.get(sol.ic1, 0):
                        continue
                # 2J/parity from the CSFs themselves (the J column of .j files is not reliable)
                two_j, parity = io.csf_jp(cf.csfs[sol.ic1 - 1][2])
                sol.two_j = two_j
                count[two_j] = count.get(two_j, 0) + 1
                name = f"{spec.name}_j{two_j}_{count[two_j]}"
                io.write_state(spec.name, sol, cf, self.workdir / f"{spec.name}.bsw", self.workdir / name)
                conf = _dominant_config(sol, cf, _confs(spec))
                states.append(State(name, conf, sol.label, two_j, parity, sol.energy, spec.name,
                                    configs=_confs(spec) if len(_confs(spec)) > 1 else None))
        states.sort(key=lambda s: s.energy)
        return states

    # ------------------------------------------------------ term dependence
    def refine(self, spec: str | ConfigSpec, varied: str | None = None,
               key: Callable[[State], str | None] | None = None, max_it: int = 40,
               min_states: int = 1, progress: Callable[[str], None] | None = print) -> dict:
        """Term-dependent orbitals for one configuration (``spec``, a name or
        :class:`ConfigSpec`): its states are divided into groups (by default by
        the parent core in the NIST label, e.g. '(1D2)'; run
        :func:`pydbsr.nist.assign` first) and for every group the orbitals
        ``varied`` are re-optimised with dbsr_mchf for the states of that group
        only (statistical weights).  Each state then keeps the orbitals of its
        group; DBSR treats the resulting non-orthogonal orbital sets.

        ``varied`` defaults to the orbitals optimised for the configuration plus
        the open shells of the reference configuration (e.g. ``'5d-,5d,5p-,5p'``
        for 5s2 5p4 5d in Xe II): relaxation of the 5p core matters as much as
        that of the 5d orbital.  States without a group (``key`` -> None) keep
        their orbitals.  Returns {group: [state names]}.

        Warning: states of the same J from different groups are then not
        orthogonal, which the close-coupling programs do not allow (dbsr_hd3
        stops with 'DPOTRF ... Cholesky factorization failed').  Use it for
        structure (energies, oscillator strengths) only; for scattering targets
        use ``Target.add(..., correlation=[...])`` instead.
        """
        spec = spec if isinstance(spec, ConfigSpec) else next(s for s in self.specs if s.name == spec)
        if varied is None:
            ref = self.specs[0]
            open_ref = [(n, l) for c in _confs(ref) for n, l, q in io.parse_config(c) if q != 2 * (2 * l + 1)]
            extra = ",".join(jj_subshells(n, l) for n, l in dict.fromkeys(open_ref))
            varied = ",".join(x for x in [spec.varied if spec.varied not in ("none", "all") else "", extra] if x)
        varied = expand_varied(varied)
        key = key or _core_from_label
        wd = self.workdir
        cf = io.CFile.read(wd / f"{spec.name}.c")
        blocks = _j_blocks(cf)
        block_of_j = {io.csf_jp(cf.csfs[a - 1][2])[0]: ib for ib, (a, _) in enumerate(blocks, 1)}
        sols = io.read_j(wd / f"{spec.name}.j")
        members = [s for s in self.states if s.source == spec.name or s.source.startswith(spec.name + "__")]
        groups: dict = {}
        for s in members:
            g = key(s)
            if g is not None:
                groups.setdefault(g, []).append(s)
        groups = {g: v for g, v in groups.items() if len(v) >= min_states}
        base = wd / "_refine" / spec.name
        if base.exists():
            shutil.rmtree(base)
        base.mkdir(parents=True)
        for ext in (".c", ".bsw"):
            shutil.copy(wd / f"{spec.name}{ext}", base / f"{spec.name}{ext}")
        knot = wd / f"{spec.name}.knot"
        shutil.copy(knot if knot.exists() else wd / f"{self.specs[0].name}.knot", base / "knot.dat")
        run("dbsr_breit3", [f"{spec.name}.c"], base, log="breit.out")

        def rank(s):                                   # energy rank of the state within its J block
            return int(s.name.rsplit("_", 1)[1])

        for g, sts in groups.items():
            gname = _safe(g)
            sb = base / gname
            sb.mkdir()
            for f in (f"{spec.name}.c", f"{spec.name}.bsw", f"{spec.name}.bnk", "knot.dat"):
                shutil.copy(base / f, sb / f)
            lev: dict = {}
            for s in sts:
                lev.setdefault(block_of_j[s.two_j], []).append(rank(s))
            levels = ";".join(f"{b}," + ",".join(map(str, sorted(v))) for b, v in sorted(lev.items()))
            if progress:
                progress(f"refine {spec.name} {g}: {len(sts)} states, varied={varied}")
            run("dbsr_mchf", [spec.name, f"varied={varied}", "eol=5", f"levels={levels}",
                              f"max_it={max_it}"], sb, log="mchf.out")
            new = io.read_j(sb / f"{spec.name}.j")
            bsw = wd / f"{spec.name}__{gname}.bsw"
            shutil.copy(sb / f"{spec.name}.bsw", bsw)
            for s in sts:
                a = blocks[block_of_j[s.two_j] - 1][0]
                old = sorted([x for x in sols if x.ic1 == a], key=lambda x: x.energy)[rank(s) - 1]
                cand = [x for x in new if x.ic1 == a]
                best = max(cand, key=lambda x: abs(sum(p * q for p, q in zip(x.coefs, old.coefs))))
                io.write_state(spec.name, best, cf, bsw, wd / s.name)
                s.energy = best.energy
                s.source = f"{spec.name}__{gname}"
        shutil.rmtree(base)
        self.states.sort(key=lambda s: s.energy)
        self.save()
        return {g: [s.name for s in v] for g, v in groups.items()}

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
