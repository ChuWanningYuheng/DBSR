"""Radiative transitions between target states: dbsr_mult3 + dbsr_dmat3.

For every pair of states ``(a, b)`` allowed by the multipole selection rules
pydbsr runs, in a private subdirectory,

    dbsr_mult3 a.c b.c E1            # angular coefficients -> mult_bnk_E1
    dbsr_dmat3 a.c b.c c c atype=E1  # radial integrals      -> zf_res

(the state files ``name.c``/``name.bsw`` written by :class:`Target`; DBSR
handles the non-orthogonal orbitals of states from different calculations)
and reads line strengths S, gf and A in the length and velocity forms from
``zf_res``.  The calculated transition energies can be replaced by the
experimental ones (:meth:`Transition.a_exp`): A scales as dE^(2L+1) at fixed
line strength.

    tr = transitions(tg, kinds=("E1",), jobs=8)
    tab = TransitionTable(tr, tg.states)           # NIST energies if assigned
    tab.branching(upper)                            # {lower: A/sum A}
"""
from __future__ import annotations

import csv
import json
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

from .constants import AU_CM
from .runner import DBSRError, run

__all__ = ["Transition", "TransitionTable", "transitions", "parse_zf_res", "allowed"]


@dataclass
class Transition:
    """One line between two target states (``lower`` has the lower calculated energy)."""
    kind: str               # 'E1', 'M1', 'E2'
    lower: str
    upper: str
    two_j_lower: int
    two_j_upper: int
    de_cm: float            # calculated transition energy, cm-1
    S: float                # line strength, length form (a.u.)
    gf: float               # gf, length form
    A: float                # s-1, length form, calculated energy
    S_v: float | None = None    # velocity form (E-type only)
    gf_v: float | None = None
    A_v: float | None = None

    @property
    def order(self) -> int:
        return int(self.kind[1:])

    def a_exp(self, de_exp_cm: float | None, form: str = "length") -> float:
        """A with the experimental transition energy (fixed line strength):
        A_exp = A_calc (dE_exp/dE_calc)^(2L+1); the velocity form, whose
        line strength itself depends on dE, scales as dE_exp^(2L+1)/dE^(2L+1) too
        (only the length form is invariant, use it by default)."""
        a = self.A if form == "length" else self.A_v
        if a is None:
            return float("nan")
        if not de_exp_cm or self.de_cm <= 0:
            return a
        return a * (abs(de_exp_cm) / self.de_cm) ** (2 * self.order + 1)

    @property
    def gauge_ratio(self) -> float | None:
        """A_velocity / A_length: far from 1 means an unreliable line."""
        return self.A_v / self.A if self.A_v and self.A else None


def allowed(kind: str, two_j1: int, p1: int, two_j2: int, p2: int) -> bool:
    """Multipole selection rules (J, parity) for E1, M1, E2, ..."""
    L = int(kind[1:])
    if not abs(two_j1 - two_j2) <= 2 * L <= two_j1 + two_j2:
        return False
    same = p1 == p2
    parity_ok = (L % 2 == 0) == same if kind[0] == "E" else (L % 2 == 1) == same
    return parity_ok


_FLOAT = r"[-+]?\d*\.?\d+(?:[EeDd][-+]?\d+)?"


def _f(s: str) -> float:
    return float(s.replace("D", "E").replace("d", "e"))


def parse_zf_res(text: str) -> list[Transition]:
    """Read the ``zf_res`` file of dbsr_dmat3 (gf='f' or 'g' output)."""
    lines = [l for l in text.splitlines()]
    out = []
    i = 0
    head = re.compile(r"^\s*([+-])\s*(\d+)\s+(" + _FLOAT + r")\s+(\S.*)$")
    while i < len(lines):
        m1 = head.match(lines[i])
        m2 = head.match(lines[i + 1]) if i + 1 < len(lines) else None
        if not (m1 and m2) or i + 3 >= len(lines) or "CM-1" not in lines[i + 2]:
            i += 1
            continue
        de = _f(lines[i + 2].split()[0])
        m = re.search(r"([EM]\d)\s+S\s*=\s*(" + _FLOAT + r")\s+(FIK|GF)\s*=\s*(" + _FLOAT + r")\s+AKI\s*=\s*("
                      + _FLOAT + ")", lines[i + 3])
        if not m:
            i += 1
            continue
        kind, S, fk, f, A = m.group(1), _f(m.group(2)), m.group(3), _f(m.group(4)), _f(m.group(5))
        # first line = lower state (dbsr_dmat writes the lower energy first)
        tj_lo, tj_up = int(m1.group(2)), int(m2.group(2))
        name_lo, name_up = m1.group(4).split()[0], m2.group(4).split()[0]
        gf = f * (tj_lo + 1) if fk == "FIK" else f
        tr = Transition(kind, name_lo, name_up, tj_lo, tj_up, de, S, gf, A)
        if i + 4 < len(lines):
            v = re.findall(_FLOAT, lines[i + 4])
            if len(v) >= 3 and not head.match(lines[i + 4]) and lines[i + 4].startswith(" " * 9):
                fv = _f(v[1])
                tr.S_v, tr.gf_v, tr.A_v = _f(v[0]), fv * (tj_lo + 1) if fk == "FIK" else fv, _f(v[2])
                i += 1
        out.append(tr)
        i += 4
    return out


def _pair_dir(base: Path, a: str, b: str, kind: str) -> Path:
    return base / f"{kind}__{a}__{b}"


def _run_pair(src: Path, base: Path, knot: Path, a: str, b: str, kind: str, keep: bool) -> list[Transition]:
    d = _pair_dir(base, a, b, kind)
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    for nm in (a, b):
        for ext in (".c", ".bsw"):
            shutil.copy(src / f"{nm}{ext}", d / f"{nm}{ext}")
    shutil.copy(knot, d / "knot.dat")
    run("dbsr_mult3", [f"{a}.c", f"{b}.c", kind], d, log="mult.out")
    run("dbsr_dmat3", [f"{a}.c", f"{b}.c", "c", "c", f"atype={kind}"], d, log="dmat.out")
    zf = d / "zf_res"
    res = parse_zf_res(zf.read_text(encoding="latin-1")) if zf.exists() else []
    for t in res:
        t.kind = kind
    if not keep:
        shutil.rmtree(d)
    return res


def transitions(target, states: Sequence | None = None, kinds: Iterable[str] = ("E1",), *,
                pairs: Sequence[tuple[str, str]] | None = None, workdir: str | Path | None = None,
                jobs: int = 1, force: bool = False, keep: bool = False,
                progress: Callable[[str], None] | None = print) -> list[Transition]:
    """Line strengths and A-values for all allowed pairs of ``states``.

    ``target``: a :class:`pydbsr.Target` (its work directory holds the state
    files) or a directory.  ``pairs``: explicit list of (name, name) instead of
    all pairs.  Results are cached in ``workdir/transitions_<kind>.json``;
    finished pairs are not recomputed (``force=True`` recomputes).
    """
    from .structure import Target
    if isinstance(target, Target):
        src = target.workdir
        states = list(target.states if states is None else states)
        knot = src / f"{target.specs[0].name}.knot" if target.specs else src / "knot.dat"
    else:
        src = Path(target)
        knot = src / "knot.dat"
        if states is None:
            raise ValueError("states= is required when target is a directory")
    if not knot.exists():
        cand = sorted(src.glob("*.knot"))
        if not cand:
            raise FileNotFoundError(f"no knot.dat / *.knot in {src}")
        knot = cand[0]
    base = Path(workdir) if workdir is not None else src / "_transitions"
    base.mkdir(parents=True, exist_ok=True)
    by_name = {s.name: s for s in states}
    out: list[Transition] = []
    for kind in kinds:
        cache = base / f"transitions_{kind}.json"
        done: dict = {}
        if cache.exists() and not force:
            done = {tuple(k.split("|")): [Transition(**t) for t in v]
                    for k, v in json.loads(cache.read_text()).items()}
        if pairs is None:
            todo = [(a.name, b.name) for i, a in enumerate(states) for b in states[i + 1:]
                    if allowed(kind, a.two_j, a.parity, b.two_j, b.parity)]
        else:
            todo = [tuple(p) for p in pairs
                    if allowed(kind, by_name[p[0]].two_j, by_name[p[0]].parity,
                               by_name[p[1]].two_j, by_name[p[1]].parity)]
        new = [p for p in todo if p not in done]
        if progress:
            progress(f"{kind}: {len(todo)} pairs, {len(todo) - len(new)} cached, running {len(new)}")

        def job(p):
            try:
                return p, _run_pair(src, base, knot, p[0], p[1], kind, keep), None
            except DBSRError as e:
                return p, [], str(e)

        errors = []
        with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
            for n, (p, res, err) in enumerate(pool.map(job, new), 1):
                if err:
                    errors.append((p, err))
                else:
                    done[p] = res
                if n % 50 == 0 or n == len(new):
                    cache.write_text(json.dumps({"|".join(k): [asdict(t) for t in v] for k, v in done.items()}))
                    if progress:
                        progress(f"  {kind}: {n}/{len(new)} pairs")
        if errors and progress:
            progress(f"  {kind}: {len(errors)} pairs failed, e.g. {errors[0][0]}: {errors[0][1][:300]}")
        for p in todo:
            out += done.get(p, [])
    return out


class TransitionTable:
    """Transitions with experimental energies, lifetimes and branching ratios.

    ``energies_cm``: {state name: excitation energy in cm-1}; by default the
    NIST energies of the states (``State.exp_energy_cm``, set by
    :func:`pydbsr.nist.assign`).  Lines are oriented by these energies (the
    calculated order of close levels can be wrong).
    """

    def __init__(self, trs: Sequence[Transition], states: Sequence | None = None,
                 energies_cm: dict | None = None, form: str = "length"):
        self.states = {s.name: s for s in states} if states is not None else {}
        if energies_cm is None:
            energies_cm = {n: s.exp_energy_cm for n, s in self.states.items() if s.exp_energy_cm is not None}
        self.energies_cm = energies_cm
        self.form = form
        self.lines = []                      # (upper, lower, Transition, A_exp, de_exp)
        for t in trs:
            ea, eb = energies_cm.get(t.lower), energies_cm.get(t.upper)
            up, lo = (t.upper, t.lower)
            de = None
            if ea is not None and eb is not None:
                de = eb - ea
                if de < 0:
                    up, lo, de = t.lower, t.upper, -de
            self.lines.append((up, lo, t, t.a_exp(de, form), de))

    def A(self, upper: str, lower: str, kind: str | None = None) -> float:
        return sum(a for u, l, t, a, _ in self.lines if u == upper and l == lower and (kind is None or t.kind == kind))

    def total(self, upper: str) -> float:
        return sum(a for u, _, _, a, _ in self.lines if u == upper)

    def lifetime_ns(self, upper: str) -> float:
        s = self.total(upper)
        return 1e9 / s if s > 0 else float("inf")

    def branching(self, upper: str) -> dict:
        s = self.total(upper)
        out: dict = {}
        for u, l, _, a, _ in self.lines:
            if u == upper and s > 0:
                out[l] = out.get(l, 0.0) + a / s
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def wavelength_nm(self, upper: str, lower: str, air: bool = True) -> float | None:
        de = None
        for u, l, t, _, d in self.lines:
            if u == upper and l == lower:
                de = d if d is not None else t.de_cm
        if not de:
            return None
        lam = 1e7 / de
        return air_wavelength(lam) if air and lam > 200 else lam

    def rows(self) -> list[dict]:
        out = []
        for u, l, t, a, de in self.lines:
            su, sl = self.states.get(u), self.states.get(l)
            lam = 1e7 / de if de else (1e7 / t.de_cm if t.de_cm else None)
            out.append(dict(kind=t.kind, upper=u, lower=l,
                            upper_nist=getattr(su, "nist_label", "") or "", lower_nist=getattr(sl, "nist_label", "") or "",
                            upper_no=getattr(su, "nist_no", None), lower_no=getattr(sl, "nist_no", None),
                            wl_air_nm=round(air_wavelength(lam), 4) if lam and lam > 200 else lam,
                            de_calc_cm=round(t.de_cm, 2), de_exp_cm=round(de, 2) if de else None,
                            S=t.S, gf=t.gf, A_calc=t.A, A_exp=a, A_vel=t.A_v,
                            gauge_ratio=round(t.gauge_ratio, 3) if t.gauge_ratio else None,
                            branching=a / self.total(u) if self.total(u) > 0 else None))
        return out

    def to_csv(self, path) -> Path:
        rows = self.rows()
        path = Path(path)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["kind"])
            w.writeheader()
            w.writerows(rows)
        return path


def air_wavelength(vac_nm: float) -> float:
    """Vacuum -> air wavelength (Peck & Reeder / NIST formula), nm, for > 200 nm."""
    s2 = (1e3 / vac_nm) ** 2                       # (1/um)^2
    n = 1 + 0.0000834254 + 0.02406147 / (130 - s2) + 0.00015998 / (38.9 - s2)
    return vac_nm / n


def cm_to_au(e_cm: float) -> float:
    return e_cm / AU_CM
