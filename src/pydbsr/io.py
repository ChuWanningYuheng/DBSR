"""Readers/writers for DBSR text files (.c, .j, target_jj, dbsr_par, knot.dat)."""
from __future__ import annotations

import re

import numpy as np
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

# ---------------------------------------------------------------------------
# configurations
# ---------------------------------------------------------------------------
L_SYMBOLS = "spdfghiklmnoqrtuv"

_SHELL_RE = re.compile(r"(\d+)([a-z])(?:\((\d+)\)|(\d+)|\^(\d+))?")


def parse_config(conf: str) -> list[tuple[int, int, int]]:
    """Parse ``'5s2 5p5'``, ``'5s(2)5p(5)'`` or ``'5s2.5p4.6s'`` -> [(n, l, q)]."""
    shells = []
    s = conf.replace(".", " ").strip()
    pos = 0
    for m in _SHELL_RE.finditer(s):
        if s[pos:m.start()].strip():
            raise ValueError(f"cannot parse configuration {conf!r} near {s[pos:m.start()]!r}")
        n, l = int(m.group(1)), L_SYMBOLS.index(m.group(2))
        q = next((int(g) for g in m.group(3, 4, 5) if g is not None), 1)
        if q > 2 * (2 * l + 1):
            raise ValueError(f"too many electrons in {n}{m.group(2)}: {q}")
        shells.append((n, l, q))
        pos = m.end()
    if s[pos:].strip() or not shells:
        raise ValueError(f"cannot parse configuration {conf!r}")
    return shells


def shell_name(n: int, l: int) -> str:
    return f"{n}{L_SYMBOLS[l]}"


def to_dbsr_conf(conf: str) -> str:
    """``'5s2 5p5'`` -> ``'5s(2)5p(5)'`` (dbsr_hf ``conf=`` syntax)."""
    return "".join(f"{shell_name(n, l)}({q})" for n, l, q in parse_config(conf))


def config_parity(conf: str) -> int:
    """+1 for even, -1 for odd configurations."""
    return -1 if sum(l * q for _, l, q in parse_config(conf)) % 2 else 1


def config_nelectrons(conf: str) -> int:
    return sum(q for _, _, q in parse_config(conf))


def pretty_conf(conf: str) -> str:
    return " ".join(f"{shell_name(n, l)}{q if q > 1 else ''}" for n, l, q in parse_config(conf))


# ---------------------------------------------------------------------------
# .c files (jj-coupled CSF lists) and .j files (solutions)
# ---------------------------------------------------------------------------
@dataclass
class CFile:
    core: str
    peel: str
    csfs: list[tuple[str, str, str]]      # (configuration, shell J, intermediate J) lines
    energy: float | None = None
    weights: list[float] | None = None

    @classmethod
    def read(cls, path) -> "CFile":
        lines = Path(path).read_text(encoding="latin-1").splitlines()
        energy = None
        head = lines[0]
        if len(head) > 15 and head[15:].strip():
            try:
                energy = float(head[15:31].replace("D", "E"))
            except ValueError:
                energy = None
        core, peel = lines[1], lines[3]
        csfs, weights, i = [], [], 5
        while i < len(lines):
            line = lines[i]
            if len(line) > 5 and line[5] == "(":
                conf = line.rstrip()
                w = None
                k = conf.rfind(")")
                tail = conf[k + 1:].strip()
                if tail:
                    w = float(tail.replace("D", "E"))
                    conf = conf[:k + 1]
                csfs.append((conf, lines[i + 1], lines[i + 2]))
                weights.append(w)
                i += 3
            elif line.startswith("***"):
                break
            else:
                i += 1
        return cls(core, peel, csfs, energy, weights if any(w is not None for w in weights) else None)


@dataclass
class Solution:
    index: int                 # 1-based solution number in the .j file
    label: str
    energy: float              # total energy, a.u.
    two_j: int
    ic1: int                   # first/last CSF (1-based) of the J-block in the .c file
    ic2: int
    coefs: list[float] = field(repr=False)


def read_j(path) -> list[Solution]:
    """Solutions of a DBSR .j file (dbsr_hf, dbsr_ci, dbsr_mchf)."""
    text = Path(path).read_text(encoding="latin-1")
    if "Solutions:" not in text:
        raise ValueError(f"{path}: no 'Solutions:' section")
    lines = text.split("Solutions:", 1)[1].splitlines()
    sols, i = [], 0
    while i < len(lines):
        m = re.match(r"^\s*(\d+)\s+(\S+)\s*$", lines[i])
        if not m or i + 1 >= len(lines):
            i += 1
            continue
        f = lines[i + 1].split()
        try:
            energy, two_j, ic1, ic2 = float(f[0]), int(f[1]), int(f[2]), int(f[3])
        except (ValueError, IndexError):
            i += 1
            continue
        coefs, i = [], i + 2
        while len(coefs) < ic2 - ic1 + 1:
            coefs += [float(x.replace("D", "E")) for x in lines[i].split()]
            i += 1
        sols.append(Solution(int(m.group(1)), m.group(2), energy, two_j, ic1, ic2, coefs))
    return sols


def csf_jp(intj_line: str) -> tuple[int, int]:
    """2J and parity from the last line of a CSF (e.g. ``'   5/2+  '``)."""
    tok = intj_line.split()[-1]
    parity = 1 if tok.endswith("+") else -1
    val = tok.rstrip("+-")
    two_j = int(val[:-2]) if val.endswith("/2") else 2 * int(val)
    return two_j, parity


def write_state(name: str, sol: Solution, cfile: CFile, bsw: str | Path | None,
                out: str | Path, eps: float = 1e-7) -> Path:
    """Equivalent of the ``jcfile`` utility: one solution -> ``out.c`` (+ ``out.bsw``).

    Produces byte-identical output to Zatsarinny's jcfile.
    """
    out = Path(out)
    with open(out.with_suffix(".c"), "w") as f:
        f.write(f"Core subshells:{sol.energy:16.8f}\n{cfile.core}\nPeel subshells:\n{cfile.peel}\nCSF(s):\n")
        # as jcfile: CSFs ordered by decreasing |c|, weight right-justified (F18.15)
        for k in sorted(range(len(sol.coefs)), key=lambda k: -abs(sol.coefs[k])):
            c = sol.coefs[k]
            if abs(c) < eps:
                continue
            conf, shj, intj = cfile.csfs[sol.ic1 - 1 + k]
            f.write(f"{conf:<74s}{c:18.15f}\n{shj}\n{intj}\n")
        f.write("*\n")
    if bsw is not None:
        shutil.copy(bsw, out.with_suffix(".bsw"))
    return out.with_suffix(".c")


# ---------------------------------------------------------------------------
# target_jj
# ---------------------------------------------------------------------------
@dataclass
class TargetEntry:
    name: str
    targ: str = ""
    two_j: int = 0
    parity: int = 1
    energy: float = 0.0
    nc: int = 0
    nw: int = 0


@dataclass
class TargetJJ:
    title: str
    nz: int
    nelc: int
    states: list[TargetEntry]
    partial_waves: list[tuple[int, int, str | None]]   # (2J, parity, perturber c-file or None)
    perturbers: list[tuple[int, str]] = field(default_factory=list)   # (klsp, name)
    nct: int = 0
    nwt: int = 0

    def write(self, path="target_jj"):
        bar = "-" * 80
        out = [self.title, bar,
               f"nz    = {self.nz:4d}      !   nuclear charge",
               f"nelc  = {self.nelc:4d}      !   number of electrons", bar,
               f"ntarg = {len(self.states):4d}      !   number of target states", bar]
        for s in self.states:
            if s.targ:
                out.append(f"{s.name:<30s}  {s.targ:<10s}  {s.two_j:4d}{s.parity:4d}"
                           f"{s.energy:18.8f}{s.nc:5d}{s.nw:5d}")
            else:
                out.append(s.name)
        if self.nct or self.nwt:
            out += [bar, f"nct = {self.nct:7d}", f"nwt = {self.nwt:7d}"]
        out += [bar, f"nlsp  = {len(self.partial_waves):4d}      !   number of partial waves", bar]
        for i, (tj, p, pert) in enumerate(self.partial_waves, 1):
            line = f"{i:3d}. {tj:3d} {p:3d}"          # <= 13 chars, see dbsr_prep
            out.append(line + (f"  {pert}" if pert else ""))
        out.append(bar)
        if self.perturbers:
            out += [f"kpert = {len(self.perturbers):4d}      !   number of additional perturbers", bar]
            out += [f"{k:3d}  {name}" for k, name in self.perturbers]
            out.append(bar)
        Path(path).write_text("\n".join(out) + "\n")

    @classmethod
    def read(cls, path="target_jj") -> "TargetJJ":
        lines = Path(path).read_text(encoding="latin-1").splitlines()
        title = lines[0]
        par = {}
        for line in lines:
            m = re.match(r"^\s*(\w+)\s*=\s*(-?\d+)", line)
            if m and m.group(1) not in par:
                par[m.group(1)] = int(m.group(2))
        i = next(k for k, l in enumerate(lines) if l.strip().startswith("ntarg")) + 2
        states = []
        for line in lines[i:i + par["ntarg"]]:
            f = line.split()
            if len(f) >= 7:
                states.append(TargetEntry(f[0], f[1], int(f[2]), int(f[3]), float(f[4].replace("D", "E")),
                                          int(f[5]), int(f[6])))
            else:
                states.append(TargetEntry(f[0]))
        pws = []
        if "nlsp" in par:
            i = next(k for k, l in enumerate(lines) if l.strip().startswith("nlsp")) + 2
            for line in lines[i:i + par["nlsp"]]:
                f = line.split()
                pws.append((int(f[1]), int(f[2]), f[3] if len(f) > 3 and len(line.rstrip()) > 13 else None))
        perts = []
        if par.get("kpert", 0) > 0:
            i = next(k for k, l in enumerate(lines) if l.strip().startswith("kpert")) + 2
            for line in lines[i:i + par["kpert"]]:
                f = line.split()
                perts.append((int(f[0]), f[1]))
        return cls(title, par["nz"], par["nelc"], states, pws, perts, par.get("nct", 0), par.get("nwt", 0))


# ---------------------------------------------------------------------------
# dbsr_par (key = value parameters shared by all programs)
# ---------------------------------------------------------------------------
def _fmt(v):
    if isinstance(v, bool):
        return str(int(v))
    if isinstance(v, float):
        return f"{v:.10g}".replace("e", "d") if ("e" in f"{v:.10g}") else f"{v:.10g}"
    return str(v)


def write_par(path, params: dict, extra_lines: Iterable[str] = ()):
    """Write ``dbsr_par``.  Lines are ``name = value`` (read with Read_ipar/Read_rpar)."""
    lines = [f"{k} = {_fmt(v)}" for k, v in params.items() if v is not None]
    lines += list(extra_lines)
    Path(path).write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# knot.dat
# ---------------------------------------------------------------------------
@dataclass
class Grid:
    """B-spline grid parameters (``knot.dat``, semi-exponential grid_type = 1).

    ``rmax`` is the R-matrix radius for scattering runs.  ``hmax`` is the
    maximal step, i.e. controls the density of B-splines at large r (it
    must resolve the fastest continuum electron: roughly hmax < 1/k_max).
    """
    rmax: float = 50.0
    hmax: float = 1.0
    ks: int = 9
    hi: float = 0.25
    he: float = 0.25
    ml: int = 1
    ksp: int = 8
    ksq: int = 9
    nuclear: str = "Fermi"

    def write(self, path, z: float, atw: float | None = None):
        atw = atw if atw is not None else 0.0
        text = f"""grid_type =   1

ks = {self.ks:4d}               !  order  of splines
ml = {self.ml:4d}               !  number of initial equal-spaced intervals

ksp= {self.ksp:4d}
ksq= {self.ksq:4d}

hi = {self.hi:18.8f}  !  initial step for r*zg
he = {self.he:18.8f}  !  exponetial step size
hmax = {self.hmax:16.8f}  !  maximum step size for r, given
rmax = {self.rmax:16.8f}  !  maximum radius, given

atomic_number = {z:8.4f}
"""
        if atw:
            text += f"atomic_weight = {atw:8.4f}\n"
        text += f"\nnuclear =  {self.nuclear}\n"
        Path(path).write_text(text)


def modify_knot(src, dst, **params):
    """Copy ``src`` knot file to ``dst`` changing e.g. ``rmax``/``hmax``/``ks``.

    The tabulated grid points are dropped: with ``grid_type = 1`` the
    programs rebuild the grid from the parameters.
    """
    lines = Path(src).read_text(encoding="latin-1").splitlines()
    out, seen = [], set()
    for line in lines:
        if line.startswith("grid points"):
            break
        m = re.match(r"^(\s*)(\w+)(\s*=\s*)([-+\d.EeDd]+)(.*)$", line)
        if m and m.group(2) in params and params[m.group(2)] is not None:
            v = params[m.group(2)]
            val = f"{v:16.8f}" if isinstance(v, float) else f"{v:4d}"
            line = f"{m.group(1)}{m.group(2)}{m.group(3)}{val}{m.group(5)}"
            seen.add(m.group(2))
        if re.match(r"^\s*(ns|nv|tmax)\s*=", line):
            continue                      # derived quantities, recomputed
        out.append(line)
    missing = [k for k, v in params.items() if v is not None and k not in seen]
    if missing:
        out.insert(1, "\n".join(f"{k} = {params[k]}" for k in missing))
    Path(dst).write_text("\n".join(out) + "\n")


def read_knot(path) -> dict:
    out = {}
    for line in Path(path).read_text(encoding="latin-1").splitlines():
        m = re.match(r"^\s*(\w+)\s*=\s*([-+\d.EeDd]+)", line)
        if m:
            try:
                out[m.group(1)] = float(m.group(2).replace("D", "E"))
            except ValueError:
                pass
        if line.startswith("grid points"):
            break
    return out


def partial_waves(nelc_target: int, jmax: float, jmin: float = None,
                  parities: Sequence[int] = (1, -1)) -> list[tuple[int, int]]:
    """All (2J, parity) of the (N+1)-electron system with Jmin <= J <= Jmax."""
    odd = (nelc_target + 1) % 2 == 1
    tj_min = (1 if odd else 0) if jmin is None else int(round(2 * jmin))
    tj_max = int(round(2 * jmax))
    if (tj_min % 2 == 1) != odd or (tj_max % 2 == 1) != odd:
        raise ValueError(f"J must be {'half-integer' if odd else 'integer'} for {nelc_target + 1} electrons")
    return [(tj, p) for tj in range(tj_min, tj_max + 1, 2) for p in parities]


# ---------------------------------------------------------------------------
# radial functions (large P and small Q components) of a bsw-file
# ---------------------------------------------------------------------------
@dataclass
class RadialFunction:
    """Dirac radial orbital: large component P(r), small component Q(r)."""
    n: int
    kappa: int
    energy: float                # orbital energy (a.u.) stored in the bsw-file
    r: "np.ndarray"
    P: "np.ndarray"
    Q: "np.ndarray"

    @property
    def l(self) -> int:
        return self.kappa if self.kappa > 0 else -self.kappa - 1

    @property
    def name(self) -> str:
        return f"{self.n}{L_SYMBOLS[self.l]}{'-' if self.kappa > 0 else ''}"

    def norm(self) -> float:
        """Integral of P^2 + Q^2 over r (cubic-spline quadrature)."""
        from scipy.interpolate import CubicSpline
        f = CubicSpline(self.r, self.P ** 2 + self.Q ** 2)
        return float(f.integrate(self.r[0], self.r[-1]))


def radial_functions(bsw, knot=None) -> list[RadialFunction]:
    """P(r), Q(r) of all orbitals of a DBSR ``.bsw`` file on a GRASP grid.

    Uses the DBSR utility ``bsw_rw`` (bsw -> GRASP w-file) in a scratch
    directory; ``knot`` is the knot file of the calculation (default: ``knot.dat``
    or ``<name>.knot`` next to the bsw-file)."""
    import struct
    import tempfile

    import numpy as np

    from .runner import run

    bsw = Path(bsw).resolve()
    if knot is None:
        for c in (bsw.with_suffix(".knot"), bsw.parent / "knot.dat"):
            if c.exists():
                knot = c
                break
        else:
            raise FileNotFoundError(f"no knot file for {bsw}")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shutil.copy(knot, tmp / "knot.dat")
        shutil.copy(bsw, tmp / "orb.bsw")
        run("bsw_rw", ["orb.bsw"], tmp)
        data = (tmp / "orb.w").read_bytes()
    out, pos = [], 0

    def rec():
        nonlocal pos
        n = struct.unpack_from("<i", data, pos)[0]
        body = data[pos + 4:pos + 4 + n]
        pos += n + 8
        return body

    if rec()[:6] != b"G92RWF":
        raise ValueError("bsw_rw output is not a GRASP w-file")
    while pos < len(data):
        n, kappa, e, nr = struct.unpack("<iidi", rec()[:20])
        pq = np.frombuffer(rec(), dtype="<f8")
        r = np.frombuffer(rec(), dtype="<f8")
        out.append(RadialFunction(n, kappa, e, r.copy(), pq[1:nr + 1].copy(), pq[nr + 1:2 * nr + 1].copy()))
    return out
