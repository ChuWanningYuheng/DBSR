"""NIST Atomic Spectra Database levels: download, cache and assign to target states.

The levels are used as experimental thresholds (``dbsr_hd iexp=1``) and to
label the computed states.  If the machine has no internet access, download
the table once elsewhere (``pydbsr nist "Xe II" -o xe2.txt``) and use
:func:`read_levels`.
"""
from __future__ import annotations

import csv
import io as _io
import os
import re
import urllib.parse
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from . import io
from .atoms import Ion

ASD_URL = "https://physics.nist.gov/cgi-bin/ASD/energy1.pl"


@dataclass
class Level:
    config: str            # NIST configuration string, e.g. '5s2.5p4.(3P).6s'
    term: str
    two_j: int
    energy_cm: float
    parity: int            # +1 / -1 (from the configuration)

    @property
    def J(self):
        return self.two_j / 2

    def shells(self) -> dict:
        return nist_shells(self.config)


def _cache_dir() -> Path:
    d = Path(os.environ.get("PYDBSR_CACHE", Path.home() / ".cache" / "pydbsr"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def fetch_levels(spectrum: str | Ion, cache: bool = True, timeout: float = 60) -> list[Level]:
    """Download energy levels of ``spectrum`` (e.g. ``'Xe II'`` or ``Ion('Xe', 1)``) in cm-1."""
    if isinstance(spectrum, Ion):
        spectrum = spectrum.spectrum
    fname = _cache_dir() / ("nist_" + spectrum.replace(" ", "_") + ".tsv")
    if cache and fname.exists():
        return parse_levels(fname.read_text())
    params = {
        "de": "0", "spectrum": spectrum, "submit": "Retrieve Data", "units": "0", "format": "3",
        "output": "0", "page_size": "15", "multiplet_ordered": "0", "conf_out": "on",
        "term_out": "on", "level_out": "on", "unc_out": "0", "j_out": "on", "lande_out": "0",
        "perc_out": "0", "biblio": "0", "temp": "",
    }
    url = ASD_URL + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "pydbsr"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        text = r.read().decode("utf-8", errors="replace")
    levels = parse_levels(text)
    if not levels:
        raise RuntimeError(f"NIST ASD returned no levels for {spectrum!r}:\n{text[:500]}")
    if cache:
        fname.write_text(text)
    return levels


def read_levels(path) -> list[Level]:
    """Levels from a saved NIST ASD table (tab-delimited, CSV) or a simple
    whitespace table ``config  term  J  energy_cm``."""
    return parse_levels(Path(path).read_text())


def _clean(x: str) -> str:
    x = x.strip()
    if x.startswith("="):
        x = x[1:]
    return x.strip().strip('"').strip()


def _parse_j(s: str) -> list[int]:
    s = _clean(s).replace("?", "").strip()
    if not s:
        return []
    out = []
    for part in re.split(r"[,;]| or ", s):
        part = part.strip()
        if not part:
            continue
        if "/" in part:
            a, b = part.split("/")
            out.append(int(a))
        else:
            try:
                out.append(int(round(2 * float(part))))
            except ValueError:
                pass
    return out


def _parse_energy(s: str) -> float | None:
    s = _clean(s)
    s = re.sub(r"[\[\]()?a-zA-Z+]", "", s).strip()
    try:
        return float(s)
    except ValueError:
        return None


def nist_shells(conf: str) -> dict:
    """``'5s2.5p4.(3P).6s'`` -> {(5,0):2, (5,1):4, (6,0):1} (parent terms removed)."""
    c = re.sub(r"\([^)]*\)|<[^>]*>", " ", conf)
    out = {}
    for n, l, q in re.findall(r"(\d+)([spdfghik])(\d*)", c):
        out[(int(n), io.L_SYMBOLS.index(l))] = int(q) if q else 1
    return out


def parse_levels(text: str) -> list[Level]:
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return []
    delim = "\t" if "\t" in lines[0] else ("," if lines[0].count(",") >= 3 else None)
    levels = []
    if delim is None:            # simple whitespace table
        for line in lines:
            if line.lstrip().startswith("#"):
                continue
            f = line.split()
            if len(f) < 4:
                continue
            e = _parse_energy(f[-1])
            js = _parse_j(f[-2])
            if e is None or not js:
                continue
            conf = f[0]
            for tj in js:
                levels.append(Level(conf, f[1] if len(f) > 4 else "", tj, e, _parity(conf)))
        return levels
    rows = list(csv.reader(_io.StringIO("\n".join(lines)), delimiter=delim))
    head = [_clean(h).lower() for h in rows[0]]

    def col(*names):
        for n in names:
            for i, h in enumerate(head):
                if h.startswith(n):
                    return i
        return None
    ic, it, ij, ie = col("configuration"), col("term"), col("j"), col("level")
    if ic is None or ij is None or ie is None:
        raise ValueError(f"unrecognised NIST table header: {rows[0]}")
    for row in rows[1:]:
        if len(row) <= max(ic, ij, ie):
            continue
        conf = _clean(row[ic])
        e = _parse_energy(row[ie])
        js = _parse_j(row[ij])
        if not conf or e is None or not js:
            continue
        term = _clean(row[it]) if it is not None else ""
        for tj in js:
            levels.append(Level(conf, term, tj, e, _parity(conf, term)))
    return levels


def _parity(conf: str, term: str = "") -> int:
    if term.endswith("*"):
        return -1
    sh = nist_shells(conf)
    return -1 if sum(l * q for (n, l), q in sh.items()) % 2 else 1


def _open_shells(sh: dict) -> dict:
    return {k: q for k, q in sh.items() if q != 2 * (2 * k[1] + 1)}


def assign(states, levels: Sequence[Level] | str | Ion, overwrite: bool = True,
           verbose: bool = True) -> list[tuple]:
    """Assign NIST levels to computed states.

    States and levels are grouped by (open-shell configuration, J, parity) and
    matched in energy order within each group.  Sets ``state.exp_energy_cm``
    (relative to the lowest assigned level of the table) and ``state.nist_label``.
    Returns a list of (state, level or None).
    """
    if isinstance(levels, (str, Ion)):
        levels = fetch_levels(levels)
    groups: dict = {}
    for lv in levels:
        key = (frozenset(_open_shells(lv.shells()).items()), lv.two_j, lv.parity)
        groups.setdefault(key, []).append(lv)
    for g in groups.values():
        g.sort(key=lambda lv: lv.energy_cm)
    sgroups: dict = {}
    for s in states:
        if not s.config:
            continue
        sh = {(n, l): q for n, l, q in io.parse_config(s.config)}
        key = (frozenset(_open_shells(sh).items()), s.two_j, s.parity)
        sgroups.setdefault(key, []).append(s)
    result = []
    for key, sts in sgroups.items():
        sts.sort(key=lambda s: s.energy)
        lvs = groups.get(key, [])
        for i, s in enumerate(sts):
            lv = lvs[i] if i < len(lvs) else None
            if lv is not None and (overwrite or s.exp_energy_cm is None):
                s.exp_energy_cm = lv.energy_cm
                s.nist_label = f"{lv.config} {lv.term} J={lv.two_j}/2" if lv.two_j % 2 else \
                    f"{lv.config} {lv.term} J={lv.two_j // 2}"
            result.append((s, lv))
    missing = [s.name for s, lv in result if lv is None]
    if missing and verbose:
        warnings.warn(f"no NIST level found for {len(missing)} state(s): {', '.join(missing[:10])}"
                      + (" ..." if len(missing) > 10 else ""))
    return result
