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
    no: int | None = None  # level number in order of increasing energy ("NIST No.")

    @property
    def J(self):
        return self.two_j / 2

    def shells(self) -> dict:
        return nist_shells(self.config)


def _cache_dir() -> Path:
    d = Path(os.environ.get("PYDBSR_CACHE", Path.home() / ".cache" / "pydbsr"))
    d.mkdir(parents=True, exist_ok=True)
    return d


_UA = "Mozilla/5.0 (X11; Linux x86_64) pydbsr (atomic data; +https://github.com/ChuWanningYuheng/DBSR)"


def _asd_params(spectrum: str, fmt: int) -> dict:
    return {
        "de": "0", "spectrum": spectrum, "units": "0", "format": str(fmt), "output": "0",
        "page_size": "15", "multiplet_ordered": "0", "conf_out": "on", "term_out": "on",
        "level_out": "on", "unc_out": "0", "j_out": "on", "lande_out": "0", "perc_out": "0",
        "biblio": "0", "temp": "", "submit": "Retrieve Data",
    }


def fetch_levels(spectrum: str | Ion, cache: bool = True, timeout: float = 60) -> list[Level]:
    """Download energy levels of ``spectrum`` (e.g. ``'Xe II'`` or ``Ion('Xe', 1)``) in cm-1.

    Tries the tab-delimited, CSV and ASCII outputs of the NIST ASD levels form.
    If NIST answers with an HTML page instead of a table (maintenance, access
    restrictions), a RuntimeError explains how to download the table by hand.
    """
    if isinstance(spectrum, Ion):
        spectrum = spectrum.spectrum
    fname = _cache_dir() / ("nist_" + spectrum.replace(" ", "_") + ".tsv")
    if cache and fname.exists():
        levels = parse_levels(fname.read_text())
        if levels:
            return levels
        fname.unlink()                                     # bad cache from an old version
    answers = []
    for fmt in (3, 2, 1):
        url = ASD_URL + "?" + urllib.parse.urlencode(_asd_params(spectrum, fmt))
        req = urllib.request.Request(url, headers={"User-Agent": _UA, "Accept": "text/plain,text/html,*/*"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                text = r.read().decode("utf-8", errors="replace")
        except OSError as e:
            answers.append(f"format={fmt}: {e}")
            continue
        levels = parse_levels(text)
        if levels:
            if cache:
                fname.write_text(text)
            return levels
        answers.append(f"format={fmt}: {_snippet(text)}")
    raise RuntimeError(
        f"could not get a level table for {spectrum!r} from NIST ASD.\n  " + "\n  ".join(answers) +
        "\nDownload it by hand: https://physics.nist.gov/PhysRefData/ASD/levels_form.html -> spectrum "
        f"'{spectrum}', 'Format output: Tab-delimited', Level units cm-1, Configuration/Term/J/Level on; "
        "save the page as a .txt file and use pydbsr.nist.read_levels('file.txt').")


def _snippet(text: str, n: int = 200) -> str:
    t = re.sub(r"<[^>]*>", " ", text)
    t = re.sub(r"\s+", " ", t).strip()
    return (t[:n] + "...") if len(t) > n else t


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
            a = part.split("/")[0].strip()
            if a.isdigit():
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
    # occupation digits must not swallow the n of the next shell ('5s5p6' = 5s 5p6)
    for n, l, q in re.findall(r"(\d+)([spdfghik])(\d*?)(?=\d+[spdfghik]|[^\d]|$)", c):
        out[(int(n), io.L_SYMBOLS.index(l))] = int(q) if q else 1
    return out


def number_levels(levels: list[Level]) -> list[Level]:
    """Set ``Level.no`` (1-based, increasing energy; equal energies keep table order)."""
    order = sorted(range(len(levels)), key=lambda i: (levels[i].energy_cm, i))
    seen = {}
    no = 0
    for i in order:
        key = (levels[i].config, levels[i].term, levels[i].energy_cm)
        if key not in seen:                 # a row "J = 1/2, 3/2" gives one level per J
            no += 1
        seen[key] = True
        levels[i].no = no if levels[i].no is None else levels[i].no
    return levels


def _strip_html(text: str) -> str:
    m = re.search(r"<pre[^>]*>(.*?)</pre>", text, re.S | re.I)
    if m:
        text = m.group(1)
    text = re.sub(r"<[^>]*>", "", text)
    return text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")


def parse_levels(text: str) -> list[Level]:
    """Level table from NIST ASD output (tab, CSV or ASCII '|' format, also inside
    HTML) or from a simple whitespace table ``config term J energy_cm``."""
    if re.search(r"<\s*(html|body|pre|table|!doctype)", text[:5000], re.I):
        text = _strip_html(text)
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return []
    hi = next((i for i, l in enumerate(lines) if re.search(r"configuration", l, re.I)
               and re.search(r"\bJ\b", l)), None)
    if hi is None:
        return _parse_whitespace(lines)
    head_line = lines[hi]
    delim = "\t" if "\t" in head_line else ("|" if "|" in head_line else ",")
    if delim == "|":
        rows = [[c for c in l.split("|")] for l in lines[hi:]]
    else:
        rows = list(csv.reader(_io.StringIO("\n".join(lines[hi:])), delimiter=delim))
    head = [_clean(h).lower() for h in rows[0]]

    def col(*names):
        for n in names:
            for i, h in enumerate(head):
                if h.startswith(n):
                    return i
        return None
    ic, it, ij, ie = col("configuration"), col("term"), col("j"), col("level")
    if ic is None or ij is None or ie is None:
        return []
    levels = []
    conf = term = ""
    for row in rows[1:]:
        if len(row) <= max(ic, ij, ie) or set("".join(row).strip()) <= set("-+ "):
            continue
        c = _clean(row[ic])
        t = _clean(row[it]) if it is not None else ""
        if c:
            conf, term = c, t
        elif t:
            term = t
        e = _parse_energy(row[ie])
        js = _parse_j(row[ij])
        if not conf or e is None or not js or not nist_shells(conf):
            continue
        for tj in js:
            levels.append(Level(conf, term, tj, e, _parity(conf, term)))
    return number_levels(levels)


def _parse_whitespace(lines) -> list[Level]:
    levels = []
    for line in lines:
        if line.lstrip().startswith("#"):
            continue
        f = line.split()
        if len(f) < 4:
            continue
        e = _parse_energy(f[-1])
        js = _parse_j(f[-2])
        if e is None or not js or not nist_shells(f[0]):
            continue
        for tj in js:
            levels.append(Level(f[0], f[1] if len(f) > 4 else "", tj, e, _parity(f[0])))
    return number_levels(levels)


def _parity(conf: str, term: str = "") -> int:
    if term.endswith("*") or term.endswith("°"):
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
                s.nist_no = lv.no
                s.nist_label = f"{lv.config} {lv.term} J={lv.two_j}/2" if lv.two_j % 2 else \
                    f"{lv.config} {lv.term} J={lv.two_j // 2}"
            result.append((s, lv))
    missing = [s.name for s, lv in result if lv is None]
    if missing and verbose:
        warnings.warn(f"no NIST level found for {len(missing)} state(s): {', '.join(missing[:10])}"
                      + (" ..." if len(missing) > 10 else ""))
    return result
