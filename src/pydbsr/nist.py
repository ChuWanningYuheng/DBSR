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
LINES_URL = "https://physics.nist.gov/cgi-bin/ASD/lines1.pl"


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


# ---------------------------------------------------------------------------
# CSV export of levels
# ---------------------------------------------------------------------------
def save_levels_csv(levels: Sequence[Level], path) -> Path:
    """Write levels as CSV (No, Configuration, Term, J, Level (cm-1), Level (eV)).
    The file can be read back with :func:`read_levels`."""
    path = Path(path)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["No", "Configuration", "Term", "J", "Level (cm-1)", "Level (eV)"])
        for lv in sorted(levels, key=lambda l: (l.no or 0, l.energy_cm)):
            j = f"{lv.two_j}/2" if lv.two_j % 2 else str(lv.two_j // 2)
            w.writerow([lv.no, lv.config, lv.term, j, f"{lv.energy_cm:.3f}", f"{lv.energy_cm / 8065.543937:.6f}"])
    return path


def download_levels_csv(spectrum: str | Ion, path=None, **kw) -> Path:
    """Download the NIST levels of ``spectrum`` and save them as a clean CSV."""
    if isinstance(spectrum, Ion):
        spectrum = spectrum.spectrum
    path = path or (spectrum.replace(" ", "_") + "_levels.csv")
    return save_levels_csv(fetch_levels(spectrum, **kw), path)


# ---------------------------------------------------------------------------
# spectral lines (to identify the levels behind observed lines)
# ---------------------------------------------------------------------------
@dataclass
class Line:
    wavelength_nm: float          # air wavelength for 200-2000 nm, as given by NIST
    intensity: str
    aki: float | None             # s^-1
    ei_cm: float | None
    ek_cm: float | None
    conf_i: str
    term_i: str
    j_i: str
    conf_k: str
    term_k: str
    j_k: str

    def __str__(self):
        a = f"{self.aki:.3e}" if self.aki else "-"
        return (f"{self.wavelength_nm:10.4f} nm  I={self.intensity:<6s} A={a:<10s} "
                f"{self.conf_i} {self.term_i} {self.j_i}  ->  {self.conf_k} {self.term_k} {self.j_k}"
                f"   (Ek = {self.ek_cm} cm-1)")


def _num(x):
    try:
        return float(re.sub(r"[^0-9eE+\-.]", "", _clean(x)))
    except ValueError:
        return None


def parse_lines(text: str) -> list[Line]:
    """NIST ASD lines output (CSV or tab, also inside HTML)."""
    if re.search(r"<\s*(html|body|pre|table|!doctype)", text[:5000], re.I):
        text = _strip_html(text)
    lines = [l for l in text.splitlines() if l.strip()]
    hi = next((i for i, l in enumerate(lines) if "wl" in l.lower() and "conf" in l.lower()), None)
    if hi is None:
        return []
    delim = "\t" if "\t" in lines[hi] else ","
    rows = list(csv.reader(_io.StringIO("\n".join(lines[hi:])), delimiter=delim))
    head = [_clean(h).lower() for h in rows[0]]

    def col(*names):
        for n in names:
            for i, h in enumerate(head):
                if h.startswith(n):
                    return i
        return None
    iw = col("obs_wl", "ritz_wl", "calc_wl")
    iw2 = col("ritz_wl", "calc_wl")
    idx = dict(intens=col("intens"), aki=col("aki"), ei=col("ei"), ek=col("ek"),
               ci=col("conf_i"), ti=col("term_i"), ji=col("j_i"), ck=col("conf_k"), tk=col("term_k"),
               jk=col("j_k"))
    get = lambda r, k: _clean(r[idx[k]]) if idx[k] is not None and idx[k] < len(r) else ""
    out = []
    for r in rows[1:]:
        if iw is None or len(r) <= iw:
            continue
        w = _num(r[iw])
        if w is None and iw2 is not None and iw2 < len(r):
            w = _num(r[iw2])
        if w is None:
            continue
        out.append(Line(w, get(r, "intens"), _num(get(r, "aki")) if get(r, "aki") else None,
                        _num(get(r, "ei")), _num(get(r, "ek")), get(r, "ci"), get(r, "ti"), get(r, "ji"),
                        get(r, "ck"), get(r, "tk"), get(r, "jk")))
    return out


def fetch_lines(spectrum: str | Ion, wmin_nm: float, wmax_nm: float, timeout: float = 60) -> list[Line]:
    """Lines of ``spectrum`` between ``wmin_nm`` and ``wmax_nm`` (air wavelengths above
    200 nm) with the lower (i) and upper (k) levels.  Uses the NIST ASD lines form;
    falls back to astroquery.nist if that is installed."""
    if isinstance(spectrum, Ion):
        spectrum = spectrum.spectrum
    params = {
        "spectra": spectrum, "limits_type": "0", "low_w": str(wmin_nm), "upp_w": str(wmax_nm),
        "unit": "1", "de": "0", "format": "2", "line_out": "0", "en_unit": "0", "output": "0",
        "bibrefs": "0", "page_size": "15", "show_obs_wl": "1", "show_calc_wl": "1",
        "unc_out": "0", "order_out": "0", "max_low_enrg": "", "show_av": "2", "max_upp_enrg": "",
        "tsb_value": "0", "min_str": "", "A_out": "0", "intens_out": "on", "max_str": "",
        "allowed_out": "1", "forbid_out": "1", "min_accur": "", "min_intens": "", "conf_out": "on",
        "term_out": "on", "enrg_out": "on", "J_out": "on", "submit": "Retrieve Data",
    }
    err = ""
    try:
        req = urllib.request.Request(LINES_URL + "?" + urllib.parse.urlencode(params),
                                     headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            text = r.read().decode("utf-8", errors="replace")
        out = parse_lines(text)
        if out:
            return out
        err = _snippet(text)
    except OSError as e:
        err = str(e)
    try:                                            # optional fallback
        import astropy.units as u
        from astroquery.nist import Nist
        tab = Nist.query(wmin_nm * u.nm, wmax_nm * u.nm, linename=spectrum, wavelength_type="vac+air")
        out = []
        for row in tab:
            w = _num(str(row["Observed"])) or _num(str(row["Ritz"]))
            if w is None:
                continue
            ei, ek = (str(row["Ei           Ek"]).split("-") + [""])[:2] if "Ei           Ek" in tab.colnames \
                else ("", "")
            out.append(Line(w, str(row["Rel."]), _num(str(row["Aki"])), _num(ei), _num(ek),
                            str(row["Lower level"]), "", "", str(row["Upper level"]), "", ""))
        if out:
            return out
    except Exception as e:                           # astroquery missing or failed
        err += f"; astroquery: {e}"
    raise RuntimeError(f"no NIST lines for {spectrum!r} in {wmin_nm}-{wmax_nm} nm: {err}\n"
                       "Download by hand: https://physics.nist.gov/PhysRefData/ASD/lines_form.html "
                       "(Format output: CSV) and use pydbsr.nist.read_lines('file.csv').")


def read_lines(path) -> list[Line]:
    return parse_lines(Path(path).read_text())


def upper_levels(lines_: Sequence[Line], levels: Sequence[Level], tol_cm: float = 1.0) -> list[tuple]:
    """Match the upper (k) level of each line to the level table: [(line, Level or None)]."""
    out = []
    for ln in lines_:
        best = None
        if ln.ek_cm is not None:
            cand = [lv for lv in levels if abs(lv.energy_cm - ln.ek_cm) <= tol_cm]
            if ln.j_k:
                tj = _parse_j(ln.j_k)
                cand = [lv for lv in cand if not tj or lv.two_j in tj] or cand
            best = min(cand, key=lambda lv: abs(lv.energy_cm - ln.ek_cm)) if cand else None
        out.append((ln, best))
    return out
