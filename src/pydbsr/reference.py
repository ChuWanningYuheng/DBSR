"""Reference cross sections for comparisons.

``read_xlsx`` reads the supplementary spreadsheet of Wang et al (2019)
(Plasma Sources Sci. Technol., DBSR e + Xe+, 67-state model): a sheet
"NIST Level Table" (No., configuration, term, J, energy in eV) and sheets
with columns ``i->j`` (NIST level numbers) of (incident energy in eV,
cross section in 1e-16 cm^2).  Requires ``openpyxl``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .constants import EV_CM
from .nist import Level, _parity


@dataclass
class ReferenceData:
    levels: list[Level]                        # in NIST order: levels[k-1] is NIST No. k
    sigma: dict = field(default_factory=dict)  # (i, j) -> (E_inc [eV], sigma [cm^2])
    sheet: dict = field(default_factory=dict)  # (i, j) -> sheet name

    def transitions(self, sheet: str | None = None):
        return sorted(k for k, s in self.sheet.items() if sheet is None or s == sheet)

    def level(self, no: int) -> Level:
        return self.levels[no - 1]

    def label(self, no: int) -> str:
        lv = self.level(no)
        return f"{lv.config} {lv.term} {lv.two_j}/2" if lv.two_j % 2 else f"{lv.config} {lv.term} {lv.two_j // 2}"


def _two_j(s: str) -> int:
    s = str(s).strip()
    return int(s.split("/")[0]) if "/" in s else int(round(2 * float(s)))


def read_xlsx(path) -> ReferenceData:
    import openpyxl
    wb = openpyxl.load_workbook(Path(path), read_only=True, data_only=True)
    levels = []
    ref = ReferenceData(levels)
    for ws in wb.worksheets:
        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            continue
        if "level" in ws.title.lower():
            for r in rows[1:]:
                if r[0] is None:
                    continue
                conf, term, j, e = (str(r[1]).strip(), str(r[2]).strip(), r[3], float(r[4]))
                levels.append(Level(conf, term, _two_j(j), e * EV_CM, _parity(conf, term), int(r[0])))
            continue
        head = rows[0]
        for c, h in enumerate(head):
            m = re.match(r"\s*(\d+)\s*->\s*(\d+)", str(h)) if h is not None else None
            if not m:
                continue
            e, s = [], []
            for r in rows[2:]:
                if c + 1 < len(r) and r[c] is not None and r[c + 1] is not None:
                    e.append(float(r[c]))
                    s.append(float(r[c + 1]))
            key = (int(m.group(1)), int(m.group(2)))
            ref.sigma[key] = (np.array(e), np.array(s) * 1e-16)
            ref.sheet[key] = ws.title
    return ref
