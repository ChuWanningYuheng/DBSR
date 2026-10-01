"""Helpers for the atomic-data notebooks: selected lines, background runs, reference data.

The selected lines and the reasons for the choice: docs/crm/lines_final.md.
Level numbers of Xe II are those of Wang et al (2019), CrossSectionsIon.xlsx
(sheet "NIST Level Table").
"""
from __future__ import annotations

import csv
import os
import re
import signal
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CM_PER_EV = 8065.544

# Xe II lines for ne/Te (docs/crm/lines_final.md): wavelength (NIST, air), upper -> lower (Wang numbers)
XE2_LINES = [
    dict(wl=545.045, upper=44, lower=16, role="ne"),
    dict(wl=545.090, upper=51, lower=24, role="Te"),
    dict(wl=561.667, upper=41, lower=15, role="ne"),
    dict(wl=543.896, upper=39, lower=13, role="Te"),
    dict(wl=553.107, upper=29, lower=7, role="Te"),
]
XE2_UPPERS = sorted({x["upper"] for x in XE2_LINES})
# all 5p4 6p levels of Wang's table: their branches are computed too (seconds), to compare with
# every A known in NIST (validation of the target model)
XE2_VALIDATION_UPPERS = [25, 26, 29, 30, 31, 35, 37, 39, 40, 41, 42, 44, 46, 49, 51, 52, 53, 54, 56]
# lines from the same upper level: their intensity ratio depends only on the A ratio (and on the
# spectral sensitivity): a test of the computed A against the measured spectrum
XE2_SAME_UPPER = [(545.045, 575.103), (545.045, 610.143), (545.090, 589.329), (545.090, 597.113),
                  (561.667, 680.574), (553.107, 541.915)]

# Ba lines (docs/crm/lines_final.md); A: NIST ASD (all branches of the upper level known)
BA_LINES = [
    dict(ion="Ba I", wl=553.548, upper="6s.6p 1P* J=1", lower="6s2 1S J=0",
         A=1.19e8, A_branches={553.548: 1.19e8, 1499.985: 2.5e5, 1130.303: 1.1e5, 1107.570: 3.1e3}),
    dict(ion="Ba II", wl=455.403, upper="6p 2P* J=3/2", lower="6s 2S J=1/2",
         A=1.11e8, A_branches={455.403: 1.11e8, 614.171: 4.12e7, 585.368: 6.00e6}),
    dict(ion="Ba II", wl=493.408, upper="6p 2P* J=1/2", lower="6s 2S J=1/2",
         A=9.53e7, A_branches={493.408: 9.53e7, 649.690: 3.10e7}),
]
for _x in BA_LINES:
    _x["branching"] = _x["A"] / sum(_x["A_branches"].values())


# ---------------------------------------------------------------- background runs
class Job:
    """``run_model.py`` as a separate process: survives closing the browser or the kernel."""

    def __init__(self, workdir, model, **args):
        self.workdir = Path(workdir)
        self.model = model
        self.args = args
        self.pidfile = Path(f"{self.workdir}.pid")
        self.log = Path(f"{self.workdir}.log")

    def running(self):
        try:
            pid = int(self.pidfile.read_text())
            state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        except Exception:
            return None
        if state == "Z":
            try:
                os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                pass
            return None
        return pid

    def start(self, **override):
        if self.running():
            print("уже идёт, pid", self.running())
            return
        args = {**self.args, **override}
        cmd = [sys.executable, "-u", str(HERE / "run_model.py"), "--model", self.model, "--workdir", str(self.workdir)]
        for k, v in args.items():
            if v is None or v is False:
                continue
            cmd += [f"--{k.replace('_', '-')}"] + ([] if v is True else [str(v)])
        self.workdir.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        proc = subprocess.Popen(cmd, stdout=open(self.log, "a"), stderr=subprocess.STDOUT,
                                start_new_session=True, env=env)
        self.pidfile.write_text(str(proc.pid))
        print("запущено, pid", proc.pid, "\nлог:", self.log, "\n", " ".join(cmd))

    def status(self, n=15):
        print("идёт, pid " + str(self.running()) if self.running() else "НЕ идёт (закончен или упал)")
        if self.log.exists():
            print("\n".join(self.log.read_text(errors="replace").splitlines()[-n:]))

    def stop(self):
        pid = self.running()
        if pid:
            os.killpg(pid, signal.SIGTERM)
            print("остановлен", pid)


# ---------------------------------------------------------------- tables
def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fnum(x):
    try:
        return float(re.sub(r"[^0-9.eE+-]", "", str(x)))
    except ValueError:
        return None


def nist_lines(path):
    """NIST ASD lines CSV (docs/crm/nist_*.csv) -> list of dicts with floats."""
    out = []
    for d in read_csv(path):
        d = {k: (v or "").replace('"', "").lstrip("=").strip() for k, v in d.items() if k}
        wl = fnum(d.get("obs_wl_air(nm)")) or fnum(d.get("ritz_wl_air(nm)"))
        out.append(dict(wl=wl, A=fnum(d.get("Aki(s^-1)")), Ei=fnum(d.get("Ei(cm-1)")), Ek=fnum(d.get("Ek(cm-1)")),
                        intens=d.get("intens", ""), lower=f"{d.get('conf_i', '')} {d.get('term_i', '')} {d.get('J_i', '')}",
                        upper=f"{d.get('conf_k', '')} {d.get('term_k', '')} {d.get('J_k', '')}"))
    return [x for x in out if x["wl"]]


def level_number(energies_cm, e, tol=0.3):
    """Number (1-based) of the level with energy e in a sorted list, or None."""
    if e is None:
        return None
    k = int(np.argmin(abs(np.asarray(energies_cm) - e)))
    return k + 1 if abs(energies_cm[k] - e) < tol else None


# ---------------------------------------------------------------- branching ratios
def branching_from_csv(path, kind=None):
    """{upper_no: {lower_no: BR}} and {upper_no: sum A} from transitions_*.csv (run_model.py)."""
    A: dict = {}
    for r in read_csv(path):
        if kind and r["kind"] != kind:
            continue
        u, l, a = r["upper_no"], r["lower_no"], fnum(r["A_exp"])
        if not u or not l or a is None:
            continue
        A.setdefault(int(u), {}).setdefault(int(l), 0.0)
        A[int(u)][int(l)] += a
    tot = {u: sum(v.values()) for u, v in A.items()}
    return {u: {l: a / tot[u] for l, a in v.items()} for u, v in A.items()}, tot, A


def line_sigma(ref, upper, br, lower_initial=1):
    """Direct emission cross section of a line: BR * sigma(initial -> upper) (Wang table)."""
    E, s = ref.sigma[(lower_initial, upper)]
    return E, s * br


# ---------------------------------------------------------------- Fursa et al 1999 (PRA 60, 4590), 1e-16 cm2
FURSA_E = np.array([5.0, 6.0, 7.0, 8.35, 9.0, 10.0, 11.44, 15.0, 20.0, 30.0, 36.67, 41.44, 50.0, 60.0, 80.0,
                    100.0, 200.0, 400.0, 600.0, 897.6])
FURSA_1P1 = dict(  # Table II: E, Q_app (Chen & Gallagher x1.03), cascade, direct Q
    E=np.array([2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.35, 9.0, 10.0, 11.44, 15.0, 20.0, 30.0, 36.67, 41.44, 50.0, 60.0,
                80.0, 100.0, 200.0, 400.0, 600.0, 897.6]),
    Q_app=np.array([4.56, 12.00, 25.84, 33.34, 37.26, 39.89, 39.00, 40.44, 41.24, 42.56, 42.47, 39.78, 35.01, 32.39,
                    30.78, 28.11, 25.49, 21.55, 18.75, 11.65, 6.81, 4.92, 3.52]),
    Q_casc=np.array([0.00, 0.00, 4.02, 5.90, 7.27, 8.22, 7.72, 7.09, 6.45, 5.74, 6.47, 5.23, 3.92, 3.36, 3.06, 2.69,
                     2.33, 1.79, 1.40, 0.76, 0.37, 0.25, 0.17]),
    Q=np.array([4.56, 12.00, 21.83, 27.43, 29.98, 31.67, 31.28, 33.35, 34.79, 36.82, 36.00, 34.55, 31.09, 29.03,
                27.72, 25.42, 23.16, 19.76, 17.35, 10.88, 6.44, 4.66, 3.35]),
)
FURSA_TABLES = {   # Tables III, IV, VI (E = FURSA_E): direct excitation from 6s2 1S0
    "6s7p 1P1": [0.48, 0.76, 0.73, 0.81, 0.49, 0.35, 0.33, 0.32, 0.39, 0.47, 0.50, 0.50, 0.50, 0.49, 0.46, 0.42,
                 0.30, 0.19, 0.14, 0.10],
    "6s8p 1P1": [0.22, 0.45, 0.71, 0.78, 0.64, 0.69, 1.00, 1.18, 1.30, 1.45, 1.46, 1.48, 1.49, 1.45, 1.25, 1.10,
                 0.74, 0.44, 0.31, 0.23],
    "5d2 1S0": [0.81, 0.89, 1.23, 1.79, 1.17, 0.70, 0.55, 0.48, 0.36, 0.38, 0.38, 0.38, 0.35, 0.31, 0.27, 0.23,
                0.14, 0.08, 0.05, 0.03],
    "6s5d 1D2": [5.45, 4.95, 4.07, 3.69, 3.59, 3.53, 3.39, 2.99, 2.74, 2.46, 2.33, 2.24, 2.00, 1.78, 1.44, 1.22,
                 0.69, 0.37, 0.25, 0.16],
    "5d2 1D2": [2.74, 2.54, 2.41, 1.87, 1.57, 1.51, 1.44, 1.28, 0.90, 0.51, 0.37, 0.31, 0.24, 0.19, 0.13, 0.10,
                0.043, 0.018, 0.012, 0.008],
    "6s6d 1D2": [1.21, 1.79, 2.25, 2.21, 1.96, 2.01, 2.28, 2.13, 1.89, 1.50, 1.29, 1.15, 0.97, 0.81, 0.62, 0.50,
                 0.25, 0.125, 0.083, 0.056],
    "6s6p 3P0": [0.133, 0.093, 0.092, 0.041, 0.024, 0.023, 0.025, 0.026, 0.016, 0.009, 0.005, 0.003, 0.002, 0.001,
                 0.0005, 0.00024, None, None, None, None],
    "6s6p 3P1": [0.553, 0.451, 0.460, 0.323, 0.269, 0.278, 0.289, 0.291, 0.257, 0.219, 0.192, 0.180, 0.161, 0.145,
                 0.122, 0.107, 0.066, 0.039, 0.028, 0.020],
    "6s6p 3P2": [0.664, 0.463, 0.461, 0.207, 0.122, 0.113, 0.127, 0.129, 0.080, 0.043, 0.025, 0.016, 0.009, 0.005,
                 0.002, 0.0012, None, None, None, None],
    "6s5d 3D1": [1.232, 0.983, 0.710, 0.385, 0.272, 0.199, 0.135, 0.068, 0.054, 0.029, 0.019, 0.013, 0.0068,
                 0.0036, 0.0014, 0.0007, None, None, None, None],
    "6s5d 3D2": [2.130, 1.712, 1.247, 0.710, 0.524, 0.404, 0.297, 0.178, 0.150, 0.102, 0.084, 0.072, 0.057, 0.047,
                 0.035, 0.029, 0.016, 0.0083, 0.0055, 0.0037],
    "6s5d 3D3": [2.875, 2.293, 1.656, 0.899, 0.635, 0.464, 0.316, 0.159, 0.127, 0.067, 0.045, 0.031, 0.016, 0.0084,
                 0.0032, 0.0015, None, None, None, None],
}
FURSA_ION = dict(  # Table VII: total ionization (renormalized Dettmann & Karstensen)
    E=np.array([5.4, 6, 7, 8, 9, 10, 12, 15, 20, 30, 40, 50, 80, 100, 150, 200, 400, 600]),
    Q=np.array([0.8, 3.3, 7.0, 10.1, 12.6, 12.0, 10.6, 10.2, 11.4, 12.8, 12.0, 11.1, 8.6, 7.9, 7.1, 5.6, 3.3, 2.4]),
)


def fursa(name):
    """(E [eV], sigma [cm2]) of a recommended Fursa et al (1999) cross section."""
    v = np.array([np.nan if x is None else x for x in FURSA_TABLES[name]], float)
    m = np.isfinite(v)
    return FURSA_E[m], v[m] * 1e-16
