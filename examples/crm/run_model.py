"""Background calculation for one CRM model (started by the notebooks).

    python run_model.py --model ba2 --workdir RUN/ba2 --stage all --jmax 25 --cores 32 --mem 100

Stages (each one is skipped if its result exists; re-run to continue):
  target       Target (dbsr_hf / mchf), NIST levels      -> target/, target_table.txt
  transitions  E1 (+ M1, E2 for metastables) A-values     -> transitions_E1.csv, ...
  scattering   inner region, partial waves J <= jmax      -> scat/h.nnn
  outer        collision strengths on an energy grid     -> omega_J<jmax>.npz
  crm          rate coefficients + A in one file          -> crm_<model>.json
  all          everything in this order

Models: see models.py (xe2_wang, xe2_ext, ba2, ba1).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pydbsr as db                      # noqa: E402
from pydbsr import crm                   # noqa: E402
from pydbsr.transitions import TransitionTable, transitions  # noqa: E402
from models import MODELS                # noqa: E402

TE_GRID = np.array([0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0])   # eV
K_PER_EV = 11604.518


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=sorted(MODELS))
    p.add_argument("--workdir", required=True)
    p.add_argument("--stage", default="all", choices=["target", "transitions", "scattering", "outer", "crm", "all"])
    p.add_argument("--levels", default=None, help="NIST level table (Wang xlsx for xe2_wang) instead of NIST ASD")
    p.add_argument("--emax-states", type=float, default=None, help="keep target states up to this energy, eV")
    p.add_argument("--no-corr", action="store_true", help="no correlation orbital")
    p.add_argument("--jmax", type=float, default=10.0)
    p.add_argument("--cores", type=int, default=16)
    p.add_argument("--mem", type=float, default=60.0, help="GB")
    p.add_argument("--hd-threads", type=int, default=8)
    p.add_argument("--scratch", default=None)
    p.add_argument("--scratch-gb", type=float, default=None)
    p.add_argument("--emax", type=float, default=40.0, help="max electron energy above the ground state, eV")
    p.add_argument("--de", type=float, default=0.0136, help="energy step, eV")
    p.add_argument("--multipoles", default="E1,E2",
                   help="radiative transitions (M1 is not used: dbsr_dmat3 stops with SIGSEGV for M1 on c-files)")
    a = p.parse_args(argv)

    wd = Path(a.workdir)
    wd.mkdir(parents=True, exist_ok=True)
    stages = ["target", "transitions", "scattering", "outer", "crm"] if a.stage == "all" else [a.stage]

    levels = None
    if a.levels:
        if a.levels.endswith(".xlsx"):
            from pydbsr import reference
            levels = reference.read_xlsx(a.levels).levels
        else:
            levels = db.nist.read_levels(a.levels)
    kw = dict(jobs=min(a.cores, 8), levels=levels, corr=not a.no_corr)
    if a.emax_states is not None:
        kw["emax_ev"] = a.emax_states
    log(f"model {a.model}: target")
    tg, states = MODELS[a.model](wd / "target", **kw)
    (wd / "target_table.txt").write_text(tg.table(states))
    log(f"{len(states)} states with NIST levels (of {len(tg.states)})")

    if "transitions" in stages:
        for kind in a.multipoles.split(","):
            out = wd / f"transitions_{kind}.csv"
            if out.exists():
                continue
            log(f"transitions {kind}")
            tr = transitions(tg, states, kinds=(kind,), jobs=a.cores, workdir=wd / "_transitions", progress=log)
            if not tr:
                log(f"no {kind} lines (all pairs failed?): {out} not written")
                continue
            TransitionTable(tr, states).to_csv(out)
            log(f"{len(tr)} {kind} lines -> {out}")

    sdir = wd / "scat"
    sc = None
    if "scattering" in stages or "outer" in stages:
        pw = db.partial_waves(tg.ion.nelc, a.jmax)
        sc = db.Scattering(tg, sdir, states=states, partial_waves=pw, exp_energies=True)
        if "scattering" in stages:
            if not (sdir / "cfg.001").exists() or len(list(sdir.glob("cfg.[0-9][0-9][0-9]"))) < len(pw):
                log("prepare / prep / conf")
                sc.prepare()
                sc.run_prep()
                sc.run_conf()
            sc.complete_target_orb()
            log(f"{len(pw)} partial waves, J <= {a.jmax:g}")
            sc.run_streamed(cores=a.cores, mem_gb=a.mem, hd_threads=a.hd_threads,
                            scratch=a.scratch, scratch_gb=a.scratch_gb)

    ofile = wd / f"omega_J{a.jmax:g}.npz"
    if "outer" in stages and not ofile.exists():
        log("outer region")
        energies = np.arange(0.005, a.emax, a.de)
        cs = sc.outer().collision_strengths(energies, jobs=a.cores, per_partial_wave=True)
        cs.save(ofile)
        log(f"-> {ofile}")

    if "crm" in stages and ofile.exists():
        cs = db.CollisionStrengths.load(ofile)
        data = build_crm(a.model, cs, states, [wd / f"transitions_{k}.csv" for k in a.multipoles.split(",")])
        out = data.save(wd / f"crm_{a.model}.json")
        log(f"CRM data -> {out}")
    log("done")


def build_crm(species, cs, states, a_files) -> crm.CRMData:
    """Rate coefficients (Maxwellian, from Upsilon) for all pairs + A-values (NIST energies)."""
    import csv
    by = {s.name: s for s in states}
    names = [n for n in cs.names if n in by]
    lab = [by[n].nist_label or n for n in names]
    data = crm.CRMData(species, names, [by[n].g for n in names], [by[n].exp_energy_cm for n in names], lab, TE_GRID)
    T = TE_GRID * K_PER_EV
    for a_i, i in enumerate(names):
        for j in names[a_i + 1:]:
            lo, up = (i, j) if by[i].exp_energy_cm <= by[j].exp_energy_cm else (j, i)
            data.add_rate(lo, up, cs.rate(lo, up, T), "pydbsr")
    for f in a_files:
        if not Path(f).exists():
            continue
        for r in csv.DictReader(open(f)):
            if r["upper"] in data.index and r["lower"] in data.index and r["A_exp"]:
                data.add_A(r["upper"], r["lower"], float(r["A_exp"]) + data.A.get(
                    (data.index[r["upper"]], data.index[r["lower"]]), 0.0))
    return data


if __name__ == "__main__":
    main()
