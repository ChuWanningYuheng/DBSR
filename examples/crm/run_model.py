"""Background calculation of atomic data for one model (started by the notebooks).

    python run_model.py --model ba2 --workdir RUN/ba2 --stage all --jmax 20 --cores 32 --mem 100

Stages (each one is skipped if its result exists; re-run to continue):
  target       Target (dbsr_hf / mchf), NIST levels assigned   -> target/, target_table.txt
  transitions  A-values, gf, S (E1, E2) between target states   -> transitions_E1.csv, transitions_E2.csv
               (--uppers 29,39,41,42,51: only the branches of these upper levels, NIST/Wang numbers)
  scattering   inner region, partial waves J <= jmax            -> scat/h.nnn
  outer        collision strengths on an energy grid           -> omega_J<jmax>.npz
  sigma        excitation cross sections, Wang's xlsx layout   -> sigma_<model>.xlsx
  rates        Maxwellian <sigma v>(Te) on the --te grid        -> rates_<model>.xlsx / .csv
  all          everything above in this order
  bound        bound (N+1)-electron states (dbsr_hd3 itype=-1; separate run, e.g. --jmax 4
               in another workdir: the matrices are rebuilt)  -> bound_states.csv

Level numbers in all output files are NIST numbers (state.nist_no; for xe2_wang
with --levels CrossSectionsIon.xlsx they are Wang's numbers).

Models: see models.py (xe2_wang, ba2, ba1).
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pydbsr as db                      # noqa: E402
from pydbsr import rates, reference      # noqa: E402
from pydbsr.nist import Level            # noqa: E402
from pydbsr.transitions import TransitionTable, transitions  # noqa: E402
from models import MODELS                # noqa: E402

TE_DEFAULT = "0.3,0.5,0.7,1,1.5,2,3,5,7,10,15,20"
K_PER_EV = 11604.518


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=sorted(MODELS))
    p.add_argument("--workdir", required=True)
    p.add_argument("--stage", default="all",
                   choices=["target", "transitions", "scattering", "outer", "sigma", "rates", "all", "bound"])
    p.add_argument("--levels", default=None, help="level table (Wang xlsx for xe2_wang) instead of NIST ASD")
    p.add_argument("--emax-states", type=float, default=None, help="keep target states up to this energy, eV")
    p.add_argument("--no-corr", action="store_true", help="no correlation orbital")
    p.add_argument("--uppers", default=None, help="transitions only from these upper levels (NIST/Wang numbers)")
    p.add_argument("--jmax", type=float, default=10.0)
    p.add_argument("--cores", type=int, default=16)
    p.add_argument("--mem", type=float, default=60.0, help="GB")
    p.add_argument("--hd-threads", type=int, default=8)
    p.add_argument("--scratch", default=None)
    p.add_argument("--scratch-gb", type=float, default=None)
    p.add_argument("--emax", type=float, default=40.0, help="max electron energy above the ground state, eV")
    p.add_argument("--de", type=float, default=0.0136, help="energy step, eV")
    p.add_argument("--sigma-initial", default="all",
                   help="initial levels (NIST numbers, comma separated) written to sigma_<model>.xlsx; 'all' = every pair")
    p.add_argument("--omega-pw", default="auto", choices=["auto", "yes", "no"],
                   help="keep Omega of every partial wave (convergence checks); auto: only up to 40 states")
    p.add_argument("--te", default=TE_DEFAULT, help="Te grid for the rate coefficients, eV (comma separated)")
    p.add_argument("--multipoles", default="E1,E2",
                   help="radiative transitions (M1 is not used: dbsr_dmat3 stops with SIGSEGV for M1 on c-files)")
    a = p.parse_args(argv)

    wd = Path(a.workdir)
    wd.mkdir(parents=True, exist_ok=True)
    stages = ["target", "transitions", "scattering", "outer", "sigma", "rates"] if a.stage == "all" else [a.stage]

    levels = None
    if a.levels:
        levels = reference.read_xlsx(a.levels).levels if a.levels.endswith(".xlsx") else db.nist.read_levels(a.levels)
    kw = dict(jobs=min(a.cores, 8), levels=levels, corr=not a.no_corr)
    if a.emax_states is not None:
        kw["emax_ev"] = a.emax_states
    log(f"model {a.model}: target")
    tg, states = MODELS[a.model](wd / "target", **kw)
    done = wd / "scat" / "pydbsr_scattering.json"
    if done.exists():        # existing scattering run: its own states (the level labels may be updated)
        import json
        by_name = {s.name: s for s in tg.states}
        states = [by_name[d["name"]] for d in json.loads(done.read_text())["states"]]
    (wd / "target_table.txt").write_text(tg.table(states))
    log(f"{len(states)} states with NIST levels (of {len(tg.states)})")
    by_no = {s.nist_no: s for s in states if s.nist_no is not None}

    if "transitions" in stages:
        for kind in a.multipoles.split(","):
            out = wd / f"transitions_{kind}.csv"
            if out.exists():
                continue
            pairs = None
            if a.uppers:
                ups = [by_no[int(u)] for u in a.uppers.split(",")]
                pairs = [(s.name, u.name) for u in ups for s in states
                         if s.name != u.name and s.exp_energy_cm is not None and s.exp_energy_cm < u.exp_energy_cm]
            log(f"transitions {kind}" + (f" from levels {a.uppers}" if a.uppers else ""))
            tr = transitions(tg, states, kinds=(kind,), pairs=pairs, jobs=a.cores, workdir=wd / "_transitions",
                             progress=log)
            if not tr:
                log(f"no {kind} lines (all pairs failed?): {out} not written")
                continue
            TransitionTable(tr, states).to_csv(out)
            log(f"{len(tr)} {kind} lines -> {out}")

    sdir = wd / "scat"
    sc = None
    if {"scattering", "outer", "bound"} & set(stages):
        pw = db.partial_waves(tg.ion.nelc, a.jmax)
        sc = db.Scattering(tg, sdir, states=states, partial_waves=pw, exp_energies=True)
        if "scattering" in stages or "bound" in stages:
            if not (sdir / "cfg.001").exists() or len(list(sdir.glob("cfg.[0-9][0-9][0-9]"))) < len(pw):
                log("prepare / prep / conf")
                sc.prepare()
                sc.run_prep()
                sc.run_conf()
            elif not (sdir / "cfg_conf3").exists() and len(list(sdir.glob("h.[0-9][0-9][0-9]"))) < len(pw):
                sc.run_conf()                # run made by older pydbsr with waves to do: clean cfg.nnn first
            sc.complete_target_orb()
            log(f"{len(pw)} partial waves, J <= {a.jmax:g}")
            itype = -1 if "bound" in stages else 0
            sc.run_streamed(cores=a.cores, mem_gb=a.mem, hd_threads=a.hd_threads, itype=itype,
                            hd_args={"msol": 30} if itype == -1 else None,
                            scratch=a.scratch, scratch_gb=a.scratch_gb)
        if "bound" in stages:
            from pydbsr.scattering import read_dbound_tab
            db.run("dbound_tab", [], sdir, log="dbound_tab.out")
            rows = read_dbound_tab(sdir / "dbound_tab")
            if rows:
                with open(wd / "bound_states.csv", "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0]))
                    w.writeheader()
                    w.writerows(rows)
            log(f"{len(rows)} bound states -> {wd / 'bound_states.csv'}")

    ofile = wd / f"omega_J{a.jmax:g}.npz"
    if "outer" in stages and not ofile.exists():
        log("outer region")
        energies = np.arange(0.005, a.emax, a.de)
        per_pw = a.omega_pw == "yes" or (a.omega_pw == "auto" and len(states) <= 40)
        cs = sc.outer().collision_strengths(energies, jobs=a.cores, per_partial_wave=per_pw)
        cs.save(ofile)
        log(f"-> {ofile}")

    if {"sigma", "rates"} & set(stages) and ofile.exists():
        cs = db.CollisionStrengths.load(ofile, partial_waves=False)
        lv, sig = excitation_sigma(cs, states)
        if "sigma" in stages:
            keep = sig if a.sigma_initial == "all" else \
                {k: v for k, v in sig.items() if k[0] in {int(x) for x in a.sigma_initial.split(",")}}
            out = reference.write_xlsx(wd / f"sigma_{a.model}.xlsx", lv, keep, sheet_of=lambda k: f"from {k[0]}")
            log(f"{len(keep)} cross sections -> {out}")
        if "rates" in stages:
            te = np.array([float(x) for x in a.te.split(",")])
            by_no_st = {s.nist_no: s for s in states}

            def e1(i, j):                      # E1-allowed pair: dipole tail of sigma above the grid
                a_, b_ = by_no_st[i], by_no_st[j]
                return a_.parity != b_.parity and abs(a_.two_j - b_.two_j) <= 2 and a_.two_j + b_.two_j > 0
            # <sigma v> from the cross sections with a tail above the computed grid (as notebooks 02, 05);
            # cs.rate integrates Upsilon over the grid only and is too low at high Te
            k = {key: rates.rates_on_grid(E, s_, te, (by_no_st[key[1]].exp_energy_cm - by_no_st[key[0]].exp_energy_cm)
                                          / 8065.544, tail="dipole" if e1(*key) else "1/E")
                 for key, (E, s_) in sig.items()}
            out = reference.write_rates_xlsx(wd / f"rates_{a.model}.xlsx", te, k,
                                             note=f"{a.model}: Maxwellian <sigma v> (cm3/s), excitation i->j "
                                                  f"(NIST numbers), from sigma of {ofile.name} (tail above the grid: ln E/E E1, 1/E other)")
            log(f"rates -> {out}")
    log("done")


def excitation_sigma(cs, states):
    """Level table + {(i, j): (E_incident, sigma)} for all excitations i -> j (NIST numbers)."""
    st = sorted((s for s in states if s.name in cs.names and s.nist_no is not None), key=lambda s: s.exp_energy_cm)
    lv = []
    for s in st:
        conf, term = s.config, ""
        if s.nist_label and " J=" in s.nist_label:
            head = s.nist_label.rsplit(" J=", 1)[0]
            conf, _, term = head.rpartition(" ")
        lv.append(Level(conf, term, s.two_j, s.exp_energy_cm, s.parity, s.nist_no))
    sig = {}
    for a_i, i in enumerate(st):
        for j in st[a_i + 1:]:
            E = cs.incident_energy(i.name)
            s = cs.sigma(i.name, j.name)
            m = np.isfinite(s) & (E > 0)
            if m.any():
                sig[(i.nist_no, j.nist_no)] = (E[m], s[m])
    return sorted(lv, key=lambda x: x.no), sig


if __name__ == "__main__":
    main()
