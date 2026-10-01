"""Target models for the CRM calculations (used by run_model.py and the notebooks).

Each model returns a computed :class:`pydbsr.Target` with NIST levels
assigned (thresholds), plus the states to use in the close-coupling run.

  xe2_wang  e + Xe+, as Wang et al (2019): 5s2 5p5, 5s 5p6, 5p4 5d, 6s, 7s, 6p
            (6d correlation orbital for the term dependence of 5d)
  xe2_ext   e + Xe+, extended: + 5p4 6d, 7p, 4f — upper levels of the visible
            lines 6d/7s/4f -> 6p, 5d (class C in docs/crm/lines.md) and the
            cascades into 6p
  ba2       e + Ba+: 6s, 5d, 7s, 6d, 8s, 7d | 6p, 7p, 4f, 5f
  ba1       e + Ba : 6s2, 6s5d, 5d2, 6p2, 6s7s, 6s6d | 6s6p, 5d6p, 6s7p, 6s4f

These are starting models without core polarisation: check the energies
(target table) and the A-values (notebooks 01, 04, 05) before trusting the
cross sections.
"""
from __future__ import annotations

from pathlib import Path

import pydbsr as db

MODELS = {}


def model(fn):
    MODELS[fn.__name__] = fn
    return fn


def _load_or_new(tdir, ion, core, grid, max_it=60):
    tdir = Path(tdir)
    if (tdir / "pydbsr_target.json").exists():
        return db.Target.load(tdir), True
    return db.Target(ion, core=core, workdir=tdir, max_it=max_it, grid=grid), False


def _finish(tg, levels, jobs, emax_ev=None):
    if not tg.states:
        tg.compute(jobs=jobs)
    db.nist.assign(tg.states, levels, verbose=False)
    tg.save()
    states = [s for s in tg.states if s.nist_no is not None]
    if emax_ev is not None:
        states = [s for s in states if s.exp_energy_cm / 8065.544 <= emax_ev]
    return tg, states


@model
def xe2_wang(tdir, jobs=4, levels=None, corr=True, emax_ev=None):
    """68 states as in Wang et al (2019); ``levels``: Wang's NIST table (keeps their numbering)."""
    tg, done = _load_or_new(tdir, db.Ion("Xe", 1), "[Kr]4d10", {"rmax": 50.0, "hmax": 0.5})
    if not done:
        tg.add("5s2 5p5")
        tg.add(["5s 5p6", "5s2 5p4 5d", "5s2 5p4 6s", "5s2 5p4 7s"],
               correlation=["5s2 5p4 6d"] if corr else None, mchf_max_it=150)
        tg.add(["5s2 5p5", "5s2 5p4 6p"])
    return _finish(tg, levels or "Xe II", jobs, emax_ev)


@model
def xe2_ext(tdir, jobs=4, levels=None, corr=True, emax_ev=19.2):
    """Wang model + 5p4 6d, 7p, 4f.  ``emax_ev`` = 19.2 keeps the levels up to
    (1D2)6d (the upper levels of 418.0, 419.3, 439.6, 461.5 nm); every state
    below is kept (close coupling must not skip states in between)."""
    tg, done = _load_or_new(tdir, db.Ion("Xe", 1), "[Kr]4d10", {"rmax": 50.0, "hmax": 0.5})
    if not done:
        tg.add("5s2 5p5")
        tg.add(["5s 5p6", "5s2 5p4 5d", "5s2 5p4 6s", "5s2 5p4 7s", "5s2 5p4 6d"],
               correlation=["5s2 5p4 7d"] if corr else None, mchf_max_it=150)
        tg.add(["5s2 5p5", "5s2 5p4 6p", "5s2 5p4 7p", "5s2 5p4 4f"])
    return _finish(tg, levels or "Xe II", jobs, emax_ev)


@model
def ba2(tdir, jobs=4, levels=None, corr=False, emax_ev=None):
    """Ba+: one valence electron above [Xe]."""
    tg, done = _load_or_new(tdir, db.Ion("Ba", 1), "[Xe]", {"rmax": 45.0, "hmax": 0.5})
    if not done:
        tg.add("6s")
        tg.add(["6s", "5d", "7s", "6d", "8s", "7d"])
        tg.add(["6p", "7p", "4f", "5f"])
    return _finish(tg, levels or "Ba II", jobs, emax_ev)


@model
def ba1(tdir, jobs=4, levels=None, corr=False, emax_ev=4.1):
    """Ba: two valence electrons above [Xe]; all states up to 4.1 eV (6s8p 1P1 at 4.04 eV)."""
    tg, done = _load_or_new(tdir, db.Ion("Ba", 0), "[Xe]", {"rmax": 50.0, "hmax": 0.5})
    if not done:
        tg.add("6s2")
        tg.add(["6s2", "6s 5d", "5d2", "6p2", "6s 7s", "6s 6d"])
        tg.add(["6s 6p", "5d 6p", "6s 7p", "6s 4f"])
    return _finish(tg, levels or "Ba I", jobs, emax_ev)
