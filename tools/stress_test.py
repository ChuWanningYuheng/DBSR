#!/usr/bin/env python3
"""Stress test of pydbsr / DBSR for electron scattering on heavy ions (Xe+).

Checks fundamental properties on real DBSR output, not on mock data:

 1. Dirac equation (hydrogen-like Xe53+, point nucleus, no QED): orbital
    energies against the exact Dirac formula, large (P) / small (Q) component
    ratio of nodeless states against the exact value, normalisation, zero
    boundary values.
 2. Xe+ target: normalisation and boundary values of all orbitals, kinetic
    balance Q ~ (P' + kappa P / r) / 2c in the valence region, fine-structure
    splitting of 5p5 2P (ab initio) against NIST, grid convergence.
 3. Coulomb functions used for matching: Wronskian F'G - FG' = 1 over a wide
    (l, eta, rho) range incl. the classically forbidden region, against mpmath.
 4. Outer region on the h-files: symmetry of the long-range coupling matrices,
    completeness of the channel set, K-matrix symmetry *before*
    symmetrisation (flux conservation), unitarity of S, |T|^2 <= 4, Omega >= 0,
    Omega_ij = Omega_ji, finite results exactly at / just above / just below
    every threshold, reproducibility of the parallel run, effect of the
    relativistic kinematics neglected in the outer region.
 5. Inner-region matrices (dbsr_mat.nnn): lower-triangle storage (symmetric by
    construction), finite elements, overlap matrix positive definite.
 6. Target consistency reported by dbsr_mat3 (off-diagonal <i|H|j>, <i|j>).
 7. Library robustness: process-group kill on timeout, error detection, unit
    tests with warnings turned into errors.

All numpy floating-point errors and all Python warnings are errors.

Usage:
    python tools/stress_test.py --scat RUN/scat [--scat ...] --target RUN/target
                                [--log stress_log.json]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
import warnings
from pathlib import Path

import numpy as np

warnings.simplefilter("error")
np.seterr(all="raise")

# use the compiled COULFG as in production runs (also from a source checkout)
if "PYDBSR_COULOMB_LIB" not in os.environ and os.environ.get("DBSR_BIN"):
    for _c in Path(os.environ["DBSR_BIN"]).glob("libpydbsr_coulomb.*"):
        os.environ["PYDBSR_COULOMB_LIB"] = str(_c)
        break

import pydbsr as db  # noqa: E402
from pydbsr import io  # noqa: E402
from pydbsr.coulomb import _mp_fg, backend, coulomb_fg  # noqa: E402
from pydbsr.outer import HData, closed_logderiv, kmatrix, read_h, t_matrix  # noqa: E402
from pydbsr.runner import DBSRError, run, which  # noqa: E402

C_DBSR = 137.03599976          # speed of light used by DBSR (ZCOM MOD_zconst)
RESULTS: list[dict] = []


def check(name, ok, detail=""):
    RESULTS.append({"test": name, "pass": bool(ok), "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    return ok


def section(title):
    print(f"\n=== {title}", flush=True)


def guarded(fn):
    def wrapper(*a, **kw):
        try:
            return fn(*a, **kw)
        except Exception as e:                      # a crash is a failed test, not an aborted run
            check(f"{fn.__name__} ran without exception", False, f"{type(e).__name__}: {e}")
            traceback.print_exc()
    return wrapper


# ---------------------------------------------------------------------------
# 1. Dirac hydrogen-like Xe53+
# ---------------------------------------------------------------------------
def dirac_energy(n, kappa, Z, c=C_DBSR):
    g = np.sqrt(kappa ** 2 - (Z / c) ** 2)
    return c * c / np.sqrt(1.0 + (Z / c / (n - abs(kappa) + g)) ** 2) - c * c


@guarded
def test_dirac_hydrogenic(tmp: Path):
    section("1. Dirac equation: hydrogen-like Xe53+ (point nucleus, no QED)")
    Z = 54
    grid = ["hi=0.002", "he=0.1", "hmax=0.5", "rmax=20"]
    cases = {"1s": ("1s(1)", [(1, -1)]), "2p": ("2p(1)", [(2, 1), (2, -2)])}
    for tag, (conf, orbs) in cases.items():
        d = tmp / f"hyd_{tag}"
        d.mkdir()
        run("dbsr_hf", ["h", "atom=Xe", "ion=H", f"conf={conf}", "nuclear=point", "mode_SE=0",
                        "mode_VP=0", *grid], d)
        knot = (d / "h.knot").read_text()
        check(f"{tag}: point nucleus used", "nuclear =  point" in knot or "nuclear = point" in knot)
        funcs = {(f.n, f.kappa): f for f in io.radial_functions(d / "h.bsw", d / "h.knot")}
        for n, kappa in orbs:
            f = funcs[(n, kappa)]
            e_exact = dirac_energy(n, kappa, Z)
            rel = abs(f.energy - e_exact) / abs(e_exact)
            check(f"{f.name} energy vs exact Dirac", rel < 1e-8,
                  f"DBSR {f.energy:.8f}  exact {e_exact:.8f}  rel.err {rel:.1e}")
            check(f"{f.name} normalisation", abs(f.norm() - 1) < 1e-5, f"|1 - int(P^2+Q^2)| = {abs(f.norm() - 1):.1e}")
            check(f"{f.name} P(a)=Q(a)=0", f.P[-1] == 0 and f.Q[-1] == 0 and np.all(np.isfinite(f.P)),
                  f"P,Q at r={f.r[-1]:.2f}: {f.P[-1]:.1e}, {f.Q[-1]:.1e}")
            if n == abs(kappa) and kappa < 0:          # nodeless: Q/P = -sqrt((1-eps)/(1+eps))
                eps = (f.energy + C_DBSR ** 2) / C_DBSR ** 2
                exact = -np.sqrt((1 - eps) / (1 + eps))
                m = np.abs(f.P) > 1e-4 * np.abs(f.P).max()
                far = m & (f.r > 1e-3)
                dev = np.max(np.abs(f.Q[far] / f.P[far] - exact))
                check(f"{f.name} small/large component ratio Q/P", dev < 1e-5 * abs(exact),
                      f"exact {exact:.7f}, max dev {dev:.1e} over r in [{f.r[far][0]:.1e}, {f.r[far][-1]:.2f}]")
                near = m & (f.r <= 1e-3)
                if near.any():   # point nucleus: P ~ r^gamma (gamma < 1) is not a polynomial
                    print(f"    (r < 1e-3 a0, point-nucleus singularity: max dev of Q/P "
                          f"{np.max(np.abs(f.Q[near] / f.P[near] - exact)):.1e}; finite nuclei are regular)")


# ---------------------------------------------------------------------------
# 2. Xe+ target
# ---------------------------------------------------------------------------
@guarded
def test_target(target_dir: Path, tmp: Path, nist_splitting_cm: float = 10537.0):
    section(f"2. Xe+ target orbitals and fine structure ({target_dir})")
    tg = db.Target.load(target_dir)
    ref = tg.specs[0].name
    inp = target_dir / f"{ref}.inp"
    if inp.exists():
        txt = inp.read_text(errors="replace")
        par = {}
        for key in ("nuclear", "mbreit", "mode_SE", "mode_VP", "core"):
            line = next((x for x in txt.splitlines() if x.strip().startswith(key)), "")
            par[key] = line.split("=", 1)[1].split("-")[0].strip() if "=" in line else "?"
        print(f"    Hamiltonian: Dirac-Coulomb, nucleus = {par['nuclear']}, mbreit = {par['mbreit']} "
              f"({'no Breit, no QED' if par['mbreit'] == '0' else 'Breit + QED (1st order)'}); core {par['core']}")
        check("finite (Fermi) nuclear charge distribution in the target", par["nuclear"].lower().startswith("fermi"),
              par["nuclear"])
    funcs = io.radial_functions(target_dir / f"{ref}.bsw", target_dir / f"{ref}.knot")
    norms = [abs(f.norm() - 1) for f in funcs]
    check("reference orbitals normalised", max(norms) < 1e-5, f"{len(funcs)} orbitals, max |1-N| = {max(norms):.1e}")
    check("orbitals vanish at the R-matrix boundary",
          all(f.P[-1] == 0 and f.Q[-1] == 0 for f in funcs), f"r_max(orbital) <= {max(f.r[-1] for f in funcs):.2f}")
    from scipy.interpolate import CubicSpline
    worst = 0.0
    for f in funcs:
        if f.n < 4:
            continue
        m = (f.r > 0.5) & (np.abs(f.P) > 1e-3 * np.abs(f.P).max())
        dP = CubicSpline(f.r, f.P)(f.r, 1)
        q_kb = (dP + f.kappa * f.P / np.where(f.r > 0, f.r, 1)) / (2 * C_DBSR)
        dev = np.sqrt(np.sum((f.Q[m] - q_kb[m]) ** 2) / np.sum(f.Q[m] ** 2))
        worst = max(worst, dev)
    check("kinetic balance Q = (P' + kappa P/r)/2c of valence orbitals (r > 0.5)", worst < 2e-2,
          f"max relative L2 deviation {worst:.1e} (expected O((e-V)/2c^2))")
    e = {f.name: f.energy for f in funcs}
    if "2p-" in e and "2p" in e:
        print(f"    inner-shell spin-orbit: e(2p1/2) - e(2p3/2) = {e['2p-'] - e['2p']:.3f} a.u.; "
              f"e(1s) = {e['1s']:.3f} a.u.")
        check("inner-shell spin-orbit splitting from the Dirac equation", e["2p"] - e["2p-"] > 5.0,
              f"{(e['2p'] - e['2p-']) * 27.211386:.1f} eV")
    ref_conf = tg.specs[0].conf
    j32 = sorted((s for s in tg.states if s.config == ref_conf and s.two_j == 3), key=lambda s: s.energy)
    j12 = sorted((s for s in tg.states if s.config == ref_conf and s.two_j == 1), key=lambda s: s.energy)
    if j32 and j12:
        split = (j12[0].energy - j32[0].energy) * io_au_cm()
        rel = abs(split - nist_splitting_cm) / nist_splitting_cm
        check("5p5 2P1/2 - 2P3/2 splitting (ab initio) vs NIST", rel < 0.05,
              f"{split:.0f} cm-1 vs {nist_splitting_cm:.0f} cm-1 ({100 * rel:.1f} %)")
    # grid convergence of the reference calculation (hi = 0.25/Z -> 0.05/Z)
    base_sols = sorted(io.read_j(target_dir / f"{ref}.j"), key=lambda s: s.energy)
    d = tmp / "gridconv"
    d.mkdir()
    base = [a for a in tg._hf_args(tg.specs[0]) if not a.startswith(("hi=", "he="))]
    run("dbsr_hf", [*base, *tg._grid_args(), "hi=0.05", "he=0.1"], d)
    run("dbsr_hf", [ref, "term=jj", "varied=none", "max_it=1", *tg._grid_args(), "hi=0.05", "he=0.1"], d)
    sols = sorted(io.read_j(d / f"{ref}.j"), key=lambda s: s.energy)
    if len(sols) >= 2 and len(base_sols) >= 2:
        s1 = (base_sols[1].energy - base_sols[0].energy) * io_au_cm()
        s2 = (sols[1].energy - sols[0].energy) * io_au_cm()
        check("fine structure converged w.r.t. the B-spline grid (hi 0.25 -> 0.05)",
              abs(s2 - s1) < 5.0, f"{s1:.1f} vs {s2:.1f} cm-1; "
              f"E_tot {base_sols[0].energy:.6f} vs {sols[0].energy:.6f} a.u.")


def io_au_cm():
    from pydbsr.constants import AU_CM
    return AU_CM


# ---------------------------------------------------------------------------
# 3. Coulomb functions
# ---------------------------------------------------------------------------
@guarded
def test_coulomb():
    section(f"3. Coulomb functions for the matching ({backend()})")
    ls = np.array([0, 1, 2, 3, 5, 8, 12, 20, 30, 45, 60])
    etas = -np.array([1e-3, 0.05, 0.3, 1.0, 3.0, 10.0, 40.0, 150.0])     # attractive, z = +1
    rhos = np.logspace(-1, 2.6, 23)
    L, E, R = np.meshgrid(ls, etas, rhos, indexing="ij")
    L, E, R = L.ravel(), E.ravel(), R.ravel()
    # the range met in practice: rho = k r_match >= k a; skip values beyond double range
    with np.errstate(all="ignore"):
        logG = np.array([float(__import__("mpmath").log(abs(_approx_g(l, e, r)))) for l, e, r in zip(L, E, R)])
    keep = logG < 600
    L, E, R = L[keep], E[keep], R[keep]
    t = time.time()
    F, G, Fp, Gp = coulomb_fg(L, E, R)
    dt = time.time() - t
    fin = np.all(np.isfinite(F) & np.isfinite(G) & np.isfinite(Fp) & np.isfinite(Gp))
    check("F, G, F', G' finite", fin, f"{L.size} points, l <= {L.max()}, eta in [{E.min()}, {E.max()}], "
          f"rho in [{R.min():.2f}, {R.max():.0f}], {dt:.1f} s")
    W = Fp * G - F * Gp
    dev = np.max(np.abs(W - 1.0))
    check("Wronskian F'G - FG' = 1", dev < 1e-9, f"max |W - 1| = {dev:.1e}")
    rng = np.random.default_rng(1)
    idx = rng.choice(L.size, 60, replace=False)
    worst = 0.0
    for i in idx:
        ref = _mp_fg(int(L[i]), float(E[i]), float(R[i]))
        for a, b in zip((F[i], G[i], Fp[i], Gp[i]), ref):
            worst = max(worst, abs(a - b) / max(abs(b), 1e-300))
    check("agreement with mpmath (60 random points)", worst < 1e-8, f"max rel. diff {worst:.1e}")


def _approx_g(l, eta, rho):
    import mpmath as mp
    with mp.workdps(20):
        return mp.coulombg(int(l), eta, rho) if rho > 5 or l < 3 else mp.mpf(10) ** (
            float(mp.log10(mp.fac2(2 * l - 1))) - l * float(mp.log10(rho)))


# ---------------------------------------------------------------------------
# 4. outer region
# ---------------------------------------------------------------------------
def expected_channels(two_j_tot, parity, two_j_t, parity_t):
    """All (target, kappa) allowed by angular momentum and parity."""
    out = []
    for t, (jt, pt) in enumerate(zip(two_j_t, parity_t)):
        for two_jc in range(abs(two_j_tot - jt), two_j_tot + jt + 1, 2):
            for kappa in (-(two_jc + 1) // 2, (two_jc + 1) // 2):   # l = j -/+ 1/2
                l = kappa if kappa > 0 else -kappa - 1
                if pt * (-1) ** l == parity:
                    out.append((t, kappa))
    return sorted(out)


@guarded
def test_outer(scat: Path):
    section(f"4. Outer region ({scat})")
    hfiles = sorted(scat.glob("h.[0-9][0-9][0-9]"))
    if not hfiles:
        check("h-files present", False, str(scat))
        return
    hd = read_h(hfiles)
    print(f"    {len(hd.blocks)} partial waves, {hd.ntarg} target states, a = {hd.ra} a0, "
          f"z = {hd.z}, b = {hd.rb}")
    # target parities from the scattering json
    sc = db.Scattering.load(scat, progress=None)
    by_name = {s.name: s for s in sc.states}
    names = sc.h_target_names()
    ptarg = np.array([by_name[n].parity for n in names])
    for blk in hd.blocks:
        tag = f"2J={blk.two_j}{'+' if blk.parity > 0 else '-'}"
        asym = max((np.max(np.abs(blk.cf[:, :, k] - blk.cf[:, :, k].T)) for k in range(blk.cf.shape[2])), default=0.0)
        check(f"{tag}: long-range coupling C_lambda symmetric", asym < 1e-10, f"max |C - C^T| = {asym:.1e}")
        lk = np.where(blk.kappa > 0, blk.kappa, -blk.kappa - 1)
        check(f"{tag}: l consistent with kappa", np.array_equal(lk, blk.l))
        got = sorted(zip(blk.target.tolist(), blk.kappa.tolist()))
        exp = expected_channels(blk.two_j, blk.parity, hd.two_j_targ, ptarg)
        missing = sorted(set(exp) - set(got))
        extra = sorted(set(got) - set(exp))
        check(f"{tag}: channel set complete", not missing and not extra and len(got) == len(exp),
              f"{len(got)} channels, expected {len(exp)}; missing {missing[:5]} extra {extra[:5]}")
        check(f"{tag}: R-matrix poles / amplitudes finite", np.all(np.isfinite(blk.poles)) and np.all(np.isfinite(blk.w)),
              f"{blk.poles.size} poles in [{blk.poles.min():.3f}, {blk.poles.max():.1f}] a.u.")

    # no negative-energy (positron-like) solutions of the Dirac Hamiltonian among the poles
    pmin = min(b.poles.min() for b in hd.blocks)
    check("no negative-energy Dirac states among the R-matrix poles", pmin > hd.etarg.min() - 10.0,
          f"lowest pole {pmin:.4f} a.u. (negative continuum would be near -2c^2 = {-2 * C_DBSR ** 2:.0f})")
    if hd.nz == 54 and hd.nelc == 53:
        b0 = next((b for b in hd.blocks if b.two_j == 0 and b.parity == 1), None)
        if b0 is not None:
            ip = (hd.etarg.min() - b0.poles.min()) * 27.211386
            check("lowest 2J=0+ pole = Xe 5p6 1S0 bound state: ionisation energy of Xe", abs(ip - 12.1298) < 0.5,
                  f"{ip:.3f} eV vs NIST 12.1298 eV")

    # energies: regular grid + exactly at / around every threshold
    e0 = hd.etarg.min()
    thr = np.unique(hd.etarg)
    grid = np.linspace(e0 + 1e-3, thr.max() + 0.5, 60)
    edge = np.concatenate([thr, thr + 1e-12, thr - 1e-12, thr + 1e-7, thr + 1e-4])
    energies = np.sort(np.concatenate([grid, edge[edge > e0 - 0.01]]))
    asym_max = unit_max = t2_max = 0.0
    n_k = 0
    t = time.time()
    for etot in energies:
        for blk in hd.blocks:
            K, op = kmatrix(blk, hd, etot, symmetrize=False)
            if op.size == 0:
                continue
            n_k += 1
            if not np.all(np.isfinite(K)):
                check("K finite", False, f"E = {etot}, 2J = {blk.two_j}")
                return
            asym_max = max(asym_max, np.max(np.abs(K - K.T)) / max(1.0, np.max(np.abs(K))))
            Ks = 0.5 * (K + K.T)
            n = Ks.shape[0]
            S = np.linalg.solve((np.eye(n) - 1j * Ks).T, (np.eye(n) + 1j * Ks).T).T
            unit_max = max(unit_max, np.max(np.abs(S.conj().T @ S - np.eye(n))))
            t2_max = max(t2_max, np.max(np.abs(t_matrix(Ks)) ** 2))
    dt = time.time() - t
    check("K-matrix symmetric before symmetrisation (flux conservation)", asym_max < 1e-6,
          f"max |K - K^T| / max(1,|K|) = {asym_max:.1e} over {n_k} K-matrices ({dt:.0f} s), incl. E = thresholds, +-1e-12")
    check("S-matrix unitary", unit_max < 1e-10, f"max |S^+S - 1| = {unit_max:.1e}")
    check("|T|^2 <= 4 (unitarity bound)", t2_max <= 4 + 1e-10, f"max |T|^2 = {t2_max:.4f}")

    # continuity at the thresholds (limit from above, attractive Coulomb field).
    # NB: on a 1e-7 a.u. scale |T|^2 may vary fast: Rydberg resonances of a closed
    # channel whose threshold lies just above (physical, not tested here)
    worst = 0.0
    for et in thr[1:]:
        for blk in hd.blocks:
            K1, op1 = kmatrix(blk, hd, et)
            K2, op2 = kmatrix(blk, hd, et + 1e-12)
            if op1.size and np.array_equal(op1, op2):
                T1, T2 = np.abs(t_matrix(K1)) ** 2, np.abs(t_matrix(K2)) ** 2
                big = T2 > 1e-4 * max(T2.max(), 1e-30)
                worst = max(worst, np.max(np.abs(T1[big] - T2[big]) / T2[big]))
    check("|T|^2 continuous at every threshold (E_t vs E_t + 1e-12 a.u.)", worst < 1e-5,
          f"max rel. difference {worst:.1e} at {thr.size - 1} thresholds")

    out = db.OuterRegion(hd, names=names)
    e_ev = np.concatenate([np.linspace(0.05, (thr.max() - e0) * 27.211386 + 5, 40), (thr - e0) * 27.211386])
    e_ev = np.sort(e_ev[e_ev > 0])
    cs1 = out.collision_strengths(e_ev, jobs=1)
    cs2 = out.collision_strengths(e_ev, jobs=2)
    om = cs1.omega
    check("Omega finite and >= 0", np.all(np.isfinite(om)) and om.min() >= 0, f"min {om.min():.2e}, max {om.max():.3f}")
    check("Omega_ij = Omega_ji (detailed balance)", np.max(np.abs(om - om.transpose(0, 2, 1))) < 1e-12 * max(1, om.max()))
    check("parallel run reproduces serial run", np.array_equal(cs1.omega, cs2.omega))
    sig = cs1.sigma(0, 1)
    check("sigma: NaN below threshold only, no inf", not np.any(np.isinf(sig)))

    # relativistic kinematics neglected outside r = a:  k^2 -> 2 e (1 + e / 2c^2).
    # Tested above the highest threshold (no closed-channel resonances); below,
    # the same correction only shifts the Rydberg resonances by e^2/2c^2.
    probe = thr.max() + np.array([0.05, 0.3, 0.8, 1.5])
    worst = 0.0
    for etot in probe:
        e_ch = etot - hd.etarg
        e_rel = etot - e_ch * (1 + e_ch / (2 * C_DBSR ** 2))
        hd_rel = HData(hd.nelc, hd.nz, hd.ntarg, hd.ra, hd.rb, e_rel, hd.two_j_targ, hd.parity_targ, hd.blocks)
        for blk in hd.blocks:
            K1, op = kmatrix(blk, hd, etot)
            K2, _ = kmatrix(blk, hd_rel, etot)
            if op.size:
                T1, T2 = np.abs(t_matrix(K1)) ** 2, np.abs(t_matrix(K2)) ** 2
                big = T1 > 1e-3 * T1.max()
                worst = max(worst, np.max(np.abs(T2[big] - T1[big]) / T1[big]))
    e_closed = 0.5                                   # a.u.: deepest closed channel of interest
    check("relativistic kinematics in the outer region negligible", worst < 1e-3,
          f"max rel. change of |T|^2 = {worst:.1e} at E <= {(probe.max() - e0) * 27.211:.0f} eV; "
          f"below thresholds: resonance shift <= {e_closed ** 2 / (2 * C_DBSR ** 2) * 27.211e3:.2f} meV")

    # propagation beyond a (long-range multipoles): converges with r_match
    etot = e0 + 1.2
    vals = []
    for rm in (None, 2 * hd.ra, 4 * hd.ra):
        T2 = []
        for blk in hd.blocks:
            K, op = kmatrix(blk, hd, etot, r_match=rm)
            T2.append(np.sum(np.abs(t_matrix(K)) ** 2) if op.size else 0.0)
        vals.append(np.array(T2))
    d1 = np.max(np.abs(vals[1] - vals[0]) / np.maximum(vals[0], 1e-12))
    d2 = np.max(np.abs(vals[2] - vals[1]) / np.maximum(vals[1], 1e-12))
    check("long-range propagation a -> 2a -> 4a converges", d2 <= d1 + 1e-6 and d2 < 0.05,
          f"sum|T|^2 change a->2a {d1:.1e}, 2a->4a {d2:.1e}")

    # closed-channel log-derivative exactly at the classical turning point
    l, z, r = 3.0, 1.0, 50.0
    kappa2 = -(l * (l + 1) / r ** 2 - 2 * z / r)
    ld = closed_logderiv([l], [kappa2 if kappa2 > 0 else 1e-6], z, r)
    check("closed-channel log-derivative finite at the turning point", np.all(np.isfinite(ld)), f"{ld[0]:.4f}")


# ---------------------------------------------------------------------------
# 5. inner-region matrices
# ---------------------------------------------------------------------------
@guarded
def test_matrices(scat: Path):
    section(f"5. Inner-region matrices dbsr_mat.nnn ({scat})")
    from pydbsr.outer import _FortranReader
    files = sorted(scat.glob("dbsr_mat.[0-9][0-9][0-9]"))
    if not files:
        print("    no dbsr_mat.nnn kept (deleted by run_streamed(cleanup=True)) - skipped")
        return
    import scipy.linalg as sl
    for f in files:
        fr = _FortranReader(f)
        try:
            ns, nch, npert = fr.ints()[:3]
            nsol = int(fr.ints()[0])
            ipsol = fr.ints().copy()
            bval = fr.reals().copy()
            fr.record()
            khm = nsol + npert
            mats = {}
            upper = nonfinite = 0
            for part in ("S", "H"):
                M = np.zeros((khm, khm))
                if part == "H":
                    M[np.arange(nsol), np.arange(nsol)] = bval
                else:
                    M[np.arange(nsol), np.arange(nsol)] = 1.0
                while True:
                    ich, jch = fr.ints()[:2]
                    if ich == 0:
                        break
                    v = fr.reals()
                    nonfinite += int(not np.all(np.isfinite(v)))
                    upper += int(ich < jch)
                    if ich <= nch and jch <= nch:
                        i1, i2, j1, j2 = ipsol[ich - 1], ipsol[ich], ipsol[jch - 1], ipsol[jch]
                        M[i1:i2, j1:j2] = v.reshape((j2 - j1, i2 - i1)).T
                    elif ich > nch and jch <= nch:
                        M[ipsol[nch] + ich - nch - 1, ipsol[jch - 1]:ipsol[jch]] = v
                    else:
                        M[ipsol[nch] + ich - nch - 1, ipsol[nch] + jch - nch - 1] = v[0]
                mats[part] = M
        finally:
            fr.close()
        check(f"{f.name}: only lower-triangle blocks stored (H, S symmetric by construction)", upper == 0,
              f"khm = {khm}, {nch} channels")
        check(f"{f.name}: all matrix elements finite", nonfinite == 0)
        S = np.tril(mats["S"]) + np.tril(mats["S"], -1).T
        t = time.time()
        smin = sl.eigh(S, eigvals_only=True, subset_by_index=[0, 0], driver="evr")[0]
        check(f"{f.name}: overlap matrix positive definite (no linear dependence)", smin > 1e-8,
              f"lowest eigenvalue of S = {smin:.2e} ({time.time() - t:.0f} s)")
        del S, mats


# ---------------------------------------------------------------------------
# 6. target consistency
# ---------------------------------------------------------------------------
@guarded
def test_target_consistency(scat: Path, tol_h: float = 1e-4, tol_s: float = 1e-5):
    section(f"6. Target states as seen by dbsr_mat3 ({scat})")
    sc = db.Scattering.load(scat, progress=None)
    err = sc.target_errors()
    names = sc.h_target_names() if list(scat.glob("h.[0-9][0-9][0-9]")) else None

    def nm(i):
        return names[i - 1] if names and 0 < i <= len(names) else str(i)
    v, i, j = err["h"]
    check("target states do not interact: max |<i|H|j>|", v < tol_h,
          f"{v:.2e} a.u. ({v * 27.211386:.3f} eV) between {nm(i)} and {nm(j)}")
    v, i, j = err["s"]
    check("target states orthogonal: max |<i|j>|", v < tol_s, f"{v:.1e} ({nm(i)}, {nm(j)})")


# ---------------------------------------------------------------------------
# 7. library robustness
# ---------------------------------------------------------------------------
@guarded
def test_library(tmp: Path, repo: Path):
    section("7. Library robustness")
    bindir = tmp / "fakebin"
    bindir.mkdir()
    marker = f"pydbsr_stress_{os.getpid()}"
    (bindir / "dbsr_sleepy").write_text(f"#!/bin/sh\n# {marker}\nsleep 57 &\nwait\n")
    (bindir / "dbsr_stop").write_text("#!/bin/sh\necho ' STOP DPOTRF info > 0 - Cholesky factorization failed'\n")
    for p in bindir.iterdir():
        p.chmod(0o755)
    old = os.environ.get("DBSR_BIN")
    os.environ["DBSR_BIN"] = str(bindir)
    try:
        t = time.time()
        try:
            run("dbsr_sleepy", timeout=1.0)
            ok = False
        except subprocess.TimeoutExpired:
            ok = True
        time.sleep(0.3)
        left = subprocess.run(["pgrep", "-f", "sleep 57"], capture_output=True, text=True).stdout.split()
        check("timeout kills the program and its children", ok and not left and time.time() - t < 5,
              f"{time.time() - t:.1f} s, leftover processes: {left}")
        try:
            run("dbsr_stop")
            ok = False
        except DBSRError as e:
            ok = "Cholesky" in str(e)
        check("'STOP ...' from a DBSR program raises DBSRError", ok)
    finally:
        if old is None:
            os.environ.pop("DBSR_BIN", None)
        else:
            os.environ["DBSR_BIN"] = old
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-W", "error", "-m", "not slow", str(repo / "tests")],
                       capture_output=True, text=True, cwd=repo)
    last = (r.stdout.strip().splitlines() or ["?"])[-1]
    check("unit tests pass with warnings as errors", r.returncode == 0, last)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scat", action="append", default=[], help="scattering run directory (h.nnn)")
    ap.add_argument("--target", help="target directory (pydbsr_target.json)")
    ap.add_argument("--log", default="stress_log.json")
    ap.add_argument("--skip-matrices", action="store_true", help="skip the (slow) overlap-matrix eigenvalue test")
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    print(f"pydbsr {db.__version__}; DBSR programs: {Path(which('dbsr_hf')).parent}; python {sys.version.split()[0]}; "
          f"numpy {np.__version__}")
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        test_dirac_hydrogenic(tmp)
        if args.target:
            test_target(Path(args.target), tmp)
        test_coulomb()
        for s in args.scat:
            test_outer(Path(s))
            if not args.skip_matrices:
                test_matrices(Path(s))
            test_target_consistency(Path(s))
        test_library(tmp, repo)
    n_fail = sum(not r["pass"] for r in RESULTS)
    print(f"\n{len(RESULTS)} checks, {len(RESULTS) - n_fail} passed, {n_fail} failed ({time.time() - t0:.0f} s)")
    for r in RESULTS:
        if not r["pass"]:
            print(f"  FAILED: {r['test']}: {r['detail']}")
    Path(args.log).write_text(json.dumps(RESULTS, indent=1))
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
