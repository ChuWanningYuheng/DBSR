"""Coulomb wave functions F_l(eta, rho), G_l(eta, rho) and their rho-derivatives.

Uses A. R. Barnett's COULFG (Steed's method, from the ZCOM library) through
the compiled ``libpydbsr_coulomb`` when available, otherwise mpmath.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

_lib = None


def _load():
    global _lib
    if _lib is not None:
        return _lib or None
    here = Path(__file__).resolve().parent
    cands = []
    if os.environ.get("PYDBSR_COULOMB_LIB"):
        cands.append(Path(os.environ["PYDBSR_COULOMB_LIB"]))
    for d in (here / "lib", here):
        cands += sorted(d.glob("libpydbsr_coulomb.*")) + sorted(d.glob("pydbsr_coulomb*.dll"))
    for c in cands:
        try:
            lib = ctypes.CDLL(str(c))
            f = lib.pydbsr_coulfg
            dp = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
            ip = np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")
            f.argtypes = [ctypes.c_int, dp, dp, ip, dp, dp, dp, dp, ip]
            f.restype = None
            _lib = lib
            return lib
        except OSError:
            continue
    _lib = False
    return None


def backend() -> str:
    return "COULFG (Fortran)" if _load() else "mpmath"


def coulomb_fg(l, eta, rho):
    """Return F, G, dF/drho, dG/drho (arrays broadcast from the inputs)."""
    l, eta, rho = np.broadcast_arrays(np.asarray(l, dtype=np.int32),
                                      np.asarray(eta, dtype=float), np.asarray(rho, dtype=float))
    shape = rho.shape
    l = np.ascontiguousarray(l.ravel(), dtype=np.int32)
    eta = np.ascontiguousarray(eta.ravel())
    rho = np.ascontiguousarray(rho.ravel())
    n = rho.size
    F, G, Fp, Gp = (np.empty(n) for _ in range(4))
    lib = _load()
    bad = np.ones(n, dtype=bool)
    # COULFG switches to (inaccurate) JWKB inside the turning point, and its
    # continued fraction CF2 does not converge for |eta| > ~2e4 (k^2 < ~1e-9 a.u.,
    # right at a threshold; it then prints "CF2 HAS FAILED") -> mpmath there
    turning = eta + np.sqrt(eta ** 2 + l * (l + 1.0))
    ok = (rho > 1.02 * turning + 1e-3) & (np.abs(eta) < 1e4)
    if lib is not None and ok.any():
        idx = np.nonzero(ok)[0]
        m = idx.size
        f_, g_, fp_, gp_ = (np.empty(m) for _ in range(4))
        ifail = np.zeros(m, dtype=np.int32)
        lib.pydbsr_coulfg(m, np.ascontiguousarray(rho[idx]), np.ascontiguousarray(eta[idx]),
                          np.ascontiguousarray(l[idx]), f_, g_, fp_, gp_, ifail)
        F[idx], G[idx], Fp[idx], Gp[idx] = f_, g_, fp_, gp_
        wr = fp_ * g_ - f_ * gp_                 # Wronskian, = 1 exactly
        bad[idx] = (ifail != 0) | ~np.isfinite(wr) | (np.abs(wr - 1.0) > 1e-10)
    for i in np.nonzero(bad)[0]:
        F[i], G[i], Fp[i], Gp[i] = _mp_fg(int(l[i]), float(eta[i]), float(rho[i]))
    return F.reshape(shape), G.reshape(shape), Fp.reshape(shape), Gp.reshape(shape)


def _mp_fg(l, eta, rho):
    import mpmath as mp
    F = mp.coulombf(l, eta, rho)
    G = mp.coulombg(l, eta, rho)
    F1 = mp.coulombf(l + 1, eta, rho)
    G1 = mp.coulombg(l + 1, eta, rho)
    # DLMF 33.4.4:  u_l' = [(l+1)/rho + eta/(l+1)] u_l - sqrt((l+1)^2+eta^2)/(l+1) u_{l+1}
    a = (l + 1) / rho + eta / (l + 1)
    b = mp.sqrt((l + 1) ** 2 + eta ** 2) / (l + 1)
    return float(F), float(G), float(a * F - b * F1), float(a * G - b * G1)
