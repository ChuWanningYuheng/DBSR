"""Rate coefficients <sigma v>(Te) from electron-impact cross sections.

Input data for a collisional-radiative model (the model itself is not part of
pydbsr): Maxwellian (or any EEDF) rate coefficients from a tabulated cross
section — Wang et al (2019), Fursa et al (1999) or a :class:`CollisionStrengths`
of an own DBSR run — and the de-excitation rate by detailed balance.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

__all__ = ["rate_from_sigma", "rates_on_grid", "deexcitation", "V_EV", "CM_PER_EV"]

CM_PER_EV = 8065.544005
V_EV = 5.930969e7            # electron speed (cm/s) = V_EV * sqrt(E/eV)


def rate_from_sigma(E_eV, sigma_cm2, Te_eV: float | None = None, threshold_eV: float | None = None,
                    eedf: Callable | None = None, ion: bool = True, tail: str = "1/E",
                    emax_Te: float = 60.0, n: int = 6000) -> float:
    """<sigma v> (cm3/s) from a tabulated cross section.

    ``E_eV`` incident energy above the initial level, ``sigma_cm2``.
    ``eedf``: g(E) normalised to 1 (e.g. :func:`pydbsr.outer.maxwell`), or
    a Maxwellian at ``Te_eV``.  Between the threshold and the first point the
    cross section is held at its first value for ions (finite at threshold)
    and interpolated linearly to zero for neutral targets; above the last
    point it falls as 1/E (``tail='1/E'``, forbidden-like, conservative) or
    as ln(E)/E (``tail='dipole'``).
    """
    E = np.asarray(E_eV, float)
    s = np.asarray(sigma_cm2, float)
    o = np.argsort(E)
    E, s = E[o], s[o]
    thr = float(E[0]) if threshold_eV is None else float(threshold_eV)
    if eedf is None:
        if Te_eV is None:
            raise ValueError("give Te_eV or eedf")
        Te = float(Te_eV)
        eedf = lambda x: 2.0 * np.sqrt(x / np.pi) * Te ** -1.5 * np.exp(-x / Te)
        top = thr + emax_Te * Te
    else:
        top = max(E[-1], thr + emax_Te * (Te_eV or 5.0))
    Eg = np.linspace(thr, top, n)
    sg = np.interp(Eg, E, s)
    lo = Eg < E[0]
    if lo.any():
        if ion:
            sg[lo] = s[0]
        else:
            sg[lo] = s[0] * (Eg[lo] - thr) / max(E[0] - thr, 1e-12)
    hi = Eg > E[-1]
    if hi.any():
        if tail == "dipole":
            sg[hi] = s[-1] * (E[-1] / Eg[hi]) * np.log(Eg[hi] / thr + 1) / np.log(E[-1] / thr + 1)
        else:
            sg[hi] = s[-1] * E[-1] / Eg[hi]
    return float(np.trapezoid(sg * V_EV * np.sqrt(Eg) * eedf(Eg), Eg))


def deexcitation(k_up, g_lower: float, g_upper: float, de_eV: float, Te_eV):
    """Detailed balance (Maxwellian): k(j->i) = k(i->j) g_i/g_j exp(dE/Te)."""
    return np.asarray(k_up) * g_lower / g_upper * np.exp(de_eV / np.asarray(Te_eV))


def rates_on_grid(E_eV, sigma_cm2, Te_grid, threshold_eV: float | None = None, **kw) -> np.ndarray:
    """Maxwellian rate coefficients (cm3/s) on a grid of Te (eV)."""
    return np.array([rate_from_sigma(E_eV, sigma_cm2, float(t), threshold_eV, **kw) for t in Te_grid])
