"""Atomic data for a collisional-radiative model (CRM) and a steady-state solver.

The data of one species are kept in :class:`CRMData`:

* levels (name, g, energy in cm-1, label),
* radiative rates A[upper, lower] (s-1),
* electron-impact excitation rate coefficients k[i -> j](Te) (cm3/s) on a Te
  grid, i below j; de-excitation follows from detailed balance,
* optional ionization rate coefficients S[i](Te) and an escape factor for
  every line (radiation trapping).

:func:`steady_state` solves the rate equations for the excited levels with
the ground-state density fixed (the usual quasi-steady-state CRM):

    0 = sum_j n_j (ne k_ji + A_ji) - n_i (sum_j ne k_ij + sum_j A_ij + ne S_i + 1/tau_i)

where ``tau_i`` is a transport (flight) time that matters for metastable
levels.  Line emission per ground-state particle is n_u A_ul.

The cross sections come from :class:`pydbsr.CollisionStrengths` (own DBSR
runs) or from tables (Wang et al 2019, Fursa et al 1999):
:func:`rate_from_sigma` integrates sigma(E) over a Maxwellian or any EEDF.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

__all__ = ["rate_from_sigma", "deexcitation", "CRMData", "steady_state", "line_emission",
           "sensitivity", "escape_factor_doppler", "doppler_k0"]

CM_PER_EV = 8065.544005
KB_EV = 8.617333262e-5
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


@dataclass
class CRMData:
    """Atomic data of one species (see the module docstring)."""
    species: str
    names: list
    g: np.ndarray
    energy_cm: np.ndarray
    labels: list = field(default_factory=list)
    Te: np.ndarray = field(default_factory=lambda: np.array([]))
    A: dict = field(default_factory=dict)          # (upper, lower) -> s-1
    k: dict = field(default_factory=dict)          # (i, j), E_i < E_j -> array over Te (cm3/s)
    S: dict = field(default_factory=dict)          # i -> array over Te (ionization, cm3/s)
    escape: dict = field(default_factory=dict)     # (upper, lower) -> escape factor (1 = optically thin)
    source: dict = field(default_factory=dict)     # (i, j) -> 'Wang2019' / 'pydbsr' / ...

    def __post_init__(self):
        self.g = np.asarray(self.g, float)
        self.energy_cm = np.asarray(self.energy_cm, float)
        self.Te = np.asarray(self.Te, float)
        self.index = {n: i for i, n in enumerate(self.names)}

    @property
    def n(self) -> int:
        return len(self.names)

    def add_rate(self, i, j, k_of_Te, source: str = ""):
        """Excitation rate i -> j (names or indices; swapped if i is above j)."""
        a, b = self._i(i), self._i(j)
        if self.energy_cm[a] > self.energy_cm[b]:
            raise ValueError(f"{self.names[a]} lies above {self.names[b]}: give the excitation")
        self.k[(a, b)] = np.asarray(k_of_Te, float)
        if source:
            self.source[(a, b)] = source

    def add_A(self, upper, lower, A: float):
        self.A[(self._i(upper), self._i(lower))] = float(A)

    def _i(self, x) -> int:
        return x if isinstance(x, (int, np.integer)) else self.index[x]

    def rate_matrix(self, ne: float, Te: float, tau: float | dict | None = None) -> np.ndarray:
        """M[i, j] = rate (s-1) from level j to level i; diagonal = -total loss."""
        n = self.n
        M = np.zeros((n, n))
        de = self.energy_cm / CM_PER_EV
        for (a, b), kk in self.k.items():
            kab = float(np.exp(np.interp(np.log(Te), np.log(self.Te), np.log(np.maximum(kk, 1e-300)))))
            kba = float(deexcitation(kab, self.g[a], self.g[b], de[b] - de[a], Te))
            M[b, a] += ne * kab
            M[a, b] += ne * kba
        for (u, l), a in self.A.items():
            M[l, u] += a * self.escape.get((u, l), 1.0)
        loss = M.sum(axis=0)                 # total rate out of each level (column sums)
        for i, ss in self.S.items():
            loss[self._i(i)] += ne * float(np.exp(np.interp(np.log(Te), np.log(self.Te),
                                                             np.log(np.maximum(ss, 1e-300)))))
        if tau is not None:
            for i in range(n):
                t = tau.get(self.names[i]) if isinstance(tau, dict) else tau
                if t:
                    loss[i] += 1.0 / t
        M[np.diag_indices(n)] = -loss
        return M

    # ------------------------------------------------------------- persistence
    def save(self, path):
        d = dict(species=self.species, names=self.names, g=self.g.tolist(), energy_cm=self.energy_cm.tolist(),
                 labels=self.labels, Te=self.Te.tolist(),
                 A=[[u, l, a] for (u, l), a in self.A.items()],
                 k=[[a, b, v.tolist(), self.source.get((a, b), "")] for (a, b), v in self.k.items()],
                 S=[[i, v.tolist()] for i, v in self.S.items()],
                 escape=[[u, l, e] for (u, l), e in self.escape.items()])
        Path(path).write_text(json.dumps(d))
        return Path(path)

    @classmethod
    def load(cls, path) -> "CRMData":
        d = json.loads(Path(path).read_text())
        out = cls(d["species"], d["names"], d["g"], d["energy_cm"], d.get("labels", []), d["Te"])
        out.A = {(u, l): a for u, l, a in d["A"]}
        out.k = {(a, b): np.array(v) for a, b, v, _ in d["k"]}
        out.source = {(a, b): s for a, b, _, s in d["k"] if s}
        out.S = {i: np.array(v) for i, v in d.get("S", [])}
        out.escape = {(u, l): e for u, l, e in d.get("escape", [])}
        return out


def steady_state(data: CRMData, ne: float, Te: float, ground=0, tau: float | dict | None = None,
                 fixed: Sequence | None = None) -> np.ndarray:
    """Populations relative to the ground level (n_ground = 1).

    ``fixed``: further levels whose populations are prescribed elsewhere
    (dict name -> population) — e.g. metastables from a transport model.
    Otherwise every level except ``ground`` is in steady state, including
    metastables (then ``tau`` should be given: without a loss channel a
    metastable level just piles up to a Boltzmann population).
    """
    M = data.rate_matrix(ne, Te, tau)
    g0 = data._i(ground)
    fix = {g0: 1.0}
    if fixed:
        fix.update({data._i(k): v for k, v in dict(fixed).items()})
    free = [i for i in range(data.n) if i not in fix]
    rhs = -sum(M[np.ix_(free, [i])][:, 0] * v for i, v in fix.items())
    x = np.linalg.solve(M[np.ix_(free, free)], rhs)
    pop = np.zeros(data.n)
    for i, v in fix.items():
        pop[i] = v
    pop[free] = x
    return pop


def line_emission(data: CRMData, pop: np.ndarray, upper, lower) -> float:
    """Photon emission coefficient per ground-state particle, n_u A_ul (escape-corrected), s-1."""
    u, l = data._i(upper), data._i(lower)
    return pop[u] * data.A.get((u, l), 0.0) * data.escape.get((u, l), 1.0)


def sensitivity(data: CRMData, lines: Sequence[tuple], ne: float, Te: float, tau=None,
                rel: float = 0.05, **kw) -> dict:
    """d ln I / d ln ne and d ln I / d ln Te for lines [(upper, lower), ...]."""
    def I(ne_, Te_):
        p = steady_state(data, ne_, Te_, tau=tau, **kw)
        return np.array([line_emission(data, p, u, l) for u, l in lines])
    i0 = I(ne, Te)
    dne = (np.log(I(ne * (1 + rel), Te)) - np.log(I(ne * (1 - rel), Te))) / (np.log(1 + rel) - np.log(1 - rel))
    dTe = (np.log(I(ne, Te * (1 + rel))) - np.log(I(ne, Te * (1 - rel)))) / (np.log(1 + rel) - np.log(1 - rel))
    return {tuple(l): dict(I=a, dlnI_dlnne=b, dlnI_dlnTe=c) for l, a, b, c in zip(lines, i0, dne, dTe)}


def doppler_k0(wl_nm: float, g_lower: float, g_upper: float, A: float, T_K: float, mass_amu: float) -> float:
    """Line-centre absorption cross section per absorber, k0/n (cm2), Doppler profile."""
    v0 = np.sqrt(2 * 1.380649e-23 * T_K / (mass_amu * 1.66053907e-27))       # m/s
    lam = wl_nm * 1e-9
    return lam ** 3 / (8 * np.pi ** 1.5) * g_upper / g_lower * A / v0 * 1e4


def escape_factor_doppler(tau0: float) -> float:
    """Holstein-type escape factor for a Doppler line of optical depth tau0
    (Mewe approximation, accurate to ~10 %): 1 for tau0 -> 0."""
    t = max(float(tau0), 0.0)
    return (2.0 - np.exp(-t / 1000.0)) / (1.0 + t)
