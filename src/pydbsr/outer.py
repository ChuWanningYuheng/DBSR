"""Outer region of the (D)BSR R-matrix method: K-matrices, collision strengths
and cross sections from the ``h.nnn`` files written by ``dbsr_hd3`` (itype=0).

Method (DBSR manual, Sect. 2.2, Eqs. 2.15-2.36):

* R-matrix at the boundary ``a``:
  ``R_ij(E) = 1/(2a) * sum_k w_ik w_jk / (E_k - E)`` (atomic units),
  relation ``P(a) = R (a P'(a) - b P(a))``  -> log-derivative
  ``Y(a) = (R^-1 + b) / a``.
* Non-relativistic outer region (Eq. 2.32), Rydberg-like form
  ``P'' = [l(l+1)/r^2 - 2z/r - k^2 + sum_lam C_lam r^-(lam+1)] P``
  with the long-range coefficients ``C`` of the h-file (= 2 x alpha),
  ``k_i^2 = 2 (E - E_i)``, ``z = Z - N``.
  Y is propagated from ``a`` to ``r_match`` with Johnson's log-derivative
  method (stable for closed channels, 4th order).
* Matching to energy-normalised Coulomb functions ``k^-1/2 (F + G K)`` in the
  open channels and to decaying functions in the closed ones gives K;
  ``S = (1 + iK)(1 - iK)^-1``, ``T = S - 1``.
* Collision strength ``Omega(i->j) = 1/2 sum_J (2J+1) sum |T_ba|^2`` and
  ``sigma(i->j) = pi a0^2 Omega / (g_i k_i^2)`` (Eq. 2.36).

The long-range multipole potentials beyond ``r_match`` and the contribution
of partial waves not included in the calculation (top-up) are neglected.
"""
from __future__ import annotations

import os
import struct
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from .constants import AU_EV, K_B_EV, PI_A0_2_CM2, RATE_UPS, V_EV_CM_S
from .coulomb import coulomb_fg

__all__ = ["HBlock", "HData", "read_h", "OuterRegion", "CollisionStrengths", "maxwell", "bugrova"]


# ---------------------------------------------------------------------------
# h.nnn reader (gfortran sequential unformatted, incl. >2 GB sub-records)
# ---------------------------------------------------------------------------
class _FortranReader:
    def __init__(self, path):
        self.f = open(path, "rb")

    def record(self) -> bytes:
        chunks = []
        while True:
            head = self.f.read(4)
            if len(head) < 4:
                raise EOFError
            n = struct.unpack("<i", head)[0]
            chunks.append(self.f.read(abs(n)))
            self.f.read(4)
            if n >= 0:
                return b"".join(chunks)

    def ints(self):
        return np.frombuffer(self.record(), dtype="<i4")

    def reals(self):
        return np.frombuffer(self.record(), dtype="<f8")

    def close(self):
        self.f.close()


@dataclass
class HBlock:
    """One partial wave (2J, parity) of an h-file."""
    two_j: int
    parity: int                 # +1 / -1
    nch: int
    l: np.ndarray               # orbital l of the continuum electron in each channel
    kappa: np.ndarray
    target: np.ndarray          # 0-based target index of each channel
    cf: np.ndarray              # (nch, nch, lamax) long-range coefficients, lam = 1..lamax
    poles: np.ndarray           # R-matrix poles (a.u.)
    w: np.ndarray               # (nch, npoles) surface amplitudes


@dataclass
class HData:
    nelc: int
    nz: int
    ntarg: int
    ra: float                   # R-matrix radius a
    rb: float                   # b in the boundary condition
    etarg: np.ndarray           # target energies (a.u.), as used for the thresholds
    two_j_targ: np.ndarray
    parity_targ: np.ndarray | None
    blocks: list[HBlock] = field(default_factory=list)

    @property
    def z(self) -> int:
        """Residual charge seen by the scattered electron."""
        return self.nz - self.nelc


def read_h(paths: str | Path | Sequence[str | Path]) -> HData:
    """Read one or several ``h.nnn`` files (or a merged ``H.DAT``)."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    data = None
    for p in paths:
        fr = _FortranReader(p)
        try:
            head = fr.record()
            nelc, nz, lrang2, lamax, ntarg = struct.unpack("<5i", head[:20])
            ra, rb = struct.unpack("<2d", head[20:36])
            etarg = fr.reals().copy()
            jt = fr.ints().copy()
            isat = fr.ints()
            ptarg = isat[ntarg:].copy() if isat.size == 2 * ntarg else None
            fr.record()                                        # Buttle coefficients (unused)
            if data is None:
                data = HData(nelc, nz, ntarg, ra, rb, etarg, jt, ptarg)
            elif ntarg != data.ntarg or abs(ra - data.ra) > 1e-10 or np.any(np.abs(etarg - data.etarg) > 1e-10):
                raise ValueError(f"{p}: target data differ from the other h-files")
            while True:
                try:
                    blk = fr.ints()
                except EOFError:
                    break
                lrgl, nspn, npty, nch, mnp2, more = blk[:6]
                nconat = fr.ints()
                lk = fr.ints()
                l, kappa = lk[:nch].copy(), lk[nch:].copy()
                cf = fr.reals().reshape((lamax, nch, nch)).transpose(2, 1, 0).copy() if lamax > 0 \
                    else np.zeros((nch, nch, 0))
                poles = fr.reals().copy()
                w = fr.reals().reshape((mnp2, nch)).T.copy()
                target = np.repeat(np.arange(ntarg), nconat)
                if target.size != nch:
                    raise ValueError(f"{p}: channel/target bookkeeping mismatch")
                data.blocks.append(HBlock(int(lrgl), 1 if npty == 0 else -1, int(nch), l, kappa,
                                          target, cf, poles, w))
                if more == 0:
                    break
        finally:
            fr.close()
    if data is None:
        raise ValueError("no h-files given")
    return data


# ---------------------------------------------------------------------------
# log-derivative propagation (Johnson 1973)
# ---------------------------------------------------------------------------
def _W(r, lfac, k2, z, cf, lam):
    """Coupling matrix W(r) in P'' = W P."""
    W = np.tensordot(cf, r ** -(lam + 1.0), axes=([2], [0])) if cf.shape[2] else np.zeros(cf.shape[:2])
    W[np.diag_indices_from(W)] += lfac / r ** 2 - 2.0 * z / r - k2
    return W


def propagate_logderiv(Y, r0, r1, lfac, k2, z, cf, nsteps):
    """Propagate the log-derivative matrix Y = P' P^-1 from r0 to r1."""
    if r1 <= r0:
        return Y
    nsteps += nsteps % 2
    h = (r1 - r0) / nsteps
    n = Y.shape[0]
    I = np.eye(n)
    lam = np.arange(1, cf.shape[2] + 1, dtype=float)
    Y = Y + (h / 3.0) * _W(r0, lfac, k2, z, cf, lam)
    for m in range(1, nsteps + 1):
        W = _W(r0 + m * h, lfac, k2, z, cf, lam)
        if m == nsteps:
            wU = W
        elif m % 2:
            wU = 4.0 * np.linalg.solve(I - (h * h / 6.0) * W, W)
        else:
            wU = 2.0 * W
        Y = np.linalg.solve(I + h * Y, Y) + (h / 3.0) * wU
    return Y


# ---------------------------------------------------------------------------
# K-matrix at one energy for one partial wave
# ---------------------------------------------------------------------------
def kmatrix(blk: HBlock, hd: HData, etot: float, r_match: float | None = None,
            step: float | None = None, bsto: float | None = None, symmetrize: bool = True):
    """K-matrix (open x open) and the list of open channels at total energy ``etot`` (a.u.).

    The exact K is real symmetric (flux conservation, S unitary).  The computed
    one is symmetric to the accuracy of the R-matrix, the propagation and the
    Coulomb functions; it is symmetrised unless ``symmetrize=False`` (use that
    to measure the asymmetry, see ``tools/stress_test.py``).
    """
    a = hd.ra
    b = hd.rb if bsto is None else bsto
    e_ch = hd.etarg[blk.target]
    k2 = 2.0 * (etot - e_ch)
    # exactly at a threshold (|k^2| < K2_THRESHOLD) the channel is taken as just
    # open: the energy-normalised Coulomb functions have a finite limit for
    # k -> 0+ (attractive field), the closed-channel solution has none (kappa = 0)
    k2 = np.where(np.abs(k2) < K2_THRESHOLD, K2_THRESHOLD, k2)
    open_ = np.nonzero(k2 > 0)[0]
    if open_.size == 0:
        return np.zeros((0, 0)), open_
    # R-matrix and log-derivative at r = a
    R = (blk.w / (blk.poles - etot)) @ blk.w.T / (2.0 * a)
    Y = (np.linalg.inv(R) + b * np.eye(blk.nch)) / a
    # propagation to r_match
    lfac = blk.l * (blk.l + 1.0)
    rm = a if r_match is None else max(r_match, a)
    if rm > a:
        kmax = np.sqrt(np.max(np.abs(k2)) + 1e-12)
        h = step if step is not None else min(0.05, 0.1 / max(kmax, 1e-3))
        Y = propagate_logderiv(Y, a, rm, lfac, k2, hd.z, blk.cf, int(np.ceil((rm - a) / h)))
    # asymptotic functions at r_match
    n, no = blk.nch, open_.size
    closed = np.setdiff1d(np.arange(n), open_)
    k = np.sqrt(k2[open_])
    eta = -hd.z / k
    F, G, Fp, Gp = coulomb_fg(blk.l[open_], eta, k * rm)
    sk = np.sqrt(k)
    f, g = F / sk, G / sk
    fp, gp = Fp * sk, Gp * sk               # d/dr = k d/drho
    Fm = np.zeros((n, no)); Fpm = np.zeros((n, no))
    Fm[open_, np.arange(no)] = f
    Fpm[open_, np.arange(no)] = fp
    Gm = np.zeros((n, n)); Gpm = np.zeros((n, n))
    Gm[open_, open_] = g
    Gpm[open_, open_] = gp
    if closed.size:
        Gm[closed, closed] = 1.0
        Gpm[closed, closed] = closed_logderiv(blk.l[closed], -k2[closed], hd.z, rm)
    X = np.linalg.solve(Gpm - Y @ Gm, -(Fpm - Y @ Fm))
    K = X[open_, :]
    return (0.5 * (K + K.T) if symmetrize else K), open_


K2_THRESHOLD = 1e-10        # a.u. of k^2 (1.4e-9 eV): below this a channel counts as at threshold


def closed_logderiv(l, kappa2, z, r):
    """Log-derivative at r of the exponentially decaying solution in closed channels.

    Four-term WKB series of the Riccati equation y' + y^2 = Q,
    Q = kappa^2 + l(l+1)/r^2 - 2z/r, where its last term is below ``_WKB_TOL``
    relative; otherwise the Whittaker function W_{z/kappa, l+1/2}(2 kappa r)
    (mpmath; cached, since the same (l, kappa^2) recurs in many partial waves
    at one energy).
    """
    l = np.atleast_1d(l).astype(float)
    kappa2 = np.atleast_1d(kappa2).astype(float)
    if np.any(kappa2 <= 0):
        raise ValueError("closed_logderiv: kappa^2 must be > 0 (channel below its threshold)")
    y, ok = _wkb_series(l * (l + 1.0), kappa2, z, r)
    out = np.where(ok, y, 0.0)
    for i in np.nonzero(~ok)[0]:
        out[i] = _whittaker_cached(float(l[i]), float(kappa2[i]), float(z), float(r))
    return out


_WKB_TOL = 1e-8                 # relative size of the last WKB term (error ~2e-8)


def _wkb_series(L, kappa2, z, r):
    """y = y0 + y1 + y2 + y3 for the decaying solution and a mask where it is accurate."""
    Q = kappa2 + L / r ** 2 - 2.0 * z / r
    Q1 = -2.0 * L / r ** 3 + 2.0 * z / r ** 2
    Q2 = 6.0 * L / r ** 4 - 4.0 * z / r ** 3
    Q3 = -24.0 * L / r ** 5 + 12.0 * z / r ** 4
    pos = Q > 0
    Qs = np.where(pos, Q, 1.0)
    q = np.sqrt(Qs)
    y0 = -q
    y1 = -Q1 / (4.0 * Qs)
    N = 5.0 * Q1 ** 2 - 4.0 * Qs * Q2
    D = 32.0 * Qs ** 2.5
    y2 = N / D
    dN = 6.0 * Q1 * Q2 - 4.0 * Qs * Q3
    dD = 80.0 * Qs ** 1.5 * Q1
    y3 = (dN / D - N * dD / D ** 2 + 2.0 * y1 * y2) / (2.0 * q)
    y = y0 + y1 + y2 + y3
    ok = pos & (np.abs(y3) < _WKB_TOL * np.abs(y)) & (np.abs(y2) < 1e-3 * np.abs(y))
    return y, ok


_WCACHE: dict = {}


def _whittaker_cached(l, kappa2, z, r):
    key = (l, kappa2, z, r)
    v = _WCACHE.get(key)
    if v is None:
        if len(_WCACHE) > 100000:
            _WCACHE.clear()
        v = _WCACHE[key] = _whittaker_logderiv(l, kappa2, z, r)
    return v


def _whittaker_logderiv(l, kappa2, z, r):
    import mpmath as mp
    kappa = mp.sqrt(kappa2)
    k, m = z / kappa, l + 0.5
    x = 2 * kappa * r
    with mp.workdps(30):
        w = mp.whitw(k, m, x)
        w1 = mp.whitw(k + 1, m, x)
        dw = (0.5 - k / x) * w - w1 / x          # DLMF 13.15.23:  x W' = (x/2 - k) W - W_{k+1}
    return float(2 * kappa * dw / w)


def t_matrix(K):
    """T = S - 1 = 2iK (1 - iK)^-1."""
    n = K.shape[0]
    return 2j * np.linalg.solve((np.eye(n) - 1j * K).T, K.T).T


# ---------------------------------------------------------------------------
# collision strengths on an energy grid
# ---------------------------------------------------------------------------
_G = {}


def _omega_one(args):
    ie, etot = args
    hd, r_match, step = _G["hd"], _G["r_match"], _G["step"]
    nt = hd.ntarg
    om = np.zeros((nt, nt))
    om_pw = np.zeros((len(hd.blocks), nt, nt)) if _G["per_pw"] else None
    for ib, blk in enumerate(hd.blocks):
        K, op = kmatrix(blk, hd, etot, r_match, step)
        if op.size == 0:
            continue
        T2 = np.abs(t_matrix(K)) ** 2
        t = blk.target[op]
        contrib = np.zeros((nt, nt))
        np.add.at(contrib, (t[None, :].repeat(len(t), 0), t[:, None].repeat(len(t), 1)), T2)
        contrib *= 0.5 * (blk.two_j + 1)
        om += contrib
        if om_pw is not None:
            om_pw[ib] = contrib
    return ie, om, om_pw


@dataclass
class CollisionStrengths:
    """Omega[i_energy, i, j] for the transition i -> j (initial i, final j)."""
    energies: np.ndarray        # electron energy relative to the ground target state, eV
    thresholds: np.ndarray      # target excitation energies, eV
    two_j: np.ndarray           # 2J of target states
    omega: np.ndarray           # (ne, nt, nt), symmetric
    omega_pw: np.ndarray | None = None   # (ne, nlsp, nt, nt)
    pw: list | None = None               # [(2J, parity)] of the partial waves
    names: list | None = None            # target-state names (h-file order)

    def index(self, state) -> int:
        """Index of a target state given by name, State object or index."""
        if isinstance(state, (int, np.integer)):
            return int(state)
        name = getattr(state, "name", state)
        if self.names is None or name not in self.names:
            raise KeyError(f"unknown target state {name!r}")
        return self.names.index(name)

    @property
    def g(self):
        return self.two_j + 1

    def omega_ij(self, i, j) -> np.ndarray:
        return self.omega[:, self.index(i), self.index(j)]

    def sigma(self, i, j, units: str = "cm2") -> np.ndarray:
        """Cross section i -> j vs incident energy (NaN below the threshold of i or j).

        ``i``, ``j``: indices, names or State objects."""
        i, j = self.index(i), self.index(j)
        e_inc = self.energies - self.thresholds[i]          # incident energy on state i, eV
        k2 = e_inc / (AU_EV / 2)                             # k^2 in Ry = (k a0)^2
        with np.errstate(divide="ignore", invalid="ignore"):
            s = self.omega[:, i, j] / (self.g[i] * k2)
        s = np.where((e_inc > 0) & (self.energies > self.thresholds[j]), s, np.nan)
        if units == "cm2":
            return s * PI_A0_2_CM2
        if units in ("pia02", "pi_a0^2"):
            return s
        if units == "a02":
            return s * np.pi
        raise ValueError(units)

    def incident_energy(self, i) -> np.ndarray:
        return self.energies - self.thresholds[self.index(i)]

    def upsilon(self, i, j, T: float | np.ndarray) -> np.ndarray:
        """Maxwellian-averaged effective collision strength (T in K).

        Integrates Omega over the final-electron energy on the computed grid:
        the grid must extend to several kT above the upper threshold.
        """
        i, j = self.index(i), self.index(j)
        lo, up = (i, j) if self.thresholds[i] <= self.thresholds[j] else (j, i)
        e_f = self.energies - self.thresholds[up]
        m = e_f >= 0
        x_e, om = e_f[m], self.omega[m, i, j]
        T = np.atleast_1d(np.asarray(T, dtype=float))
        out = []
        for t in T:
            kt = K_B_EV * t
            x = x_e / kt
            out.append(np.trapezoid(om * np.exp(-x), x) if hasattr(np, "trapezoid") else np.trapz(om * np.exp(-x), x))
        return np.array(out)

    def rate(self, i, j, T: float | np.ndarray) -> np.ndarray:
        """Rate coefficient i -> j in cm^3/s (excitation or de-excitation)."""
        i, j = self.index(i), self.index(j)
        T = np.atleast_1d(np.asarray(T, dtype=float))
        ups = self.upsilon(i, j, T)
        de = self.thresholds[j] - self.thresholds[i]
        boltz = np.exp(-np.maximum(de, 0) / (K_B_EV * T))
        return RATE_UPS / (self.g[i] * np.sqrt(T)) * ups * boltz

    def rate_eedf(self, i, j, eedf) -> float:
        """Rate coefficient (cm^3/s) for i -> j with an electron energy distribution:
        ``Q = integral sigma(E) sqrt(2E/m) g(E) dE`` (Zhu et al 2019, Eq. 18).

        ``eedf``: callable g(E [eV]) normalised to 1, e.g. :func:`maxwell` or
        :func:`bugrova`.  Integrated on the computed energy grid, so the grid
        must cover the part of the distribution above the threshold.
        """
        e = self.incident_energy(i)
        sig = np.nan_to_num(self.sigma(i, j), nan=0.0)
        m = e > 0
        v = V_EV_CM_S * np.sqrt(e[m])                        # electron speed, cm/s
        f = sig[m] * v * eedf(e[m])
        return float(np.trapezoid(f, e[m]) if hasattr(np, "trapezoid") else np.trapz(f, e[m]))

    def save(self, path):
        extra = {} if self.omega_pw is None else {"omega_pw": self.omega_pw}
        if self.names is not None:
            extra["names"] = np.array(self.names)
        if self.pw is not None:
            extra["pw"] = np.array(self.pw)
        np.savez_compressed(path, energies=self.energies, thresholds=self.thresholds, two_j=self.two_j,
                            omega=self.omega, **extra)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        get = lambda k: d[k] if k in d.files else None
        names = get("names")
        pw = get("pw")
        return cls(d["energies"], d["thresholds"], d["two_j"], d["omega"], get("omega_pw"),
                   None if pw is None else [tuple(x) for x in pw.tolist()],
                   None if names is None else names.tolist())


def maxwell(Te_eV: float):
    """Maxwellian electron energy distribution g(E), E and Te in eV (normalised to 1)."""
    def g(E):
        E = np.asarray(E, float)
        return 2.0 * np.sqrt(E / np.pi) * Te_eV ** -1.5 * np.exp(-E / Te_eV)
    return g


def bugrova(Te_eV: float):
    """Bugrova distribution (Hall-thruster channel; Zhu et al 2019, Eq. 19):
    g(E) = 15/4 u^-5/2 E^1/2 (u - E) for E <= u, mean energy 3u/7 = 3Te/2."""
    u = 3.5 * Te_eV

    def g(E):
        E = np.asarray(E, float)
        return np.where(E <= u, 3.75 * u ** -2.5 * np.sqrt(np.clip(E, 0, None)) * (u - E), 0.0)
    return g


class OuterRegion:
    """Collision strengths from the R-matrix data.

    >>> outer = OuterRegion.from_files(sorted(Path('run').glob('h.[0-9][0-9][0-9]')))
    >>> cs = outer.collision_strengths(np.linspace(0.1, 30, 2000), jobs=32)
    >>> cs.sigma(0, 3)
    """

    def __init__(self, hdata: HData, r_match: float | None = None, step: float | None = None,
                 names: Sequence[str] | None = None):
        self.h = hdata
        self.r_match = r_match
        self.step = step
        self.names = list(names) if names is not None else None
        if self.names is not None and len(self.names) != hdata.ntarg:
            raise ValueError(f"{len(self.names)} names for {hdata.ntarg} target states")

    @classmethod
    def from_files(cls, paths, **kw):
        return cls(read_h(paths), **kw)

    @property
    def thresholds_ev(self) -> np.ndarray:
        return (self.h.etarg - self.h.etarg.min()) * AU_EV

    def kmatrix(self, block: int, energy_ev: float):
        """K-matrix of partial wave ``block`` (0-based) at electron energy (eV above ground)."""
        return kmatrix(self.h.blocks[block], self.h, self.h.etarg.min() + energy_ev / AU_EV,
                       self.r_match, self.step)

    def collision_strengths(self, energies_ev, jobs: int | None = None,
                            per_partial_wave: bool = False) -> CollisionStrengths:
        """Omega(i->j) at electron energies (eV, relative to the lowest target state)."""
        energies = np.asarray(energies_ev, dtype=float)
        e0 = self.h.etarg.min()
        tasks = [(i, e0 + e / AU_EV) for i, e in enumerate(energies)]
        nt, nb = self.h.ntarg, len(self.h.blocks)
        omega = np.zeros((len(energies), nt, nt))
        omega_pw = np.zeros((len(energies), nb, nt, nt)) if per_partial_wave else None
        _G.update(hd=self.h, r_match=self.r_match, step=self.step, per_pw=per_partial_wave)
        jobs = jobs or 1
        if jobs > 1:
            import multiprocessing as mp
            ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else None
            with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx,
                                     initializer=_init_worker,
                                     initargs=(self.h, self.r_match, self.step, per_partial_wave)) as pool:
                results = pool.map(_omega_one, tasks, chunksize=max(1, len(tasks) // (8 * jobs)))
                for ie, om, opw in results:
                    omega[ie] = om
                    if omega_pw is not None:
                        omega_pw[ie] = opw
        else:
            for t in tasks:
                ie, om, opw = _omega_one(t)
                omega[ie] = om
                if omega_pw is not None:
                    omega_pw[ie] = opw
        pw = [(b.two_j, b.parity) for b in self.h.blocks]
        return CollisionStrengths(energies, self.thresholds_ev, self.h.two_j_targ.copy(), omega, omega_pw, pw,
                                  self.names)


def _init_worker(hd, r_match, step, per_pw):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _G.update(hd=hd, r_match=r_match, step=step, per_pw=per_pw)
