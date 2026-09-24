import numpy as np
import pytest
from scipy.integrate import solve_ivp

from pydbsr.coulomb import coulomb_fg
from pydbsr.outer import HBlock, HData, kmatrix, propagate_logderiv


def _hdata(z, a, etarg, blk):
    hd = HData(nelc=10, nz=10 + z, ntarg=len(etarg), ra=a, rb=0.0, etarg=np.asarray(etarg, float),
               two_j_targ=np.zeros(len(etarg), int), parity_targ=None)
    hd.blocks.append(blk)
    return hd


def _block_from_Y(Y, a, etot, l, target, cf=None):
    """Synthetic R-matrix data reproducing log-derivative Y(a) at energy etot.

    R = (a Y)^-1 = 1/(2a) sum_k w_k w_k^T / (E_k - E) with one pole per eigenvector.
    """
    R = np.linalg.inv(a * Y)
    R = 0.5 * (R + R.T)
    lam, U = np.linalg.eigh(R)
    # choose poles E_k = E + 1/(2a lam_k) and amplitudes w = eigenvectors
    poles = etot + 1.0 / (2 * a * lam)
    n = len(l)
    cf = np.zeros((n, n, 2)) if cf is None else cf
    return HBlock(two_j=0, parity=1, nch=n, l=np.asarray(l), kappa=-np.asarray(l) - 1,
                  target=np.asarray(target), cf=cf, poles=poles, w=U)


@pytest.mark.parametrize("z,l,delta,k", [(1, 0, 0.3, 0.7), (1, 2, -0.8, 1.3), (0, 1, 0.5, 0.9), (3, 4, 1.1, 0.4)])
@pytest.mark.parametrize("r_match_extra", [0.0, 37.0])
def test_single_channel_coulomb_phase(z, l, delta, k, r_match_extra):
    a = 20.0
    etot = 0.5 * k * k                 # single target at E = 0
    eta = -z / k
    F, G, Fp, Gp = coulomb_fg(l, eta, k * a)
    u, up = F * np.cos(delta) + G * np.sin(delta), k * (Fp * np.cos(delta) + Gp * np.sin(delta))
    blk = _block_from_Y(np.array([[up / u]]), a, etot, [l], [0])
    hd = _hdata(z, a, [0.0], blk)
    K, op = kmatrix(blk, hd, etot, r_match=a + r_match_extra, step=0.01, relativistic=False)
    assert op.tolist() == [0]
    assert K[0, 0] == pytest.approx(np.tan(delta), rel=1e-7, abs=1e-9)


def test_propagation_against_ode():
    """Coupled open channels with dipole/quadrupole long-range coupling."""
    rng = np.random.default_rng(1)
    n, z = 3, 1
    l = np.array([0, 1, 2])
    k2 = np.array([1.2, 0.9, 0.5])
    lfac = l * (l + 1.0)
    cf = np.zeros((n, n, 2))
    c = rng.normal(size=(n, n)); cf[:, :, 0] = 0.8 * (c + c.T)
    c = rng.normal(size=(n, n)); cf[:, :, 1] = 1.5 * (c + c.T)
    r0, r1 = 15.0, 40.0
    Y0 = rng.normal(size=(n, n)); Y0 = Y0 + Y0.T

    def W(r):
        M = cf[:, :, 0] / r ** 2 + cf[:, :, 1] / r ** 3
        return M + np.diag(lfac / r ** 2 - 2 * z / r - k2)

    def rhs(r, y):
        P = y[:n * n].reshape(n, n); dP = y[n * n:].reshape(n, n)
        return np.concatenate([dP.ravel(), (W(r) @ P).ravel()])

    y0 = np.concatenate([np.eye(n).ravel(), Y0.ravel()])
    sol = solve_ivp(rhs, [r0, r1], y0, rtol=1e-11, atol=1e-12, method="DOP853")
    P = sol.y[:n * n, -1].reshape(n, n); dP = sol.y[n * n:, -1].reshape(n, n)
    Y_ref = dP @ np.linalg.inv(P)
    Y = propagate_logderiv(Y0, r0, r1, lfac, k2, z, cf, 4000)
    assert np.allclose(Y, Y_ref, rtol=1e-6, atol=1e-6 * np.abs(Y_ref).max())


def test_closed_channel_rmatch_independence():
    """With no coupling outside a, K must not depend on the matching radius."""
    a, z = 20.0, 1
    etarg = [0.0, 0.0, 0.6]                  # channel 3 closed at the energy below
    etot = 0.35
    l = [0, 2, 1]
    rng = np.random.default_rng(3)
    Y = rng.normal(size=(3, 3)); Y = 0.5 * (Y + Y.T)
    blk = _block_from_Y(Y, a, etot, l, [0, 1, 2])
    hd = _hdata(z, a, etarg, blk)
    K1, op = kmatrix(blk, hd, etot, r_match=a + 40)
    K2, _ = kmatrix(blk, hd, etot, r_match=a + 80)
    assert op.tolist() == [0, 1]
    assert np.allclose(K1, K2, rtol=1e-5, atol=1e-6)
    assert np.allclose(K1, K1.T)


@pytest.mark.parametrize("l,k2,z,r", [(0, 1.0, 1, 60.0), (3, 0.5, 1, 80.0), (2, 2.0, 0, 50.0),
                                      (1, 0.01, 1, 60.0), (4, 0.001, 1, 50.0)])
def test_closed_channel_logderiv(l, k2, z, r):
    from pydbsr.outer import _whittaker_logderiv, closed_logderiv
    assert closed_logderiv(l, k2, z, r)[0] == pytest.approx(_whittaker_logderiv(l, k2, z, r), rel=1e-5)


@pytest.mark.parametrize("z,l,delta,e", [(1, 0, 0.3, 2.0), (1, 3, -0.7, 40.0), (2, 1, 1.1, 0.02)])
def test_relativistic_matching(z, l, delta, e):
    """kmatrix(relativistic=True) matches to Coulomb functions with the Dirac
    k^2 = 2e(1 + e/2c^2) and eta = -z(1 + e/c^2)/k; the non-relativistic
    matching of the same R-matrix gives a different phase."""
    from pydbsr.constants import C_AU
    a = 20.0
    k = np.sqrt(2 * e * (1 + e / (2 * C_AU ** 2)))
    eta = -z * (1 + e / C_AU ** 2) / k
    F, G, Fp, Gp = coulomb_fg(l, eta, k * a)
    u, up = F * np.cos(delta) + G * np.sin(delta), k * (Fp * np.cos(delta) + Gp * np.sin(delta))
    blk = _block_from_Y(np.array([[up / u]]), a, e, [l], [0])
    hd = _hdata(z, a, [0.0], blk)
    for r_extra in ((0.0, 25.0) if e < 10 else (0.0,)):     # (propagation over 25 a0 at k = 9 costs)
        K, _ = kmatrix(blk, hd, e, r_match=a + r_extra, relativistic=True)
        assert K[0, 0] == pytest.approx(np.tan(delta), rel=1e-7, abs=1e-9)
    Kn, _ = kmatrix(blk, hd, e, relativistic=False)
    assert abs(Kn[0, 0] - np.tan(delta)) > 1e-7 * (1 + e)
