import numpy as np
import pytest

from pydbsr import crm, transitions as tr

# zf_res as written by dbsr_dmat3 (Gen_zf, gf='f'), two lines, the first with
# the velocity form
ZF = """
+ 3   -7232.12345678  5p4_6s1_j3_1
- 5   -7232.03456789  5p4_6p1_j5_1
   19508.73 CM-1     5125.95 ANGS(VAC)   5124.52 ANGS(AIR)
 E1  S =  1.23450D+00   FIK=  1.10000D-01   AKI =  4.50000D+07
          1.10000D+00         9.80000D-02          4.00000D+07

+ 1   -7232.10000000  5p4_6s1_j1_1
- 3   -7232.03456789  5p4_6p1_j3_2
   14361.10 CM-1     6963.25 ANGS(VAC)   6961.33 ANGS(AIR)
 E1  S =  2.00000D-01   FIK=  5.00000D-02   AKI =  1.00000D+07
"""


def test_parse_zf_res():
    t = tr.parse_zf_res(ZF)
    assert len(t) == 2
    a, b = t
    assert (a.lower, a.upper, a.two_j_lower, a.two_j_upper) == ("5p4_6s1_j3_1", "5p4_6p1_j5_1", 3, 5)
    assert a.kind == "E1" and np.isclose(a.de_cm, 19508.73) and np.isclose(a.A, 4.5e7)
    assert np.isclose(a.gf, 0.11 * 4) and np.isclose(a.A_v, 4.0e7) and np.isclose(a.gauge_ratio, 4.0 / 4.5)
    assert b.A_v is None and np.isclose(b.S, 0.2)


def test_a_exp_scaling_and_table():
    a, b = tr.parse_zf_res(ZF)
    assert np.isclose(a.a_exp(2 * a.de_cm), 8 * a.A)                  # E1: dE^3
    st = [type("S", (), dict(name=n, exp_energy_cm=e, nist_label=n, nist_no=i))()
          for i, (n, e) in enumerate([("5p4_6s1_j3_1", 0.0), ("5p4_6p1_j5_1", 19508.73)])]
    tab = tr.TransitionTable([a], st)
    assert np.isclose(tab.branching("5p4_6p1_j5_1")["5p4_6s1_j3_1"], 1.0)
    assert np.isclose(tab.lifetime_ns("5p4_6p1_j5_1"), 1e9 / a.A)
    assert abs(tab.wavelength_nm("5p4_6p1_j5_1", "5p4_6s1_j3_1") - 512.452) < 0.01


@pytest.mark.parametrize("kind,j1,p1,j2,p2,ok", [
    ("E1", 3, 1, 5, -1, True), ("E1", 3, 1, 5, 1, False), ("E1", 1, 1, 5, -1, False),
    ("E1", 0, 1, 0, -1, False), ("M1", 3, -1, 1, -1, True), ("E2", 1, 1, 5, 1, True), ("E2", 1, 1, 1, 1, False)])
def test_selection_rules(kind, j1, p1, j2, p2, ok):
    assert tr.allowed(kind, j1, p1, j2, p2) == ok


def test_air_wavelength():
    assert abs(tr.air_wavelength(553.701) - 553.548) < 0.002     # Ba I resonance line


def test_rate_from_sigma_constant():
    # constant sigma above threshold 0: <sigma v> = sigma * <v> = sigma sqrt(8 kT / pi m)
    E = np.linspace(0, 400, 2000)
    k = crm.rate_from_sigma(E, np.full_like(E, 1e-16), Te_eV=2.0, threshold_eV=0.0, tail="1/E")
    vbar = crm.V_EV * np.sqrt(8 * 2.0 / np.pi / 2)        # sqrt(8kT/pi m) = V_EV*sqrt(4T/pi)
    assert abs(k / (1e-16 * vbar) - 1) < 2e-3


def test_two_level_coronal_limit_and_boltzmann():
    d = crm.CRMData("X", ["g", "u"], [1, 3], [0.0, 8065.544], Te=[0.5, 1, 2, 5])
    d.add_rate("g", "u", [1e-10] * 4)
    d.add_A("u", "g", 1e8)
    p = crm.steady_state(d, ne=1e10, Te=1.0)
    assert np.isclose(p[1], 1e10 * 1e-10 / (1e8 + 1e10 * 1e-10 / 3 * np.exp(1.0)), rtol=1e-6)
    d.A.clear()                                           # no radiation: Boltzmann
    p = crm.steady_state(d, ne=1e10, Te=1.0)
    assert np.isclose(p[1], 3 * np.exp(-1.0), rtol=1e-6)
    s = crm.sensitivity(d, [], 1e10, 1.0)
    assert s == {}


def test_sensitivity_and_save(tmp_path):
    d = crm.CRMData("X", ["g", "m", "u"], [1, 5, 3], [0.0, 8000.0, 20000.0], Te=[0.5, 1, 2, 5])
    d.add_rate("g", "m", [1e-11, 3e-10, 1e-9, 2e-9])
    d.add_rate("g", "u", [1e-14, 1e-12, 1e-10, 1e-9])
    d.add_rate("m", "u", [1e-9, 5e-9, 1e-8, 1e-8])
    d.add_A("u", "g", 1e8)
    d.add_A("u", "m", 3e7)
    d.save(tmp_path / "x.json")
    e = crm.CRMData.load(tmp_path / "x.json")
    assert e.k.keys() == d.k.keys() and e.A == d.A
    s = crm.sensitivity(e, [("u", "g")], 1e11, 1.0, tau=1e-5)
    v = s[("u", "g")]
    assert v["I"] > 0 and v["dlnI_dlnne"] > 1.0          # stepwise via the metastable: faster than ne
    assert 0 < crm.escape_factor_doppler(10) < 1 and np.isclose(crm.escape_factor_doppler(0), 1.0)
    k0 = crm.doppler_k0(553.548, 1, 3, 1.19e8, 1500, 137.33)
    assert 2.5e-11 < k0 < 4e-11
