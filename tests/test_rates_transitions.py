import numpy as np
import pytest

from pydbsr import rates, transitions as tr

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
    k = rates.rate_from_sigma(E, np.full_like(E, 1e-16), Te_eV=2.0, threshold_eV=0.0, tail="1/E")
    vbar = rates.V_EV * np.sqrt(8 * 2.0 / np.pi / 2)        # sqrt(8kT/pi m) = V_EV*sqrt(4T/pi)
    assert abs(k / (1e-16 * vbar) - 1) < 2e-3


def test_rates_on_grid_and_detailed_balance():
    E = np.linspace(2.0, 200, 3000)
    k = rates.rates_on_grid(E, np.full_like(E, 1e-16), [1.0, 2.0], threshold_eV=2.0)
    assert k[1] > k[0] > 0
    kd = rates.deexcitation(k, 1, 3, 2.0, np.array([1.0, 2.0]))
    assert np.allclose(kd, k / 3 * np.exp([2.0, 1.0]))


def test_wang_format_roundtrip(tmp_path):
    pytest.importorskip("openpyxl")
    from pydbsr import reference
    from pydbsr.nist import Level
    lv = [Level("5p5", "2P*", 3, 0.0, -1, 1), Level("5p4(3P2)6s", "2[2]", 5, 93068.44, 1, 2)]
    E = np.array([11.6, 12.0, 15.0])
    reference.write_xlsx(tmp_path / "s.xlsx", lv, {(1, 2): (E, np.array([1.0, 0.5, 0.2]) * 1e-16)},
                         sheet_of=lambda k: "gs->6s")
    ref = reference.read_xlsx(tmp_path / "s.xlsx")
    assert [l.no for l in ref.levels] == [1, 2] and abs(ref.levels[1].energy_cm - 93068.44) < 0.1
    e, s = ref.sigma[(1, 2)]
    assert np.allclose(e, E) and np.allclose(s, [1e-16, 0.5e-16, 0.2e-16])
    reference.write_rates_xlsx(tmp_path / "k.xlsx", [1.0, 2.0], {(1, 2): [1e-14, 2e-13]})
    assert (tmp_path / "k.csv").read_text().splitlines()[1].startswith("1.0,1.00000e-14")
