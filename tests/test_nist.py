from pathlib import Path

from pydbsr import nist
from pydbsr.structure import State

DATA = Path(__file__).parent / "data"


def test_parse_levels():
    lv = nist.read_levels(DATA / "nist_sample.tsv")
    assert len(lv) == 5                      # the ionisation limit row is skipped
    assert lv[0].two_j == 3 and lv[0].parity == -1 and lv[0].energy_cm == 0.0
    assert lv[2].energy_cm == 93068.4 and lv[2].parity == 1
    assert nist.nist_shells("5s2.5p4.(3P).6s") == {(5, 0): 2, (5, 1): 4, (6, 0): 1}


def test_assign():
    lv = nist.read_levels(DATA / "nist_sample.tsv")
    st = [State("a", "5s2 5p5", "", 3, -1, -1.0), State("b", "5s2 5p5", "", 1, -1, -0.9),
          State("c", "5s 5p6", "", 1, 1, -0.5), State("d", "5s2 5p4 6s", "", 5, 1, -0.6),
          State("e", "5s2 5p4 6s", "", 1, 1, -0.55)]
    res = dict((s.name, l) for s, l in nist.assign(st, lv, verbose=False))
    assert st[1].exp_energy_cm == 10537.01
    assert st[2].exp_energy_cm == 90873.8
    assert st[3].exp_energy_cm == 93068.4
    assert res["e"] is None and st[4].exp_energy_cm is None


ASCII = """<html><body><pre>
-----------------------------------------------------------------
 Configuration       | Term  |   J |      Level (cm-1)      |
-----------------------------------------------------------------
 5s2.5p5             | 2P*   | 3/2 |          0.000         |
                     |       | 1/2 |      10537.010         |
-----------------------------------------------------------------
 5s.5p6              | 2S    | 1/2 |      90873.83          |
 Xe III (3P<2>)      | Limit |  -- |     [169180]           |
</pre></body></html>"""

CSV = '''Configuration,Term,J,"Level (cm-1)",Reference
"5s2.5p5","2P*","3/2","0.000",""
"5s2.5p5","2P*","1/2","10537.010",""
'''

HTML_PAGE = '<!DOCTYPE html><html><head><title>NIST</title></head><body><a href="/" title="Home"><img/></a>' \
            '<p>Atomic Spectra Database, J 3/2 Home"><</p></body></html>'


def test_parse_ascii_html_csv():
    lv = nist.parse_levels(ASCII)
    assert [(l.config, l.two_j, l.energy_cm) for l in lv] == [
        ("5s2.5p5", 3, 0.0), ("5s2.5p5", 1, 10537.01), ("5s.5p6", 1, 90873.83)]
    assert lv[1].term == "2P*" and lv[1].parity == -1
    assert [l.two_j for l in nist.parse_levels(CSV)] == [3, 1]


def test_html_page_is_not_a_table():
    assert nist.parse_levels(HTML_PAGE) == []


LINES_CSV = '''element,sp_num,obs_wl_air(nm),unc_obs_wl,ritz_wl_air(nm),unc_ritz_wl,intens,Aki(s^-1),Acc,Ei(cm-1),Ek(cm-1),conf_i,term_i,J_i,conf_k,term_k,J_k,Type
Xe,3,="400.0000",="0.001",="400.0001",="",="1000",="1.0e+08",="C",="100.000",="25100.000",="5s2.5p3.(4S*).6s",="5S*",="2",="5s2.5p3.(4S*).6p",="5P",="3",
Xe,3,="",="",="410.0000",="",="",="",="",="100.000",="24490.000",="5s2.5p3.(4S*).6s",="5S*",="2",="5s2.5p3.(4S*).6p",="5P",="2",
'''


def test_parse_lines_and_upper_levels():
    ln = nist.parse_lines(LINES_CSV)
    assert [x.wavelength_nm for x in ln] == [400.0, 410.0]      # obs, then ritz when obs missing
    assert ln[0].aki == 1.0e8 and ln[0].ek_cm == 25100.0 and ln[0].j_k == "3"
    levels = [nist.Level("5s2.5p3.(4S*).6p", "5P", 6, 25100.2, -1, 40),
              nist.Level("5s2.5p3.(4S*).6p", "5P", 4, 24490.0, -1, 38)]
    m = nist.upper_levels(ln, levels)
    assert m[0][1].no == 40 and m[1][1].no == 38


def test_levels_csv_roundtrip(tmp_path):
    lv = nist.read_levels(DATA / "nist_sample.tsv")
    p = nist.save_levels_csv(lv, tmp_path / "x.csv")
    lv2 = nist.read_levels(p)
    key = lambda x: (x.no, x.config, x.two_j, x.energy_cm)
    assert sorted(map(key, lv)) == sorted(map(key, lv2))
