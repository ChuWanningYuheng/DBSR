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
