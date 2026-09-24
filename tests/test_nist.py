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
