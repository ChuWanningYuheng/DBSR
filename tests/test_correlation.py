from pydbsr import Ion
from pydbsr.scattering import Scattering
from pydbsr.structure import ConfigSpec, State, _correlation_orbitals


def test_correlation_orbitals():
    spec = ConfigSpec(conf="5s 5p6 + 5s2 5p4 5d", name="x", correlation="5s2 5p4 6d")
    assert _correlation_orbitals(spec) == ["6d-", "6d"]
    assert _correlation_orbitals(ConfigSpec(conf="5s2 5p4 5d", name="y")) == []


def test_orth_conditions_format(tmp_path):
    s = State("a", "5s2 5p4 5d", "", 1, 1, 0.0, correlation_orbitals=["6d-", "6d"])
    t = State("b", "5s2 5p5", "", 3, -1, -1.0)
    sc = Scattering([s, t], tmp_path, ion=Ion("Xe", 1), jmax=0, progress=None)
    lines = sc.orth_conditions()
    # columns read by read_orth_jj: Aort(2:6), Aort(8:12), Aort(15:15)
    assert lines == ["< kd- | 6d- >=0", "< kd  | 6d  >=0"]
    assert all(line[6] == "|" and line[14] == "0" for line in lines)
