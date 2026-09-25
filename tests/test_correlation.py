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


def test_superseded_reference(tmp_path):
    from pydbsr import Ion, Target
    tg = Target(Ion("Xe", 1), core="[Kr]4d10", workdir=tmp_path)
    tg.add("5s2 5p5")
    tg.add(["5s 5p6", "5s2 5p4 5d"])
    tg.add(["5s2 5p5", "5s2 5p4 6p"])
    assert tg.superseded() == {tg.specs[0].name}
    assert tg.specs[2].varied == "6p-,6p"


def test_cfg_orbitals_high_l(tmp_path):
    """DBSR labels l >= 21 with chr(l + 102); mk must see every continuum l."""
    from pydbsr.scattering import Scattering
    s = State("a", "5s2 5p5", "", 3, -1, 0.0)
    sc = Scattering([s], tmp_path, ion=Ion("Xe", 1), jmax=0, progress=None)
    peel = [" 5s 1", " 5p-1", "10d 1", " kl-3", " kp 2", " k" + chr(30 + 102) + "-1", " k" + chr(54 + 102) + " 2"]
    text = ("Core subshells:  -7446.4\n  1s   2s   2p-  2p   4d-  4d \nPeel subshells:\n" + "".join(peel) +
            "\nCSF(s):\n")
    (tmp_path / "cfg.001").write_bytes(text.encode("latin-1"))
    orbs = sc._cfg_orbitals(1)
    assert sorted(l for c, l in orbs if c) == [1, 8, 30, 54]
    assert max(l for c, l in orbs if not c) == 2
    assert sc.multipole_max(1) == 56
