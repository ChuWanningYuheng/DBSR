import pytest
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


def test_orbitals_from(tmp_path):
    """e + Ba with one orbital set: the even states take 5d and 6p from the odd ones."""
    from pydbsr import Ion, Target
    tg = Target(Ion("Ba", 0), core="[Xe]", workdir=tmp_path)
    tg.add("6s2")
    odd = tg.add(["6s 6p", "5d 6p"])
    even = tg.add(["6s2", "6s 5d", "6p2", "6s 6d"], orbitals_from=odd.name)
    assert odd.varied == "6p-,6p,5d-,5d" and odd.inp == "6s2.bsw"
    assert even.varied == "6d-,6d" and even.inp == f"{odd.name}.bsw"
    with pytest.raises(ValueError):
        tg.add("6s 7s", orbitals_from="nope")


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


def test_complete_target_orb(tmp_path):
    """Orbitals of non-leading CSFs (5d- here) are added to target_orb, correlation ones (6d) not."""
    (tmp_path / "targ_001.c").write_text(
        "Core subshells:  -7446.0\n  1s\nPeel subshells:\n 5s 1 5p-1 5p 1 5d-1 5d 1 6d 1\nCSF(s):\n"
        " 5s 1( 2) 5p-1( 2) 5p 1( 2) 5d 1( 1)                                      -0.80\n"
        "                          2      5/2\n                                    5/2-\n"
        " 5s 1( 2) 5p-1( 2) 5p 1( 2) 5d-1( 1)                                       0.55\n"
        "                          2      3/2\n                                    5/2-\n"
        " 5s 1( 2) 5p-1( 2) 5p 1( 2) 6d 1( 1)                                       0.20\n"
        "                          2      5/2\n                                    5/2-\n"
        " 5s 1( 2) 5p-1( 1) 5p 1( 3) 5d 1( 1)                                       0.05\n"
        "               1/2      3/2      5/2\n                           2        5/2-\n"
        "*\n")
    head = "target   001      a" + " " * 60 + "\n"
    orig = (head + " 5s 1     2.0      5s 1   1.000\n 5p-1     2.0      5p-1   1.000\n"
            " 5p 1     2.0      5p 1   1.000\n 5d 1     1.0      5d 1   1.000\n*\n")
    (tmp_path / "target_orb").write_text(orig)
    s = State("a", "5s2 5p4 5d", "", 5, 1, 0.0, correlation_orbitals=["6d-", "6d"])
    sc = Scattering([s], tmp_path, ion=Ion("Xe", 1), jmax=0, progress=None)
    assert sc.complete_target_orb() == 1
    lines = (tmp_path / "target_orb").read_text().splitlines()
    assert lines[-2] == " 5d-1     0.3      5d-1   1.000"
    assert lines[-1] == "*"
    assert sc.complete_target_orb() == 0                     # idempotent


def test_complete_target_orb_two_orbital_sets(tmp_path):
    """An added orbital of the second set gets the substitution orbital dbsr_prep3 gave it
    in another state (5d-2 -> 5d-1, e + Ba); older entries 5d-2 -> 5d-2 are corrected."""
    (tmp_path / "targ_001.c").write_text(
        "Core subshells:  -7446.0\n  1s\nPeel subshells:\n 6s 1 6p-2 5d-2\nCSF(s):\n"
        " 6s 1( 1) 6p-2( 1)                                                          0.90\n"
        "                          1      1/2\n                                    1-\n"
        " 5d-2( 1) 6p-2( 1)                                                          0.43\n"
        "                          3/2    1/2\n                                    1-\n"
        "*\n")
    (tmp_path / "targ_002.c").write_text(
        "Core subshells:  -7446.0\n  1s\nPeel subshells:\n 6p-2 5d-2\nCSF(s):\n"
        " 5d-2( 1) 6p-2( 1)                                                          1.00\n"
        "                          3/2    1/2\n                                    1-\n"
        "*\n")
    pad = " " * 60
    old = (f"target   001      a{pad}\n 6s 1     1.0      6s 1   1.000\n 6p-2     1.0      6p-2   1.000\n"
           " 5d-2     0.2      5d-2   1.000\n*\n"
           f"target   002      b{pad}\n 6p-2     1.0      6p-2   1.000\n 5d-2     1.0      5d-1   0.975\n*\n")
    (tmp_path / "target_orb").write_text(old)
    s = State("a", "6s 6p", "", 1, -1, 0.0)
    sc = Scattering([s], tmp_path, ion=Ion("Ba", 0), jmax=0.5, progress=None)
    assert sc.complete_target_orb() == 1                     # the older entry corrected
    lines = (tmp_path / "target_orb").read_text().splitlines()
    assert lines[3] == " 5d-2     0.2      5d-1   0.975"
    assert sc.complete_target_orb() == 0
    # fresh file from dbsr_prep3: the entry is added with the right substitution orbital
    (tmp_path / "target_orb").write_text(old.replace(" 5d-2     0.2      5d-2   1.000\n", ""))
    assert sc.complete_target_orb() == 1
    lines = (tmp_path / "target_orb").read_text().splitlines()
    assert lines[3] == " 5d-2     0.2      5d-1   0.975"
