import shutil
from pathlib import Path

import pytest

from pydbsr import io
from pydbsr.atoms import Ion, expand_core, core_electrons
from pydbsr.runner import bin_dir, run

DATA = Path(__file__).parent / "data"


def test_config_parsing():
    assert io.to_dbsr_conf("5s2 5p4 5d") == "5s(2)5p(4)5d(1)"
    assert io.to_dbsr_conf("5s(2)5p(5)") == "5s(2)5p(5)"
    assert io.config_parity("5s2 5p5") == -1
    assert io.config_parity("5s2 5p4 6s") == 1
    assert io.config_nelectrons("5s 5p6") == 7
    with pytest.raises(ValueError):
        io.parse_config("5p7")


def test_core_and_ion():
    sh = expand_core("[Kr]4d10")
    assert sh[-1] == "4d" and core_electrons(sh) == 46
    assert expand_core("[4d]") == sh
    ion = Ion("Xe", 1)
    assert ion.z == 54 and ion.nelc == 53 and ion.spectrum == "Xe II"


def test_partial_waves():
    pw = io.partial_waves(53, 1)          # 54 electrons -> integer J
    assert pw == [(0, 1), (0, -1), (2, 1), (2, -1)]
    pw = io.partial_waves(36, 1.5)        # 37 electrons -> half-integer J
    assert pw[0] == (1, 1) and pw[-1] == (3, -1)


def test_read_j_and_csf_j():
    sols = io.read_j(DATA / "5p4_6s1.j")
    cf = io.CFile.read(DATA / "5p4_6s1.c")
    assert len(sols) == 8 and len(cf.csfs) == 8
    js = sorted(io.csf_jp(cf.csfs[s.ic1 - 1][2])[0] for s in sols)
    assert js == [1, 1, 1, 3, 3, 3, 5, 5]
    assert all(io.csf_jp(c[2])[1] == 1 for c in cf.csfs)


def test_target_jj_roundtrip(tmp_path):
    t = io.TargetJJ("e + Xe+", 54, 53, [io.TargetEntry("a"), io.TargetEntry("b")],
                    [(0, 1, None), (2, -1, None)], [(1, "pert")])
    t.write(tmp_path / "target_jj")
    r = io.TargetJJ.read(tmp_path / "target_jj")
    assert [s.name for s in r.states] == ["a", "b"]
    assert r.partial_waves == [(0, 1, None), (2, -1, None)]
    assert r.perturbers == [(1, "pert")]
    for line in (tmp_path / "target_jj").read_text().splitlines():
        if line[:4].strip().rstrip(".").isdigit() and "." in line[:5]:
            assert len(line) <= 13          # dbsr_prep reads a perturber name after column 13


@pytest.mark.skipif(bin_dir() is None and shutil.which("jcfile") is None, reason="DBSR programs not installed")
def test_write_state_matches_jcfile(tmp_path):
    for f in ("5p4_6s1.c", "5p4_6s1.j"):
        shutil.copy(DATA / f, tmp_path / f)
    cf = io.CFile.read(tmp_path / "5p4_6s1.c")
    for sol in io.read_j(tmp_path / "5p4_6s1.j"):
        io.write_state("5p4_6s1", sol, cf, None, tmp_path / f"py_{sol.index}")
        run("jcfile", ["5p4_6s1.j", str(sol.index), f"f_{sol.index}.c", "0.0000001"], tmp_path)
        assert (tmp_path / f"py_{sol.index}.c").read_text() == (tmp_path / f"f_{sol.index}.c").read_text()
