"""End-to-end run: e + Xe+ with a 3-state target (about 1 minute)."""
import numpy as np
import pytest

import pydbsr as db

pytestmark = pytest.mark.skipif(db.bin_dir() is None, reason="DBSR programs not installed")


@pytest.mark.slow
def test_xe_plus_small(tmp_path):
    ion = db.Ion("Xe", 1)
    tg = db.Target(ion, core="[Kr]4d10", workdir=tmp_path / "target")
    tg.add("5s2 5p5")
    tg.add("5s 5p6")
    tg.compute(progress=None)
    assert [s.two_j for s in tg.states] == [3, 1, 1]
    sc = db.Scattering(tg, tmp_path / "scat", jmax=1, jobs=2, exp_energies=False, progress=None)
    sc.run()
    assert len(sc.h_files()) == 4
    out = sc.outer()
    assert out.h.ntarg == 3
    cs = out.collision_strengths([0.5, 3.0, 20.0])
    om = cs.omega
    assert np.all(om >= 0) and np.allclose(om, om.transpose(0, 2, 1))
    assert om[0, 0, 1] == 0.0 and om[1, 0, 1] > 0      # 2P1/2 threshold ~1.3 eV
    bound = sc.bound_states(msol=5)
    assert bound[0]["two_j"] == 0 and bound[0]["E_bind"] < -0.3   # Xe 5p6 ground state
