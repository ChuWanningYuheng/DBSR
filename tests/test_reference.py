import numpy as np
import pytest

openpyxl = pytest.importorskip("openpyxl")

from pydbsr import reference  # noqa: E402


def test_read_xlsx_layout(tmp_path):
    """Layout of the Wang et al (2019) supplement: 'i->j' above the sigma column."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "NIST Level Table"
    ws.append(["No.", "Configuration", "Term", "J", "Level(eV)"])
    ws.append([1, "5p5", "2P°", "3/2", 0.0])
    ws.append([2, "5p5", "2P°", "1/2", 1.306423])
    ws2 = wb.create_sheet("gs->x")
    ws2.append([None, "1->2", None, "2->1"])
    ws2.append(["Energy(eV)", "Sigma(1E-16 cm^2)", "Energy(eV)", "Sigma(1E-16 cm^2)"])
    ws2.append([1.4, 0.5, 0.2, 7.0])
    ws2.append([1.35, 0.6, 0.1, 8.0])
    ws2.append([2.0, 0.4, None, None])
    p = tmp_path / "ref.xlsx"
    wb.save(p)
    ref = reference.read_xlsx(p)
    assert [lv.no for lv in ref.levels] == [1, 2]
    e, s = ref.sigma[(1, 2)]
    assert np.allclose(e, [1.35, 1.4, 2.0]) and np.allclose(s, [0.6e-16, 0.5e-16, 0.4e-16])
    e, s = ref.sigma[(2, 1)]
    assert np.allclose(e, [0.1, 0.2]) and np.allclose(s, [8e-16, 7e-16])
