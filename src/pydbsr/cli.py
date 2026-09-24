"""Command line: ``pydbsr info | bin | run PROGRAM ARGS | nist SPECTRUM``."""
from __future__ import annotations

import argparse
import os
import sys


def main(argv=None):
    from . import __version__, available_programs, bin_dir, nist
    from .coulomb import backend

    p = argparse.ArgumentParser(prog="pydbsr", description="Python interface to DBSR")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("info", help="installation info")
    sub.add_parser("bin", help="print the directory with the DBSR executables")
    r = sub.add_parser("run", help="run a DBSR program: pydbsr run dbsr_hf name ...")
    r.add_argument("program")
    r.add_argument("args", nargs=argparse.REMAINDER)
    n = sub.add_parser("nist", help="download NIST levels, e.g. pydbsr nist 'Xe II' --csv xe2.csv")
    n.add_argument("spectrum")
    n.add_argument("-o", "--output", help="save the raw NIST answer to this file")
    n.add_argument("--csv", help="save the levels as a clean CSV")
    ln = sub.add_parser("lines", help="NIST lines with their levels, e.g. pydbsr lines 'Xe III' 474 479")
    ln.add_argument("spectrum")
    ln.add_argument("wmin", type=float, help="nm")
    ln.add_argument("wmax", type=float, help="nm")
    ln.add_argument("--csv", help="save the lines as CSV")
    a = p.parse_args(argv)

    if a.cmd == "info":
        print(f"pydbsr {__version__}")
        print(f"executables: {bin_dir() or '$PATH'}")
        print("programs:", " ".join(available_programs()))
        print(f"Coulomb functions: {backend()}")
    elif a.cmd == "bin":
        print(bin_dir() or "")
    elif a.cmd == "run":
        from .runner import which
        exe = which(a.program)
        os.execv(exe, [exe, *a.args])
    elif a.cmd == "nist":
        levels = nist.fetch_levels(a.spectrum)
        if a.output:
            src = nist._cache_dir() / ("nist_" + a.spectrum.replace(" ", "_") + ".tsv")
            with open(a.output, "w") as f:
                f.write(src.read_text())
        if a.csv:
            nist.save_levels_csv(levels, a.csv)
        for lv in levels:
            print(f"{lv.no:4d} {lv.config:<28s} {lv.term:<10s} {lv.two_j:3d}/2 {lv.energy_cm:14.3f}")
    elif a.cmd == "lines":
        found = nist.fetch_lines(a.spectrum, a.wmin, a.wmax)
        if a.csv:
            import csv
            with open(a.csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["wavelength_nm", "intensity", "Aki_s-1", "Ei_cm-1", "Ek_cm-1",
                            "conf_i", "term_i", "J_i", "conf_k", "term_k", "J_k"])
                for x in found:
                    w.writerow([x.wavelength_nm, x.intensity, x.aki, x.ei_cm, x.ek_cm,
                                x.conf_i, x.term_i, x.j_i, x.conf_k, x.term_k, x.j_k])
        for x in found:
            print(x)
    return 0


if __name__ == "__main__":
    sys.exit(main())
