"""e + Xe+ : comparison with the DBSR cross sections of Wang et al (2019),
Plasma Sources Sci. Technol. (Part I), supplementary file CrossSectionsIon.xlsx.

Their model: 67 states (5s2 5p5, 5s 5p6, 5p4 6s, 5p4 5d, 5p4 6p, 5p4 7s),
R-matrix radius a = 50 a0, thresholds adjusted to NIST, J <= 50.

Stages (the run can be interrupted and restarted: finished partial waves are kept):
  python xe_plus_vs_wang2019.py --xlsx CrossSectionsIon.xlsx --jmax 10  --cores 64 --mem 200
  python xe_plus_vs_wang2019.py --xlsx CrossSectionsIon.xlsx --jmax 25  ...   (adds J = 11..25)
Output: <workdir>/compare/*.png, rates.txt, omega.npz
"""
import argparse
from pathlib import Path

import numpy as np

import pydbsr as db
from pydbsr import reference
from pydbsr.outer import bugrova, maxwell

p = argparse.ArgumentParser()
p.add_argument("--xlsx", required=True, help="CrossSectionsIon.xlsx (Wang et al 2019 supplement)")
p.add_argument("--workdir", default="xe_plus_wang2019")
p.add_argument("--jmax", type=float, default=10.0, help="max total J of e + Xe+ (paper: 50)")
p.add_argument("--cores", type=int, default=32, help="CPU cores to use")
p.add_argument("--mem", type=float, default=150.0, help="memory budget, GB")
p.add_argument("--hd-threads", type=int, default=16, help="threads of one dbsr_hd3 (LAPACK)")
p.add_argument("--emax", type=float, default=60.0, help="max electron energy, eV")
p.add_argument("--de", type=float, default=0.0136, help="energy step, eV (paper: 0.0136 = 0.001 Ry)")
p.add_argument("--no-7s", action="store_true", help="drop 5p4 7s (60 instead of 68 states)")
p.add_argument("--no-corr", action="store_true", help="no 6d correlation orbital (term-dependent 5d)")
args = p.parse_args()

wd = Path(args.workdir)
ref = reference.read_xlsx(args.xlsx)

# ---------------------------------------------------------------- target (a = 50 a0 as in the paper)
ion = db.Ion("Xe", 1)
tdir = wd / "target"
if (tdir / "pydbsr_target.json").exists():
    tg = db.Target.load(tdir)
else:
    tg = db.Target(ion, core="[Kr]4d10", workdir=tdir, max_it=60, grid={"rmax": 50.0, "hmax": 0.5})
    tg.add("5s2 5p5")                              # reference: all orbitals optimised
    # all even-parity states in ONE CI calculation: 5s5p6 - 5p4 5d - 5p4 6s - 5p4 7s
    # mix strongly, and states from separate calculations would interact with
    # each other in the close-coupling equations (dbsr_mat3 'Target hamiltonian
    # errors', up to 0.1 a.u. with separate runs).  The 6d correlation orbital
    # (dbsr_mchf on the physical states) gives the term dependence of 5d.
    even = ["5s 5p6", "5s2 5p4 5d", "5s2 5p4 6s"] + ([] if args.no_7s else ["5s2 5p4 7s"])
    tg.add(even, correlation=[] if args.no_corr else ["5s2 5p4 6d"], mchf_max_it=150)
    tg.add("5s2 5p4 6p")
    tg.compute(jobs=min(args.cores, 6))
db.nist.assign(tg.states, ref.levels)              # NIST energies and level numbers from the xlsx
tg.save()
print(tg.table())
states = [s for s in tg.states if s.nist_no is not None]

# ---------------------------------------------------------------- inner region, streamed
sdir = wd / "scat"
pw = db.partial_waves(ion.nelc, args.jmax)
sc = db.Scattering(tg, sdir, states=states, partial_waves=pw, exp_energies=True)
if not (sdir / "cfg.001").exists() or len(list(sdir.glob("cfg.[0-9][0-9][0-9]"))) < len(pw):
    sc.prepare()
    sc.run_prep()
    sc.run_conf()
for d in sc.wave_sizes()[:4] + sc.wave_sizes()[-2:]:
    print(d)
sc.run_streamed(cores=args.cores, mem_gb=args.mem, hd_threads=args.hd_threads)

# ---------------------------------------------------------------- outer region
out = sc.outer(r_match=None)
energies = np.arange(0.01, args.emax, args.de)
cs = out.collision_strengths(energies, jobs=args.cores, per_partial_wave=True)
cdir = wd / "compare"
cdir.mkdir(exist_ok=True)
cs.save(cdir / "omega.npz")

nist_of = {s.name: s.nist_no for s in states}
name_of = {v: k for k, v in nist_of.items()}

# ---------------------------------------------------------------- rates: ours vs paper
lines = [f"# jmax = {args.jmax}; rate coefficients (cm^3/s); ratio = pydbsr / Wang2019",
         "# i  j  sheet        Te(eV)  Maxwell_ours  Maxwell_ref  ratio   Bugrova_ours  Bugrova_ref  ratio"]
for (i, j) in ref.transitions():
    if i not in name_of or j not in name_of:
        continue
    e_ref, s_ref = ref.sigma[(i, j)]
    for te in (2.0, 5.0, 10.0, 20.0):
        r = []
        for f in (maxwell(te), bugrova(te)):
            ours = cs.rate_eedf(name_of[i], name_of[j], f)
            v = 5.930969e7 * np.sqrt(e_ref)
            theirs = float(np.trapezoid(s_ref * v * f(e_ref), e_ref))
            r += [ours, theirs, ours / theirs if theirs > 0 else np.nan]
        lines.append(f"{i:3d} {j:3d}  {ref.sheet[(i, j)]:<10s} {te:6.1f}  "
                     f"{r[0]:11.3e} {r[1]:11.3e} {r[2]:6.2f}   {r[3]:11.3e} {r[4]:11.3e} {r[5]:6.2f}")
(cdir / "rates.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines[:40]))

# ---------------------------------------------------------------- plots
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for sheet in sorted(set(ref.sheet.values())):
        trs = [t for t in ref.transitions(sheet) if t[0] in name_of and t[1] in name_of][:12]
        if not trs:
            continue
        n = len(trs)
        fig, axs = plt.subplots((n + 2) // 3, 3, figsize=(12, 3 * ((n + 2) // 3)), squeeze=False)
        for ax, (i, j) in zip(axs.flat, trs):
            e_ref, s_ref = ref.sigma[(i, j)]
            ax.plot(e_ref, s_ref / 1e-16, lw=0.6, label="Wang 2019")
            ax.plot(cs.incident_energy(name_of[i]), cs.sigma(name_of[i], name_of[j]) / 1e-16, lw=0.6,
                    label=f"pydbsr, J<={args.jmax:g}")
            ax.set_title(f"{i} -> {j}: {ref.label(j)}", fontsize=8)
            ax.set_xscale("log")
            ax.set_xlabel("E (eV)", fontsize=8)
            ax.set_ylabel("$10^{-16}$ cm$^2$", fontsize=8)
        axs.flat[0].legend(fontsize=7)
        for ax in list(axs.flat)[n:]:
            ax.set_visible(False)
        fig.tight_layout()
        fig.savefig(cdir / f"{sheet.replace('->', '_to_')}.png", dpi=130)
        plt.close(fig)
except ImportError:
    pass
print(f"results in {cdir}")
