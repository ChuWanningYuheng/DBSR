"""e + Xe+ excitation cross sections with pydbsr.

Target model: 5s2 5p5, 5s 5p6, 5s2 5p4 5d, 5s2 5p4 6s, 5s2 5p4 6p  (60 jj states),
thresholds from NIST, close-coupling with the lowest N states, J <= Jmax.

Run:  python xe_plus_excitation.py [--nstates 20] [--jmax 5] [--jobs 8] [--threads 4]
"""
import argparse
import warnings

import numpy as np

import pydbsr as db

p = argparse.ArgumentParser()
p.add_argument("--nstates", type=int, default=20, help="target states in the close-coupling expansion")
p.add_argument("--jmax", type=float, default=5, help="max total J of e + Xe+")
p.add_argument("--jobs", type=int, default=4, help="partial waves in parallel")
p.add_argument("--threads", type=int, default=4, help="OpenMP/BLAS threads per partial wave")
p.add_argument("--nist", default=None, help="saved NIST table (if no internet on the cluster)")
p.add_argument("--emax", type=float, default=40.0, help="max electron energy, eV")
args = p.parse_args()

# ---------------------------------------------------------------- target
ion = db.Ion("Xe", 1)                                   # Xe+ : Z = 54, N = 53
# grid: R-matrix radius 50 a0, B-spline step <= 0.5 a0 at large r (electrons up to ~40-50 eV)
tg = db.Target(ion, core="[Kr]4d10", workdir="xe_target", max_it=40, grid={"rmax": 50.0, "hmax": 0.5})
tg.add("5s2 5p5")          # reference: all orbitals optimised
# states of one parity in one CI calculation: orthogonal and non-interacting
# target states (otherwise dbsr_mat3 reports 'Target hamiltonian errors');
# 6d = correlation orbital for the term dependence of 5d
tg.add(["5s 5p6", "5s2 5p4 5d", "5s2 5p4 6s"], correlation=["5s2 5p4 6d"])
tg.add(["5s2 5p5", "5s2 5p4 6p"])   # replaces the reference states
tg.compute(jobs=4)          # dbsr_hf (LS -> jj) + splitting into J-states (jcfile)

# ---------------------------------------------------------------- NIST levels
try:
    levels = db.nist.read_levels(args.nist) if args.nist else db.nist.fetch_levels(ion)
    db.nist.assign(tg.states, levels)
    use_exp = True
except (OSError, RuntimeError) as e:                      # no internet or NIST unavailable
    warnings.warn(f"NIST not available ({e}); using computed thresholds")
    use_exp = False
print(tg.table())

# ---------------------------------------------------------------- inner region
states = tg.select(max_states=args.nstates)
if use_exp and any(s.exp_energy_cm is None for s in states):
    warnings.warn("some states have no NIST level -> computed thresholds are used")
    use_exp = False

sc = db.Scattering(tg, "xe_scat", states=states, jmax=args.jmax,
                   exp_energies=use_exp,
                   prep_args={"eps_core": "1.d-2", "eps_ovl": "1.d-2"},   # as in the original script
                   jobs=args.jobs, threads=args.threads)
sc.run()                                        # prep -> conf -> breit -> mat -> hd

# ---------------------------------------------------------------- outer region
outer = sc.outer(r_match=100.0)                 # propagate long-range potentials to 100 a0
energies = np.linspace(0.01, args.emax, 4000)   # eV above the ground state
cs = outer.collision_strengths(energies, jobs=args.jobs * args.threads, per_partial_wave=True)
cs.save("xe_plus_omega.npz")

# ---------------------------------------------------------------- output
tj = sc.target_table().states                  # order used in the h-files (sorted by energy)
names = {s.name: s for s in states}
with open("xe_plus_sigma_from_ground.dat", "w") as f:
    f.write("# E_inc(eV) " + " ".join(f"sigma(1->{j + 1})[cm2]" for j in range(1, len(tj))) + "\n")
    cols = [cs.sigma(0, j) for j in range(1, len(tj))]
    for ie, e in enumerate(cs.incident_energy(0)):
        f.write(f"{e:10.4f} " + " ".join(f"{c[ie]:12.4e}" for c in cols) + "\n")
print("thresholds (eV):", np.round(cs.thresholds, 4))
print("partial-wave convergence of Omega(1->2) at the highest energy:", cs.omega_pw[-1, :, 0, 1])

try:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for j in range(1, min(len(tj), 8)):
        ax.plot(cs.incident_energy(0), cs.sigma(0, j), label=f"1 -> {j + 1}: {tj[j].name}")
    ax.set_xlabel("electron energy (eV)")
    ax.set_ylabel("cross section (cm$^2$)")
    ax.set_yscale("log")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("xe_plus_sigma.png", dpi=150)
except ImportError:
    pass
