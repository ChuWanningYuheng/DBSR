"""Physical constants (CODATA 2018) and unit conversions."""
AU_EV = 27.211386245988          # hartree in eV
AU_CM = 219474.6313632           # hartree in cm^-1
RY_EV = AU_EV / 2
A0_CM = 0.529177210903e-8        # Bohr radius in cm
PI_A0_2_CM2 = 3.141592653589793 * A0_CM ** 2   # pi a0^2 in cm^2 = 8.797e-17
K_B_EV = 8.617333262e-5          # Boltzmann constant, eV/K

# SI (CODATA 2018 exact / recommended)
_H = 6.62607015e-34              # Planck constant, J s
_ME = 9.1093837015e-31           # electron mass, kg
_KB = 1.380649e-23               # Boltzmann constant, J/K
_EV = 1.602176634e-19            # J per eV
# electron speed v = V_EV_CM_S * sqrt(E[eV]) in cm/s  (non-relativistic)
V_EV_CM_S = (2.0 * _EV / _ME) ** 0.5 * 100.0
# Maxwellian rate q = RATE_UPS / (g_i sqrt(T[K])) * Upsilon * exp(-dE/kT)  in cm^3/s:
# RATE_UPS = h^2 / ((2 pi m)^3/2 sqrt(k_B))  (= 8.629e-6 cm^3 s^-1 K^1/2)
RATE_UPS = _H ** 2 / ((2.0 * 3.141592653589793 * _ME) ** 1.5 * _KB ** 0.5) * 1e6
C_AU = 137.035999084             # speed of light, atomic units (1/alpha)
EV_CM = AU_CM / AU_EV            # cm^-1 per eV
