"""Elements, ions, and closed-core notation."""
from __future__ import annotations

from dataclasses import dataclass

from .io import L_SYMBOLS, parse_config

SYMBOLS = """H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn
Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm
Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu
Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og""".split()

NOBLE_CORES = {
    "He": "1s",
    "Ne": "1s 2s 2p",
    "Ar": "1s 2s 2p 3s 3p",
    "Kr": "1s 2s 2p 3s 3p 3d 4s 4p",
    "Xe": "1s 2s 2p 3s 3p 3d 4s 4p 4d 5s 5p",
    "Rn": "1s 2s 2p 3s 3p 3d 4s 4p 4d 4f 5s 5p 5d 6s 6p",
}

ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII",
         "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX"]


def atomic_number(symbol_or_z) -> int:
    if isinstance(symbol_or_z, int):
        return symbol_or_z
    s = str(symbol_or_z).strip()
    if s.isdigit():
        return int(s)
    try:
        return SYMBOLS.index(s.capitalize()) + 1
    except ValueError:
        raise ValueError(f"unknown element {symbol_or_z!r}") from None


def expand_core(core: str) -> list[str]:
    """``'[Kr]4d10'`` / ``'[Kr] 4d'`` / ``'1s 2s 2p'`` -> ``['1s', '2s', ..., '4d']``.

    Occupation numbers in the core are ignored (core shells are closed).
    """
    core = core.strip()
    shells: list[str] = []
    if core.startswith("["):
        gas, rest = core[1:].split("]", 1)
        if gas in NOBLE_CORES:
            shells += NOBLE_CORES[gas].split()
        else:                      # dbsr_hf style '[4d]' = everything up to 4d
            n, l = int(gas[:-1]), L_SYMBOLS.index(gas[-1])
            order = NOBLE_CORES["Rn"].split() + ["5f", "6d", "7s", "7p"]
            shells += order[:order.index(f"{n}{L_SYMBOLS[l]}") + 1]
        core = rest
    if core.strip():
        for n, l, _ in parse_config(core):
            s = f"{n}{L_SYMBOLS[l]}"
            if s not in shells:
                shells.append(s)
    return shells


def core_electrons(shells: list[str]) -> int:
    return sum(2 * (2 * L_SYMBOLS.index(s[-1]) + 1) for s in shells)


@dataclass(frozen=True)
class Ion:
    """Target atom or ion, e.g. ``Ion('Xe', 1)`` = Xe+ = Xe II."""
    element: str | int
    charge: int = 0

    @property
    def z(self) -> int:
        return atomic_number(self.element)

    @property
    def symbol(self) -> str:
        return SYMBOLS[self.z - 1]

    @property
    def nelc(self) -> int:
        return self.z - self.charge

    @property
    def spectrum(self) -> str:
        """NIST spectrum name, e.g. ``'Xe II'``."""
        return f"{self.symbol} {ROMAN[self.charge]}"

    @property
    def name(self) -> str:
        return self.symbol + ("" if self.charge == 0 else (f"{self.charge}+" if self.charge > 1 else "+"))

    def __str__(self):
        return f"{self.name} (Z={self.z}, N={self.nelc})"
