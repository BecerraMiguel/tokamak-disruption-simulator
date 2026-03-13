"""
ITER machine definition for FreeGSNKE.

Parses the DINA-IMAS tokamak_config.dat file and converts it to the
FreeGSNKE-compatible data structures: active_coils, limiter, wall.

ITER parameters (reference, 15 MA DT scenario):
  R_major = 6.2 m, a_minor = 2.0 m, B0 = 5.3 T
  Ip_max = 15 MA, q95 ~ 3.0, kappa ~ 1.7, delta ~ 0.33

Coil layout (from tokamak_config.dat):
  CS: CS3U, CS2U, CS1U, CS1L, CS2L, CS3L  (central solenoid, 6 modules)
  PF: PF1, PF2, PF3, PF4, PF5, PF6        (poloidal field coils)
  VS: VS3U, VS3L                            (vertical stability coils)

Each coil is treated as an independent circuit (allows prescribing any
current waveform individually). Passive structures (vessel eddy currents)
are omitted for Phase 1 — they matter for fast transients (VDEs) which
are handled by DREAM, not FreeGSNKE.
"""

from __future__ import annotations

import os
import numpy as np

# --- ITER physical parameters (public design values) -------------------------

ITER_PARAMS = {
    "R_major": 6.2,     # m, major radius
    "a_minor": 2.0,     # m, minor radius
    "B0": 5.3,          # T, vacuum toroidal field on axis
    "Ip_15MA": 15.0e6,  # A, full-current DT plasma
    "kappa": 1.75,      # plasma elongation
    "delta": 0.33,      # plasma triangularity
    "q95_ref": 3.0,     # reference safety factor at 95% flux surface
    "fvac": 5.3 * 6.2,  # T·m, F_vac = B0 * R_major
}

# Path to the DINA-IMAS tokamak config
_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..",
    "DINA-IMAS", "machines", "iter", "tokamak_config.dat",
)
_DEFAULT_CONFIG_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../../..","../DINA-IMAS/machines/iter/tokamak_config.dat")
)

# Resistivity values (Ω·m)
# SC coils have effectively zero DC resistance; use a small value for numerics
_SC_RESISTIVITY = 1.0e-10   # superconducting (CS, PF coils)
_VS_RESISTIVITY = 7.0e-8    # resistive copper-like (VS3 coils)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_tokamak_config(config_path: str) -> dict:
    """
    Parse DINA-IMAS tokamak_config.dat and return raw coil + limiter data.

    Returns
    -------
    data : dict with keys:
        "coils"   : list of coil dicts (name, R, Z, dR, dZ, n_strands, n_turns, polarity, circuit)
        "limiter" : list of (R, Z) tuples — first wall contour
        "domain"  : (R_min, R_max, Z_min, Z_max) — computational domain
    """
    coils = []
    limiter_pts = []
    domain = None

    with open(config_path) as f:
        lines = [l.strip() for l in f.readlines()]

    i = 0
    while i < len(lines):
        line = lines[i]

        # ---- COILS section --------------------------------------------------
        if line.startswith("COILS"):
            i += 1
            n_coils = int(lines[i])
            i += 1
            for _ in range(n_coils):
                name = lines[i]
                i += 1
                toks = lines[i].split()
                n_strands = int(toks[0])
                n_turns   = int(toks[1])
                polarity  = int(toks[2])
                circuit   = int(toks[3])
                i += 1
                toks2 = lines[i].split()
                R, Z, dR, dZ = float(toks2[0]), float(toks2[1]), float(toks2[2]), float(toks2[3])
                i += 1
                coils.append({
                    "name": name,
                    "R": R, "Z": Z,
                    "dR": dR, "dZ": dZ,
                    "n_strands": n_strands,
                    "n_turns": n_turns,
                    "polarity": polarity,
                    "circuit": circuit,
                })

        # ---- LIMITER section ------------------------------------------------
        elif line.startswith("Limiter"):
            i += 1
            n_lim = int(lines[i])
            i += 1
            for _ in range(n_lim):
                toks = lines[i].split()
                limiter_pts.append((float(toks[0]), float(toks[1])))
                i += 1

        # ---- Computational domain -------------------------------------------
        elif line.startswith("area"):
            i += 1
            r_range = [float(x) for x in lines[i].split()]
            i += 1
            z_range = [float(x) for x in lines[i].split()]
            domain = (r_range[0], r_range[1], z_range[0], z_range[1])
            i += 1

        else:
            i += 1

    return {"coils": coils, "limiter": limiter_pts, "domain": domain}


# ---------------------------------------------------------------------------
# Machine builder
# ---------------------------------------------------------------------------

def _build_active_coils(raw_coils: list) -> dict:
    """
    Convert parsed coil list to FreeGSNKE active_coils dict.

    Each coil is represented by a single centre filament.
    The total ampere-turns are encoded via the multiplier:
        multiplier = n_strands * n_turns

    For coils that share a circuit index (CS1U/CS1L, VS3U/VS3L),
    they are grouped into a FreeGSNKE sub-dict (up-down circuit).
    """
    # Group coils by circuit index to find shared circuits
    from collections import defaultdict
    circuit_groups: dict[int, list] = defaultdict(list)
    for coil in raw_coils:
        circuit_groups[coil["circuit"]].append(coil)

    active_coils = {}

    for circuit_idx, members in circuit_groups.items():
        if len(members) == 1:
            # Independent single coil
            c = members[0]
            n_tot = c["n_strands"] * c["n_turns"]
            resistivity = _VS_RESISTIVITY if "VS" in c["name"] else _SC_RESISTIVITY
            active_coils[c["name"]] = {
                "R": [c["R"]],
                "Z": [c["Z"]],
                "dR": c["dR"],
                "dZ": c["dZ"],
                "resistivity": resistivity,
                "polarity": float(c["polarity"]),
                "multiplier": float(n_tot),
            }
        else:
            # Linked circuit (e.g. CS1 = CS1U + CS1L, VS3 = VS3U + VS3L)
            # Name the circuit after the common prefix
            names = [m["name"] for m in members]
            # Find common prefix (e.g. "CS1" from "CS1U"/"CS1L")
            prefix = _common_prefix(names)
            circuit_name = prefix if prefix else f"CIRCUIT_{circuit_idx}"

            sub = {}
            for m in members:
                sub_key = m["name"][len(prefix):]  # e.g. "U" or "L"
                if not sub_key:
                    sub_key = m["name"]
                n_tot = m["n_strands"] * m["n_turns"]
                resistivity = _VS_RESISTIVITY if "VS" in m["name"] else _SC_RESISTIVITY
                sub[sub_key] = {
                    "R": [m["R"]],
                    "Z": [m["Z"]],
                    "dR": m["dR"],
                    "dZ": m["dZ"],
                    "resistivity": resistivity,
                    "polarity": float(m["polarity"]),
                    "multiplier": float(n_tot),
                }
            active_coils[circuit_name] = sub

    return active_coils


def _common_prefix(strings: list[str]) -> str:
    if not strings:
        return ""
    prefix = strings[0]
    for s in strings[1:]:
        while not s.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    return prefix


def _build_limiter(limiter_pts: list) -> list:
    """Convert (R, Z) tuples to FreeGSNKE limiter format.

    Ensures the contour is closed by appending the first point at the end
    if it is not already closed. FreeGSNKE's matplotlib Path.contains_points
    requires a closed polygon for correct inside/outside determination.
    """
    import math
    pts = list(limiter_pts)
    if pts:
        r0, z0 = pts[0]
        r1, z1 = pts[-1]
        # Close if not already closed
        if math.hypot(r1 - r0, z1 - z0) > 1e-6:
            pts = pts + [(r0, z0)]
    return [{"R": r, "Z": z} for r, z in pts]


def build_iter_machine(
    config_path: str | None = None,
    include_passives: bool = False,
    verbose: bool = True,
) -> object:
    """
    Build a FreeGSNKE machine object representing ITER.

    Parameters
    ----------
    config_path : str, optional
        Path to tokamak_config.dat. Defaults to the DINA-IMAS repo location.
    include_passives : bool
        If True, include simplified vessel passive structures for eddy current
        modelling. Default False (passives are needed for fast transients only).
    verbose : bool
        Print build progress.

    Returns
    -------
    tokamak : freegsnke.machine_update.Machine
        FreeGSNKE machine object with R/M matrices computed.
    active_coils : dict
        Raw active coil data (for reference/voltage control).
    domain : tuple
        (R_min, R_max, Z_min, Z_max) computational domain.
    """
    import freegsnke.build_machine as bm

    if config_path is None:
        config_path = _resolve_config_path()

    if verbose:
        print(f"Parsing ITER machine config: {config_path}")

    raw = parse_tokamak_config(config_path)
    active_coils = _build_active_coils(raw["coils"])
    limiter_data = _build_limiter(raw["limiter"])
    wall_data    = limiter_data          # use limiter as computational boundary

    if verbose:
        print(f"  Active coils/circuits: {list(active_coils.keys())}")
        print(f"  Limiter points: {len(limiter_data)}")

    tokamak = bm.tokamak(
        active_coils_data=active_coils,
        passive_coils_data=[] if not include_passives else None,
        limiter_data=limiter_data,
        wall_data=wall_data,
    )

    return tokamak, active_coils, raw["domain"]


def _resolve_config_path() -> str:
    """
    Search for tokamak_config.dat relative to this file's location.
    Looks in:
      1. The DINA-IMAS clone next to the project root
      2. The configs/ directory
    """
    # Option 1: DINA-IMAS clone at ../DINA-IMAS (relative to project root)
    here = os.path.dirname(__file__)
    project_root = os.path.normpath(os.path.join(here, "../.."))
    candidate1 = os.path.normpath(
        os.path.join(project_root, "../DINA-IMAS/machines/iter/tokamak_config.dat")
    )
    if os.path.isfile(candidate1):
        return candidate1

    # Option 2: copied into configs/
    candidate2 = os.path.join(project_root, "configs", "iter", "tokamak_config.dat")
    if os.path.isfile(candidate2):
        return candidate2

    raise FileNotFoundError(
        "Cannot find tokamak_config.dat. Either:\n"
        f"  (a) ensure DINA-IMAS is cloned at {candidate1}\n"
        f"  (b) copy it to {candidate2}"
    )


# ---------------------------------------------------------------------------
# Convenience: initial coil current estimates for a 15 MA reference equilibrium
# ---------------------------------------------------------------------------

def reference_coil_currents_15MA() -> dict:
    """
    Return FreeGSNKE circuit currents (A) for ITER 15 MA DT flat-top.

    Derived from the DINA-IMAS 15MA_40ka scenario scr_data.dat at t≈75s.
    The physical coil currents (in kA) from that file are converted to
    FreeGSNKE's circuit.current convention:

        circuit.current = I_physical_kA × 1000 × n_physical_turns / multiplier

    where multiplier = n_sim_strands × n_sim_turns from tokamak_config.dat and
    n_physical_turns comes from DINA control_init_1.dat.

    Physical currents at flat-top (t≈75s), in kA:
        CS3U: +4.0,  CS2U: -10.8,  CS1: -21.64,  CS2L: -7.708,  CS3L: +5.5
        PF1: +5.021, PF2: -3.13,   PF3: -5.027,   PF4: -4.272
        PF5: -7.831, PF6: +17.334, VS3: 0.0

    n_physical_turns:
        CS*/CS*: 554 per module, PF1: 248.6, PF2: 115.2, PF3: 185.9,
        PF4: 169.9, PF5: 216.8, PF6: 459.4

    Returns dict: {circuit_name: circuit_current_A}
    """
    # circuit.current = I_phys_kA * 1e3 * n_phys_turns / multiplier
    return {
        "CS3U":   +4.0e3 * 554 / 90,    # +24,622 A
        "CS2U":  -10.8e3 * 554 / 90,    # -66,480 A
        "CS1":   -21.64e3 * 554 / 180,  # -66,594 A  (CS1U+CS1L, 180 mult each)
        "CS2L":   -7.708e3 * 554 / 90,  # -47,438 A
        "CS3L":   +5.5e3 * 554 / 90,    # +33,822 A
        "PF1":    +5.021e3 * 248.6 / 100,   # +12,482 A
        "PF2":    -3.13e3 * 115.2 / 50,     #  -7,211 A
        "PF3":    -5.027e3 * 185.9 / 100,   #  -9,345 A
        "PF4":    -4.272e3 * 169.9 / 100,   #  -7,258 A
        "PF5":    -7.831e3 * 216.8 / 100,   # -16,977 A
        "PF6":   +17.334e3 * 459.4 / 800,   #  +9,950 A
        "VS3":    0.0,
    }
