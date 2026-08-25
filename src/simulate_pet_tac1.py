"""
simulate_pet_tac1.py

Generate realistic dynamic PET time-activity curves (TACs)
using a two-tissue compartment model (2TCM).

Project structure expected:

PET/
├── data/
├── figures/
└── src/
    └── simulate_pet_tac1.py

The script always saves output to PET/data/,
independent of the current working directory.
"""

# =====================================================================
# IMPORTS
# =====================================================================

from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp


# =====================================================================
# PROJECT PATHS
# =====================================================================

# Location of this script:
# PET/src/simulate_pet_tac1.py
SRC_DIR = Path(__file__).resolve().parent

# Project root:
# PET/
PROJECT_DIR = SRC_DIR.parent

# Standard project folders
DATA_DIR = PROJECT_DIR / "data"
FIGURES_DIR = PROJECT_DIR / "figures"

# Create folders if they do not exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Output file
OUTPUT_FILE = DATA_DIR / "simulated_pet_tacs.npz"


# =====================================================================
# 1. ARTERIAL INPUT FUNCTION
# =====================================================================

def arterial_input_function(
    t,
    A1=800.0,
    A2=200.0,
    lambda1=4.0,
    lambda2=0.3,
    t0=0.05,
):
    """
    Bi-exponential arterial plasma input function.

    Parameters
    ----------
    t : float or ndarray
        Time in minutes.

    A1, A2 : float
        Amplitudes.

    lambda1, lambda2 : float
        Decay constants (1/min).

    t0 : float
        Injection delay in minutes.

    Returns
    -------
    Cp : float or ndarray
        Plasma activity.
    """

    t = np.asarray(t, dtype=float)

    tau = np.maximum(t - t0, 0.0)

    Cp = (
        A1 * np.exp(-lambda1 * tau)
        + A2 * np.exp(-lambda2 * tau)
    )

    # No tracer before injection
    Cp = np.where(t < t0, 0.0, Cp)

    return Cp


# =====================================================================
# 2. TWO-TISSUE COMPARTMENT MODEL
# =====================================================================

def two_tissue_ode(t, y, K1, k2, k3, k4):
    """
    Standard reversible 2-tissue compartment model.

    dC1/dt = K1*Cp - (k2+k3)*C1 + k4*C2
    dC2/dt = k3*C1 - k4*C2
    """

    C1, C2 = y

    Cp = arterial_input_function(t)

    dC1_dt = (
        K1 * Cp
        - (k2 + k3) * C1
        + k4 * C2
    )

    dC2_dt = (
        k3 * C1
        - k4 * C2
    )

    return [dC1_dt, dC2_dt]


# =====================================================================
# 3. SIMULATE ONE TISSUE TAC
# =====================================================================

def simulate_tac(
    frame_edges,
    K1=0.25,
    k2=0.20,
    k3=0.10,
    k4=0.02,
    vB=0.05,
    noise_std_frac=0.08,
    seed=None,
):
    """
    Simulate one dynamic PET tissue TAC.

    Returns
    -------
    frame_midtimes : ndarray
        PET frame mid-times.

    Cp : ndarray
        Plasma activity at frame mid-times.

    Ct_true : ndarray
        Noise-free tissue TAC.

    Ct_noisy : ndarray
        Noise-corrupted tissue TAC.
    """

    frame_edges = np.asarray(frame_edges, dtype=float)

    # -------------------------------------------------------------
    # PET frame mid-times
    # -------------------------------------------------------------

    frame_midtimes = (
        frame_edges[:-1] + frame_edges[1:]
    ) / 2.0

    # -------------------------------------------------------------
    # Fine time grid for ODE integration
    # -------------------------------------------------------------

    t_fine = np.linspace(
        frame_edges[0],
        frame_edges[-1],
        10000,
    )

    # -------------------------------------------------------------
    # Solve 2TCM
    # -------------------------------------------------------------

    solution = solve_ivp(
        two_tissue_ode,
        (
            frame_edges[0],
            frame_edges[-1],
        ),
        y0=[0.0, 0.0],
        t_eval=t_fine,
        args=(K1, k2, k3, k4),
        method="LSODA",
        rtol=1e-7,
        atol=1e-9,
    )

    if not solution.success:
        raise RuntimeError(
            f"ODE integration failed: {solution.message}"
        )

    C1_fine = solution.y[0]
    C2_fine = solution.y[1]

    Cp_fine = arterial_input_function(t_fine)

    # -------------------------------------------------------------
    # Total tissue activity
    # -------------------------------------------------------------

    Ct_fine = (
        (1.0 - vB) * (C1_fine + C2_fine)
        + vB * Cp_fine
    )

    # -------------------------------------------------------------
    # PET frame averaging
    # -------------------------------------------------------------

    Ct_true = np.zeros(len(frame_midtimes))

    for i in range(len(frame_midtimes)):

        t_start = frame_edges[i]
        t_end = frame_edges[i + 1]

        mask = (
            (t_fine >= t_start)
            & (t_fine <= t_end)
        )

        if np.sum(mask) < 2:
            raise RuntimeError(
                f"Insufficient integration points "
                f"for frame {i}."
            )

        # Trapezoidal frame average
        Ct_true[i] = (
            np.trapz(
                Ct_fine[mask],
                t_fine[mask],
            )
            / (t_end - t_start)
        )

    # -------------------------------------------------------------
    # Plasma activity at frame mid-times
    # -------------------------------------------------------------

    Cp = arterial_input_function(frame_midtimes)

    # -------------------------------------------------------------
    # PET-like signal-dependent noise
    # -------------------------------------------------------------

    rng = np.random.default_rng(seed)

    noise_sigma = (
        noise_std_frac
        * np.sqrt(np.maximum(Ct_true, 1e-6))
    )

    noise = rng.normal(
        loc=0.0,
        scale=noise_sigma,
        size=Ct_true.shape,
    )

    Ct_noisy = Ct_true + noise

    # PET activity cannot be negative
    Ct_noisy = np.maximum(Ct_noisy, 0.0)

    return (
        frame_midtimes,
        Cp,
        Ct_true,
        Ct_noisy,
    )


# =====================================================================
# 4. SIMULATE MULTIPLE BRAIN REGIONS
# =====================================================================

def simulate_multi_region(
    frame_edges,
    region_params,
    noise_std_frac=0.08,
    seed=0,
):
    """
    Simulate several tissue regions with different
    kinetic properties.
    """

    results = {}

    Cp = None
    frame_midtimes = None

    for i, (region_name, params) in enumerate(
        region_params.items()
    ):

        (
            frame_midtimes,
            Cp,
            Ct_true,
            Ct_noisy,
        ) = simulate_tac(
            frame_edges=frame_edges,
            noise_std_frac=noise_std_frac,
            seed=seed + i,
            **params,
        )

        results[region_name] = {
            "true": Ct_true,
            "noisy": Ct_noisy,
            "K1": params["K1"],
            "k2": params["k2"],
            "k3": params["k3"],
            "k4": params["k4"],
            "vB": params["vB"],
        }

    return results, Cp, frame_midtimes


# =====================================================================
# 5. MAIN SIMULATION
# =====================================================================

if __name__ == "__main__":

    print("=" * 80)
    print("DYNAMIC PET TAC SIMULATION")
    print("=" * 80)

    print("\nProject directory:")
    print(PROJECT_DIR)

    print("\nData directory:")
    print(DATA_DIR)

    print("\nOutput file:")
    print(OUTPUT_FILE)

    # -------------------------------------------------------------
    # Dynamic PET frame schedule
    #
    # 0-2 min   : 8 x 15 s
    # 2-10 min  : 8 x 60 s
    # 10-60 min : 10 x 5 min
    # -------------------------------------------------------------

    early_edges = np.arange(
        0.0,
        2.0 + 0.25,
        0.25,
    )

    middle_edges = np.arange(
        2.0,
        10.0 + 1.0,
        1.0,
    )

    late_edges = np.arange(
        10.0,
        60.0 + 5.0,
        5.0,
    )

    # Avoid repeating boundaries
    frame_edges = np.concatenate([
        early_edges,
        middle_edges[1:],
        late_edges[1:],
    ])

    # -------------------------------------------------------------
    # Define tissue kinetic parameters
    # -------------------------------------------------------------

    regions = {

        "grey_matter": {
            "K1": 0.30,
            "k2": 0.25,
            "k3": 0.12,
            "k4": 0.02,
            "vB": 0.05,
        },

        "white_matter": {
            "K1": 0.15,
            "k2": 0.20,
            "k3": 0.05,
            "k4": 0.02,
            "vB": 0.03,
        },

        "lesion": {
            "K1": 0.35,
            "k2": 0.15,
            "k3": 0.20,
            "k4": 0.01,
            "vB": 0.06,
        },
    }

    # -------------------------------------------------------------
    # Run simulation
    # -------------------------------------------------------------

    results, Cp, frame_midtimes = simulate_multi_region(
        frame_edges=frame_edges,
        region_params=regions,
        noise_std_frac=0.08,
        seed=42,
    )

    # -------------------------------------------------------------
    # Prepare data for saving
    # -------------------------------------------------------------

    save_dict = {
        "frame_edges": frame_edges,
        "frame_midtimes": frame_midtimes,
        "Cp": Cp,
    }

    for region_name, data in results.items():

        save_dict[
            f"{region_name}_true"
        ] = data["true"]

        save_dict[
            f"{region_name}_noisy"
        ] = data["noisy"]

        save_dict[
            f"{region_name}_K1"
        ] = data["K1"]

        save_dict[
            f"{region_name}_k2"
        ] = data["k2"]

        save_dict[
            f"{region_name}_k3"
        ] = data["k3"]

        save_dict[
            f"{region_name}_k4"
        ] = data["k4"]

        save_dict[
            f"{region_name}_vB"
        ] = data["vB"]

    # -------------------------------------------------------------
    # Save
    # -------------------------------------------------------------

    np.savez(
        OUTPUT_FILE,
        **save_dict,
    )

    # -------------------------------------------------------------
    # Verify output
    # -------------------------------------------------------------

    if not OUTPUT_FILE.exists():
        raise RuntimeError(
            "Output file was not created."
        )

    # -------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------

    print("\n" + "=" * 80)
    print("DYNAMIC PET SIMULATION COMPLETED")
    print("=" * 80)

    print(
        f"\nNumber of PET frames: "
        f"{len(frame_midtimes)}"
    )

    print(
        f"Time range: "
        f"{frame_edges[0]:.3f} - "
        f"{frame_edges[-1]:.3f} min"
    )

    print("\nRegions:")

    for region_name, params in regions.items():

        print(
            f"  {region_name:15s} "
            f"K1={params['K1']:.3f}, "
            f"k2={params['k2']:.3f}, "
            f"k3={params['k3']:.3f}, "
            f"k4={params['k4']:.3f}, "
            f"vB={params['vB']:.3f}"
        )

    print("\nSaved successfully to:")
    print(OUTPUT_FILE)

    print("\nOutput file exists:")
    print(OUTPUT_FILE.exists())

    print("\n" + "=" * 80)
