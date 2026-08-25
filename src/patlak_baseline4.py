"""
patlak_baseline.py

Standard Patlak graphical analysis for dynamic PET TACs.

This provides a conventional kinetic-analysis baseline for comparison
with the model-independent Hankel-DMD analysis.

Patlak formulation:

    y(t) = Ct(t) / Cp(t)

    x(t) = integral_0^t Cp(tau) dtau / Cp(t)

For sufficiently late times, if the tracer behaves approximately as
an irreversible compartment, the Patlak plot becomes linear:

    Ct/Cp = Ki * integral(Cp)/Cp + V

where:

    Ki = Patlak slope (net influx rate)
    V  = intercept

The current simulated PET dataset contains:
    - frame_midtimes
    - Cp
    - grey_matter_true/noisy
    - white_matter_true/noisy
    - lesion_true/noisy

The analysis is performed using the original 26 PET frames.
"""

import numpy as np
import matplotlib.pyplot as plt
import os


# ==============================================================
# PATHS
# ==============================================================

DATA_FILE = r"C:\Users\Acer\Desktop\PET\data\simulated_pet_tacs.npz"

OUTPUT_DATA = (
    r"C:\Users\Acer\Desktop\PET\data\patlak_results.npz"
)

OUTPUT_FIGURE = (
    r"C:\Users\Acer\Desktop\PET\figures\patlak_analysis.png"
)


# ==============================================================
# USER SETTINGS
# ==============================================================

# Patlak linear region starts approximately at this time.
# You can later test 8, 10, 12, 15 min, etc.
T_STAR = 10.0

REGIONS = [
    "grey_matter",
    "white_matter",
    "lesion"
]


# ==============================================================
# PATLAK FUNCTION
# ==============================================================

def patlak_analysis(t, Ct, Cp, t_star):
    """
    Perform Patlak graphical analysis.

    Parameters
    ----------
    t : array
        PET frame mid-times (min)

    Ct : array
        Tissue TAC

    Cp : array
        Plasma/input-function TAC

    t_star : float
        Time after which the Patlak plot is fitted.

    Returns
    -------
    Ki : float
        Patlak slope / net influx rate

    intercept : float
        Patlak intercept

    R2 : float
        Coefficient of determination of the linear fit

    x : array
        Integral(Cp)/Cp

    y : array
        Ct/Cp

    fit_y : array
        Linear fitted values
    """

    t = np.asarray(t, dtype=float)
    Ct = np.asarray(Ct, dtype=float)
    Cp = np.asarray(Cp, dtype=float)

    # ----------------------------------------------------------
    # Basic checks
    # ----------------------------------------------------------

    if not (len(t) == len(Ct) == len(Cp)):
        raise ValueError(
            "t, Ct and Cp must have the same length."
        )

    if np.any(Cp <= 0):
        raise ValueError(
            "Cp contains zero or negative values. "
            "Patlak calculation requires positive Cp."
        )

    # ----------------------------------------------------------
    # Cumulative plasma integral
    #
    # Trapezoidal integration:
    #
    # integral_0^t Cp(tau) dtau
    # ----------------------------------------------------------

    cumulative_Cp = np.zeros_like(Cp)

    for i in range(1, len(t)):
        cumulative_Cp[i] = (
            cumulative_Cp[i - 1]
            + 0.5
            * (Cp[i] + Cp[i - 1])
            * (t[i] - t[i - 1])
        )

    # ----------------------------------------------------------
    # Patlak coordinates
    # ----------------------------------------------------------

    x_all = cumulative_Cp / Cp
    y_all = Ct / Cp

    # ----------------------------------------------------------
    # Select linear region
    # ----------------------------------------------------------

    mask = t >= t_star

    if np.sum(mask) < 3:
        raise ValueError(
            "Too few points after t_star for linear regression."
        )

    x = x_all[mask]
    y = y_all[mask]
    t_fit = t[mask]

    # ----------------------------------------------------------
    # Linear regression
    #
    # y = Ki*x + intercept
    # ----------------------------------------------------------

    A = np.vstack([
        x,
        np.ones_like(x)
    ]).T

    Ki, intercept = np.linalg.lstsq(
        A,
        y,
        rcond=None
    )[0]

    # ----------------------------------------------------------
    # Calculate R^2
    # ----------------------------------------------------------

    fit_y = Ki * x + intercept

    residuals = y - fit_y

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    if ss_tot > 0:
        R2 = 1.0 - ss_res / ss_tot
    else:
        R2 = np.nan

    return (
        Ki,
        intercept,
        R2,
        x,
        y,
        fit_y,
        t_fit,
        x_all,
        y_all
    )


# ==============================================================
# MAIN PROGRAM
# ==============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("PATLAK GRAPHICAL ANALYSIS OF DYNAMIC PET TACs")
    print("=" * 70)

    # ----------------------------------------------------------
    # Load data
    # ----------------------------------------------------------

    print("\nLoading:")
    print(DATA_FILE)

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"\nInput file not found:\n{DATA_FILE}"
        )

    data = np.load(DATA_FILE)

    print("\nAvailable arrays:")
    for key in data.files:
        print(f"    {key}")

    # ----------------------------------------------------------
    # Load time and plasma input function
    # ----------------------------------------------------------

    t = data["frame_midtimes"]
    Cp = data["Cp"]

    print("\n" + "-" * 70)
    print("PET INPUT FUNCTION")
    print("-" * 70)

    print(f"Number of PET frames : {len(t)}")
    print(
        f"Time range           : "
        f"{t[0]:.3f} - {t[-1]:.3f} min"
    )

    print(f"Cp minimum           : {np.min(Cp):.6f}")
    print(f"Cp maximum           : {np.max(Cp):.6f}")

    print(f"\nPatlak t*            : {T_STAR:.2f} min")

    # ----------------------------------------------------------
    # Find t* index
    # ----------------------------------------------------------

    t_star_idx = np.searchsorted(t, T_STAR)

    print(
        f"Patlak fitting starts at frame index : "
        f"{t_star_idx}"
    )

    print(
        f"Actual fitting start time             : "
        f"{t[t_star_idx]:.3f} min"
    )

    # ----------------------------------------------------------
    # Prepare results
    # ----------------------------------------------------------

    results = {}

    # ----------------------------------------------------------
    # Figure
    # ----------------------------------------------------------

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10, 14),
        sharex=False
    )

    # ----------------------------------------------------------
    # Analyze each region
    # ----------------------------------------------------------

    for ax, region in zip(axes, REGIONS):

        print("\n")
        print("-" * 70)
        print(region.upper())
        print("-" * 70)

        Ct_noisy = data[f"{region}_noisy"]
        Ct_true = data[f"{region}_true"]

        # ------------------------------------------------------
        # Noisy TAC
        # ------------------------------------------------------

        (
            Ki_noisy,
            intercept_noisy,
            R2_noisy,
            x_noisy,
            y_noisy,
            fit_noisy,
            t_fit,
            x_all_noisy,
            y_all_noisy
        ) = patlak_analysis(
            t,
            Ct_noisy,
            Cp,
            T_STAR
        )

        # ------------------------------------------------------
        # True TAC
        # ------------------------------------------------------

        (
            Ki_true,
            intercept_true,
            R2_true,
            x_true,
            y_true,
            fit_true,
            _,
            x_all_true,
            y_all_true
        ) = patlak_analysis(
            t,
            Ct_true,
            Cp,
            T_STAR
        )

        # ------------------------------------------------------
        # Print results
        # ------------------------------------------------------

        print("\nNOISY TAC")
        print(f"Patlak Ki       : {Ki_noisy:.6f} 1/min")
        print(f"Intercept       : {intercept_noisy:.6f}")
        print(f"R²              : {R2_noisy:.6f}")

        print("\nTRUE TAC")
        print(f"Patlak Ki       : {Ki_true:.6f} 1/min")
        print(f"Intercept       : {intercept_true:.6f}")
        print(f"R²              : {R2_true:.6f}")

        # ------------------------------------------------------
        # Save numerical results
        # ------------------------------------------------------

        results[f"{region}_Ki_noisy"] = Ki_noisy
        results[f"{region}_intercept_noisy"] = intercept_noisy
        results[f"{region}_R2_noisy"] = R2_noisy

        results[f"{region}_Ki_true"] = Ki_true
        results[f"{region}_intercept_true"] = intercept_true
        results[f"{region}_R2_true"] = R2_true

        results[f"{region}_x_noisy"] = x_noisy
        results[f"{region}_y_noisy"] = y_noisy
        results[f"{region}_fit_noisy"] = fit_noisy

        results[f"{region}_x_true"] = x_true
        results[f"{region}_y_true"] = y_true
        results[f"{region}_fit_true"] = fit_true

        # ------------------------------------------------------
        # Plot noisy Patlak points
        # ------------------------------------------------------

        ax.scatter(
            x_noisy,
            y_noisy,
            s=35,
            label="Noisy TAC"
        )

        # ------------------------------------------------------
        # Plot noisy linear regression
        # ------------------------------------------------------

        ax.plot(
            x_noisy,
            fit_noisy,
            linewidth=2,
            label=(
                f"Patlak fit "
                f"(Ki={Ki_noisy:.4f} min$^{{-1}})"
            )
        )

        # ------------------------------------------------------
        # Plot true data
        # ------------------------------------------------------

        ax.plot(
            x_true,
            y_true,
            "--",
            linewidth=2,
            label="True TAC"
        )

        ax.set_title(
            region.replace("_", " ").title()
        )

        ax.set_xlabel(
            r"$\int_0^t C_p(\tau)d\tau / C_p(t)$ (min)"
        )

        ax.set_ylabel(
            r"$C_t(t)/C_p(t)$"
        )

        ax.grid(True, alpha=0.3)

        ax.legend()

        # ------------------------------------------------------
        # Add R² annotation
        # ------------------------------------------------------

        ax.text(
            0.03,
            0.95,
            f"t* = {t[t_star_idx]:.2f} min\n"
            f"Ki = {Ki_noisy:.4f} min$^{{-1}}$\n"
            f"R² = {R2_noisy:.4f}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(
                boxstyle="round",
                alpha=0.15
            )
        )

    # ----------------------------------------------------------
    # Figure title
    # ----------------------------------------------------------

    fig.suptitle(
        "Patlak Graphical Analysis of Dynamic PET TACs",
        fontsize=16
    )

    plt.tight_layout()

    # ----------------------------------------------------------
    # Save figure
    # ----------------------------------------------------------

    os.makedirs(
        os.path.dirname(OUTPUT_FIGURE),
        exist_ok=True
    )

    plt.savefig(
        OUTPUT_FIGURE,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    # ----------------------------------------------------------
    # Save numerical results
    # ----------------------------------------------------------

    results["t"] = t
    results["Cp"] = Cp
    results["t_star"] = T_STAR
    results["t_star_index"] = t_star_idx

    os.makedirs(
        os.path.dirname(OUTPUT_DATA),
        exist_ok=True
    )

    np.savez(
        OUTPUT_DATA,
        **results
    )

    # ----------------------------------------------------------
    # Final summary
    # ----------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("PATLAK ANALYSIS COMPLETED")
    print("=" * 70)

    for region in REGIONS:

        Ki = results[f"{region}_Ki_noisy"]
        R2 = results[f"{region}_R2_noisy"]

        print(
            f"\n{region.upper()}"
        )

        print(
            f"  Patlak Ki = {Ki:.6f} 1/min"
        )

        print(
            f"  R²        = {R2:.6f}"
        )

    print("\nResults saved to:")
    print(OUTPUT_DATA)

    print("\nFigure saved to:")
    print(OUTPUT_FIGURE)

    print("\n" + "=" * 70)
