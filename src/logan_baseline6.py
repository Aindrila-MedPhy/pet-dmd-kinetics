"""
logan_baseline.py

Standard Logan graphical analysis for reversible dynamic PET.

Logan plot:
    y(t) = integral_0^t Ct(tau) dtau / Ct(t)

    x(t) = integral_0^t Cp(tau) dtau / Ct(t)

For sufficiently late times, the relationship becomes approximately
linear:

    y = VT * x + intercept

The slope provides an estimate of the total distribution volume (VT).

This provides an independent graphical baseline for comparison with
Hankel-DMD and the 2TCM analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import os


# ============================================================
# Logan analysis
# ============================================================

def cumulative_trapezoid(y, t):
    """
    Cumulative trapezoidal integration without scipy.
    """
    integral = np.zeros_like(y, dtype=float)

    integral[1:] = np.cumsum(
        0.5 * (y[1:] + y[:-1]) * np.diff(t)
    )

    return integral


def logan_analysis(t, Ct, Cp, t_star):
    """
    Perform Logan graphical analysis.

    Parameters
    ----------
    t : array
        PET frame mid-times (min)

    Ct : array
        Tissue TAC

    Cp : array
        Plasma input function

    t_star : float
        Time after which Logan plot is assumed linear.

    Returns
    -------
    VT
        Logan slope

    intercept
        Linear-fit intercept

    R2
        Coefficient of determination

    x
        Logan x-axis

    y
        Logan y-axis

    fit_mask
        Boolean mask for fitted points
    """

    # --------------------------------------------------------
    # Cumulative integrals
    # --------------------------------------------------------

    int_Cp = cumulative_trapezoid(Cp, t)
    int_Ct = cumulative_trapezoid(Ct, t)

    # --------------------------------------------------------
    # Avoid division by zero
    # --------------------------------------------------------

    valid = Ct > 1e-12

    x_all = np.full_like(t, np.nan, dtype=float)
    y_all = np.full_like(t, np.nan, dtype=float)

    x_all[valid] = int_Cp[valid] / Ct[valid]
    y_all[valid] = int_Ct[valid] / Ct[valid]

    # --------------------------------------------------------
    # Select late-time linear region
    # --------------------------------------------------------

    fit_mask = (
        valid &
        (t >= t_star) &
        np.isfinite(x_all) &
        np.isfinite(y_all)
    )

    x = x_all[fit_mask]
    y = y_all[fit_mask]

    if len(x) < 3:
        raise ValueError(
            "Not enough points for Logan linear regression."
        )

    # --------------------------------------------------------
    # Linear regression
    # --------------------------------------------------------

    A = np.vstack([x, np.ones_like(x)]).T

    VT, intercept = np.linalg.lstsq(
        A, y, rcond=None
    )[0]

    # --------------------------------------------------------
    # R-squared
    # --------------------------------------------------------

    y_pred = VT * x + intercept

    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    if ss_tot > 0:
        R2 = 1.0 - ss_res / ss_tot
    else:
        R2 = np.nan

    return (
        VT,
        intercept,
        R2,
        x_all,
        y_all,
        fit_mask
    )


# ============================================================
# Main program
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("LOGAN GRAPHICAL ANALYSIS OF DYNAMIC PET TACs")
    print("=" * 70)

    # --------------------------------------------------------
    # File paths
    # --------------------------------------------------------

    data_file = (
        r"C:\Users\Acer\Desktop\PET\data\simulated_pet_tacs.npz"
    )

    output_file = (
        r"C:\Users\Acer\Desktop\PET\data\logan_results.npz"
    )

    figure_file = (
        r"C:\Users\Acer\Desktop\PET\figures\logan_analysis.png"
    )

    os.makedirs(
        os.path.dirname(figure_file),
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("\nLoading:")
    print(data_file)

    data = np.load(data_file)

    print("\nAvailable arrays:")

    for key in data.files:
        print(f"    {key}")

    # --------------------------------------------------------
    # Time and plasma input function
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Logan linearization time
    # --------------------------------------------------------

    t_star = 10.0

    fit_start = np.searchsorted(t, t_star)

    print(f"\nLogan t*              : {t_star:.2f} min")
    print(
        f"Logan fitting starts  : frame index {fit_start}"
    )
    print(
        f"Actual fitting time   : {t[fit_start]:.3f} min"
    )

    # --------------------------------------------------------
    # Regions
    # --------------------------------------------------------

    regions = [
        "grey_matter",
        "white_matter",
        "lesion"
    ]

    results = {}

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        3, 1,
        figsize=(12, 14)
    )

    # --------------------------------------------------------
    # Analyze each region
    # --------------------------------------------------------

    for ax, region in zip(axes, regions):

        print("\n" + "-" * 70)
        print(region.upper())
        print("-" * 70)

        Ct_noisy = data[f"{region}_noisy"]
        Ct_true = data[f"{region}_true"]

        # ====================================================
        # NOISY TAC
        # ====================================================

        (
            VT_noisy,
            intercept_noisy,
            R2_noisy,
            x_noisy,
            y_noisy,
            mask_noisy
        ) = logan_analysis(
            t,
            Ct_noisy,
            Cp,
            t_star
        )

        # ====================================================
        # TRUE TAC
        # ====================================================

        (
            VT_true,
            intercept_true,
            R2_true,
            x_true,
            y_true,
            mask_true
        ) = logan_analysis(
            t,
            Ct_true,
            Cp,
            t_star
        )

        # ----------------------------------------------------
        # Ground-truth VT from kinetic parameters
        # ----------------------------------------------------

        K1_true = float(data[f"{region}_K1"])
        k2_true = float(data[f"{region}_k2"])
        k3_true = float(data[f"{region}_k3"])
        k4_true = float(data[f"{region}_k4"])

        VT_2TCM_true = (
            K1_true / k2_true
        ) * (
            1.0 + k3_true / k4_true
        )

        # ----------------------------------------------------
        # Relative Logan error
        # ----------------------------------------------------

        error_noisy = (
            100.0 *
            abs(VT_noisy - VT_2TCM_true) /
            abs(VT_2TCM_true)
        )

        error_true = (
            100.0 *
            abs(VT_true - VT_2TCM_true) /
            abs(VT_2TCM_true)
        )

        # ----------------------------------------------------
        # Print results
        # ----------------------------------------------------

        print("\nNOISY TAC")
        print(
            f"Logan VT       : {VT_noisy:.6f}"
        )
        print(
            f"Intercept      : {intercept_noisy:.6f}"
        )
        print(
            f"R²             : {R2_noisy:.6f}"
        )
        print(
            f"VT error       : {error_noisy:.2f}%"
        )

        print("\nTRUE TAC")
        print(
            f"Logan VT       : {VT_true:.6f}"
        )
        print(
            f"Intercept      : {intercept_true:.6f}"
        )
        print(
            f"R²             : {R2_true:.6f}"
        )
        print(
            f"VT error       : {error_true:.2f}%"
        )

        print("\n2TCM ground-truth VT")
        print(
            f"VT             : {VT_2TCM_true:.6f}"
        )

        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------

        results[f"{region}_logan_VT_noisy"] = VT_noisy
        results[f"{region}_logan_VT_true"] = VT_true

        results[f"{region}_logan_intercept_noisy"] = (
            intercept_noisy
        )

        results[f"{region}_logan_intercept_true"] = (
            intercept_true
        )

        results[f"{region}_logan_R2_noisy"] = R2_noisy
        results[f"{region}_logan_R2_true"] = R2_true

        results[f"{region}_true_VT_2TCM"] = (
            VT_2TCM_true
        )

        # ----------------------------------------------------
        # Plot noisy Logan points
        # ----------------------------------------------------

        ax.scatter(
            x_noisy[mask_noisy],
            y_noisy[mask_noisy],
            label="Noisy TAC",
            s=35
        )

        # ----------------------------------------------------
        # Plot true Logan points
        # ----------------------------------------------------

        ax.plot(
            x_true[mask_true],
            y_true[mask_true],
            linestyle="--",
            label="True TAC"
        )

        # ----------------------------------------------------
        # Logan fitted line
        # ----------------------------------------------------

        x_fit = x_noisy[mask_noisy]

        y_fit = (
            VT_noisy * x_fit +
            intercept_noisy
        )

        ax.plot(
            x_fit,
            y_fit,
            linewidth=2,
            label=(
                f"Logan fit "
                f"(VT={VT_noisy:.3f})"
            )
        )

        ax.set_title(
            region.replace("_", " ").title()
        )

        ax.set_xlabel(
            r"$\int_0^t C_p(\tau)d\tau / C_t(t)$"
        )

        ax.set_ylabel(
            r"$\int_0^t C_t(\tau)d\tau / C_t(t)$"
        )

        ax.grid(True, alpha=0.3)

        ax.legend()

    # --------------------------------------------------------
    # Overall figure title
    # --------------------------------------------------------

    fig.suptitle(
        "Logan Graphical Analysis of Dynamic PET TACs",
        fontsize=16
    )

    plt.tight_layout()

    plt.savefig(
        figure_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    # --------------------------------------------------------
    # Save numerical results
    # --------------------------------------------------------

    np.savez(
        output_file,
        **results
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("LOGAN ANALYSIS COMPLETED")
    print("=" * 70)

    for region in regions:

        VT = results[
            f"{region}_logan_VT_noisy"
        ]

        R2 = results[
            f"{region}_logan_R2_noisy"
        ]

        VT_true = results[
            f"{region}_true_VT_2TCM"
        ]

        print(f"\n{region.upper()}")
        print(
            f"  Logan VT = {VT:.6f}"
        )
        print(
            f"  R²       = {R2:.6f}"
        )
        print(
            f"  True VT  = {VT_true:.6f}"
        )

    print("\nResults saved to:")
    print(output_file)

    print("\nFigure saved to:")
    print(figure_file)

    print("\n" + "=" * 70)
