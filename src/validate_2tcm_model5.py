"""
validate_2tcm_model.py

Robust validation of the reversible 2-tissue-compartment model
for dynamic PET TACs.

Purpose:
    1. Fit 2TCM using multi-start nonlinear least squares.
    2. Avoid dependence on a single initial guess.
    3. Use wider parameter bounds.
    4. Calculate VT from fitted kinetic parameters.
    5. Compare fitted TAC with noisy and true TAC.
    6. Save results for subsequent comparison with Logan/Patlak/Hankel-DMD.

2TCM equations:

dC1/dt = K1*Cp - (k2+k3)*C1 + k4*C2
dC2/dt = k3*C1 - k4*C2

Tissue concentration:

Ct = (1-vB)*(C1+C2) + vB*Cp

Distribution volume:

VT = K1/k2 * (1 + k3/k4)
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.optimize import least_squares


# ============================================================
# PATHS
# ============================================================

DATA_FILE = r"C:\Users\Acer\Desktop\PET\data\simulated_pet_tacs.npz"

RESULT_FILE = (
    r"C:\Users\Acer\Desktop\PET\data\2tcm_validation_results.npz"
)

FIGURE_FILE = (
    r"C:\Users\Acer\Desktop\PET\figures\2tcm_validation.png"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("ROBUST 2TCM VALIDATION")
print("=" * 80)

print("\nLoading:")
print(DATA_FILE)

data = np.load(DATA_FILE)

t = data["frame_midtimes"]
Cp = data["Cp"]

print("\nPET frames :", len(t))
print("Time range :", t[0], "-", t[-1], "min")


# ============================================================
# 2TCM FORWARD MODEL
# ============================================================

def interpolate_cp(t_query):
    """
    Linear interpolation of plasma input function.
    """

    return np.interp(
        t_query,
        t,
        Cp,
        left=Cp[0],
        right=Cp[-1]
    )


def two_tcm_solution(params, t_eval):
    """
    Solve reversible 2TCM.

    params = [K1, k2, k3, k4, vB]
    """

    K1, k2, k3, k4, vB = params

    def rhs(time, state):

        C1, C2 = state

        cp = interpolate_cp(time)

        dC1 = (
            K1 * cp
            - (k2 + k3) * C1
            + k4 * C2
        )

        dC2 = (
            k3 * C1
            - k4 * C2
        )

        return [dC1, dC2]

    sol = solve_ivp(
        rhs,
        (t[0], t[-1]),
        [0.0, 0.0],
        t_eval=t_eval,
        method="LSODA",
        rtol=1e-7,
        atol=1e-9
    )

    C1 = sol.y[0]
    C2 = sol.y[1]

    Cp_eval = interpolate_cp(t_eval)

    Ct = (
        (1.0 - vB) * (C1 + C2)
        + vB * Cp_eval
    )

    return Ct


# ============================================================
# VT
# ============================================================

def calculate_vt(params):
    """
    Total distribution volume for reversible 2TCM.

    VT = K1/k2 * (1 + k3/k4)
    """

    K1, k2, k3, k4, vB = params

    return (K1 / k2) * (1.0 + k3 / k4)


# ============================================================
# FIT QUALITY
# ============================================================

def calculate_metrics(observed, predicted):

    residual = observed - predicted

    rmse = np.sqrt(
        np.mean(residual ** 2)
    )

    ss_res = np.sum(residual ** 2)

    ss_tot = np.sum(
        (observed - np.mean(observed)) ** 2
    )

    if ss_tot > 0:
        r2 = 1.0 - ss_res / ss_tot
    else:
        r2 = np.nan

    return rmse, r2


# ============================================================
# MULTI-START FIT
# ============================================================

def fit_2tcm(tac):

    """
    Multi-start nonlinear least-squares fit.

    Wider bounds than the previous implementation.
    """

    # --------------------------------------------------------
    # Parameter bounds
    # --------------------------------------------------------

    lower = np.array([
        0.001,     # K1
        0.001,     # k2
        0.001,     # k3
        0.001,     # k4
        0.0        # vB
    ])

    upper = np.array([
        2.0,        # K1
        2.0,        # k2
        2.0,        # k3
        2.0,        # k4
        0.30        # vB
    ])

    # --------------------------------------------------------
    # Several initial guesses
    # --------------------------------------------------------

    initial_guesses = [

        [0.10, 0.10, 0.05, 0.02, 0.03],
        [0.20, 0.20, 0.10, 0.02, 0.05],
        [0.30, 0.25, 0.12, 0.02, 0.05],
        [0.40, 0.20, 0.20, 0.02, 0.05],
        [0.50, 0.30, 0.30, 0.05, 0.05],
        [0.70, 0.50, 0.50, 0.10, 0.10],
        [1.00, 1.00, 0.50, 0.10, 0.10],
    ]

    best_result = None
    best_cost = np.inf

    # --------------------------------------------------------
    # Residual function
    # --------------------------------------------------------

    def residuals(params):

        try:

            predicted = two_tcm_solution(
                params,
                t
            )

            residual = predicted - tac

            if not np.all(np.isfinite(residual)):
                return np.ones_like(tac) * 1e10

            return residual

        except Exception:

            return np.ones_like(tac) * 1e10

    # --------------------------------------------------------
    # Multi-start optimization
    # --------------------------------------------------------

    for i, x0 in enumerate(initial_guesses):

        print(
            f"    Starting fit {i+1}/{len(initial_guesses)}..."
        )

        try:

            result = least_squares(
                residuals,
                x0,
                bounds=(lower, upper),
                method="trf",
                loss="linear",
                max_nfev=5000,
                xtol=1e-10,
                ftol=1e-10,
                gtol=1e-10
            )

            cost = np.sum(result.fun ** 2)

            if cost < best_cost:

                best_cost = cost
                best_result = result

        except Exception as exc:

            print(
                "      Fit failed:",
                exc
            )

    if best_result is None:
        raise RuntimeError("All 2TCM fits failed.")

    params = best_result.x

    predicted = two_tcm_solution(
        params,
        t
    )

    rmse, r2 = calculate_metrics(
        tac,
        predicted
    )

    vt = calculate_vt(params)

    return params, vt, predicted, rmse, r2


# ============================================================
# REGIONS
# ============================================================

regions = [
    "grey_matter",
    "white_matter",
    "lesion"
]


results = {}


# ============================================================
# FIT EACH REGION
# ============================================================

for region in regions:

    print("\n")
    print("-" * 80)
    print(region.upper())
    print("-" * 80)

    tac_noisy = data[f"{region}_noisy"]
    tac_true = data[f"{region}_true"]

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    true_params = np.array([
        float(data[f"{region}_K1"]),
        float(data[f"{region}_k2"]),
        float(data[f"{region}_k3"]),
        float(data[f"{region}_k4"]),
        float(data[f"{region}_vB"])
    ])

    true_vt = calculate_vt(true_params)

    print("\nGROUND TRUTH")
    print(
        f"K1 = {true_params[0]:.6f}"
    )
    print(
        f"k2 = {true_params[1]:.6f}"
    )
    print(
        f"k3 = {true_params[2]:.6f}"
    )
    print(
        f"k4 = {true_params[3]:.6f}"
    )
    print(
        f"vB = {true_params[4]:.6f}"
    )
    print(
        f"VT = {true_vt:.6f}"
    )

    # --------------------------------------------------------
    # Fit noisy TAC
    # --------------------------------------------------------

    print("\nFITTING NOISY TAC...")

    fitted_params, fitted_vt, fitted_tac, rmse_noisy, r2_noisy = (
        fit_2tcm(tac_noisy)
    )

    # --------------------------------------------------------
    # Evaluate fit against true TAC
    # --------------------------------------------------------

    rmse_true, r2_true = calculate_metrics(
        tac_true,
        fitted_tac
    )

    # --------------------------------------------------------
    # Parameter errors
    # --------------------------------------------------------

    param_errors = (
        np.abs(
            (fitted_params - true_params)
            / true_params
        ) * 100.0
    )

    vt_error = (
        abs(fitted_vt - true_vt)
        / true_vt
        * 100.0
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\nFITTED PARAMETERS")

    print(
        f"K1 = {fitted_params[0]:.6f}"
    )

    print(
        f"k2 = {fitted_params[1]:.6f}"
    )

    print(
        f"k3 = {fitted_params[2]:.6f}"
    )

    print(
        f"k4 = {fitted_params[3]:.6f}"
    )

    print(
        f"vB = {fitted_params[4]:.6f}"
    )

    print(
        f"VT = {fitted_vt:.6f}"
    )

    print("\nPARAMETER ERRORS")

    print(
        f"K1 error = {param_errors[0]:.2f}%"
    )

    print(
        f"k2 error = {param_errors[1]:.2f}%"
    )

    print(
        f"k3 error = {param_errors[2]:.2f}%"
    )

    print(
        f"k4 error = {param_errors[3]:.2f}%"
    )

    print(
        f"vB error = {param_errors[4]:.2f}%"
    )

    print(
        f"VT error = {vt_error:.2f}%"
    )

    print("\nFIT QUALITY")

    print(
        f"RMSE vs noisy TAC = {rmse_noisy:.6f}"
    )

    print(
        f"R² vs noisy TAC   = {r2_noisy:.6f}"
    )

    print(
        f"RMSE vs true TAC  = {rmse_true:.6f}"
    )

    print(
        f"R² vs true TAC    = {r2_true:.6f}"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    results[region] = {

        "true_params": true_params,
        "fitted_params": fitted_params,

        "true_vt": true_vt,
        "fitted_vt": fitted_vt,

        "vt_error": vt_error,

        "parameter_errors": param_errors,

        "fitted_tac": fitted_tac,

        "rmse_noisy": rmse_noisy,
        "r2_noisy": r2_noisy,

        "rmse_true": rmse_true,
        "r2_true": r2_true
    }


# ============================================================
# PLOT
# ============================================================

fig, axes = plt.subplots(
    3,
    1,
    figsize=(12, 14),
    sharex=True
)

for ax, region in zip(axes, regions):

    tac_noisy = data[f"{region}_noisy"]
    tac_true = data[f"{region}_true"]

    fitted_tac = results[region]["fitted_tac"]

    vt = results[region]["fitted_vt"]
    r2 = results[region]["r2_noisy"]

    ax.plot(
        t,
        tac_true,
        linewidth=2,
        label="Ground truth"
    )

    ax.scatter(
        t,
        tac_noisy,
        s=35,
        alpha=0.7,
        label="Noisy TAC"
    )

    ax.plot(
        t,
        fitted_tac,
        "--",
        linewidth=2,
        label="2TCM fit"
    )

    ax.set_title(
        f"{region.replace('_', ' ').title()} "
        f"(VT={vt:.3f}, R²={r2:.4f})"
    )

    ax.set_ylabel("Activity")

    ax.grid(alpha=0.3)

    ax.legend()


axes[-1].set_xlabel("Time (min)")

fig.suptitle(
    "Robust Reversible 2TCM Validation",
    fontsize=16
)

fig.tight_layout()

plt.savefig(
    FIGURE_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# SAVE RESULTS
# ============================================================

save_dict = {}

for region in regions:

    for key, value in results[region].items():

        save_dict[
            f"{region}_{key}"
        ] = value

np.savez(
    RESULT_FILE,
    **save_dict
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 80)
print("ROBUST 2TCM VALIDATION COMPLETED")
print("=" * 80)

print(
    f"\nResults saved to:\n{RESULT_FILE}"
)

print(
    f"\nFigure saved to:\n{FIGURE_FILE}"
)

print("\nSUMMARY")
print("-" * 80)

print(
    f"{'Region':15s}"
    f"{'True VT':>12s}"
    f"{'Fitted VT':>14s}"
    f"{'VT error %':>14s}"
    f"{'R²':>12s}"
)

print("-" * 80)

for region in regions:

    true_vt = results[region]["true_vt"]
    fitted_vt = results[region]["fitted_vt"]
    vt_error = results[region]["vt_error"]
    r2 = results[region]["r2_noisy"]

    print(
        f"{region:15s}"
        f"{true_vt:12.4f}"
        f"{fitted_vt:14.4f}"
        f"{vt_error:14.2f}"
        f"{r2:12.5f}"
    )

print("=" * 80)
