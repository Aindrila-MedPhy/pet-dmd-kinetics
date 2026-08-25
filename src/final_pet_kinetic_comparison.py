# =============================================================================
# FINAL PET KINETIC ANALYSIS COMPARISON
#
# Models:
#   1. Patlak
#   2. Logan
#   3. Robust 2TCM
#   4. Validated Hankel-DMD
#
# FINAL OUTPUT:
#   - One numerical NPZ file
#   - One text summary
#   - ONE combined 2x2 comparison figure
#
# IMPORTANT:
#   - Patlak uses stored noisy-data result
#   - Logan uses stored noisy-data result
#   - Robust 2TCM uses stored validation result
#   - Hankel-DMD uses ALREADY VALIDATED stored rates
#   - DMD is NOT treated as a VT estimator
#   - DMD rates are NOT recalculated from eigenvalues here
# =============================================================================


from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# 1. PROJECT DIRECTORIES
# =============================================================================
#
# This script is located in:
#
#     PET/src/final_pet_kinetic_comparison.py
#
# Therefore:
#
#     PROJECT_DIR = PET
#     DATA_DIR    = PET/data
#     FIGURES_DIR = PET/figures
#
# This avoids dependence on the current working directory.
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

DATA_DIR = PROJECT_DIR / "data"
FIGURES_DIR = PROJECT_DIR / "figures"

DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 2. INPUT FILES
# =============================================================================

PATLAK_FILE = DATA_DIR / "patlak_results.npz"
LOGAN_FILE = DATA_DIR / "logan_results.npz"
TCM_FILE = DATA_DIR / "2tcm_validation_results.npz"
DMD_FILE = DATA_DIR / "dmd_kinetic_results_validated.npz"


# =============================================================================
# 3. OUTPUT FILES
# =============================================================================

OUTPUT_NPZ = DATA_DIR / "FINAL_PET_KINETIC_COMPARISON.npz"
OUTPUT_TXT = DATA_DIR / "FINAL_PET_KINETIC_COMPARISON.txt"

FINAL_FIGURE = FIGURES_DIR / "FINAL_PET_KINETIC_COMPARISON.png"


# =============================================================================
# 4. HELPER FUNCTIONS
# =============================================================================

def scalar(value):
    """
    Convert a NumPy scalar/one-element array to float.
    """

    arr = np.asarray(value)

    if arr.size != 1:
        raise ValueError(
            f"Expected scalar value, but received array "
            f"with shape {arr.shape}"
        )

    return float(arr.reshape(-1)[0])


def get_required(data, key):
    """
    Retrieve a required scalar key from an NPZ file.
    """

    if key not in data:
        raise KeyError(
            f"\nRequired key not found:\n"
            f"    {key}\n\n"
            f"Available keys:\n"
            f"{list(data.keys())}"
        )

    return scalar(data[key])


def percentage_error(estimated, true_value):
    """
    Absolute percentage error.
    """

    if not np.isfinite(estimated):
        return np.nan

    if not np.isfinite(true_value) or true_value == 0:
        return np.nan

    return abs(
        (estimated - true_value) / true_value
    ) * 100.0


# =============================================================================
# 5. LOAD DATA
# =============================================================================

print()
print("=" * 100)
print("FINAL PET KINETIC ANALYSIS COMPARISON")
print("=" * 100)

print()
print("Project directory:")
print(PROJECT_DIR)

print()
print("Data directory:")
print(DATA_DIR)

print()
print("Figures directory:")
print(FIGURES_DIR)

print()
print("Loading data files...")


# -----------------------------------------------------------------------------
# Patlak
# -----------------------------------------------------------------------------

print()
print("Patlak file:")
print(f"  {PATLAK_FILE}")

if not PATLAK_FILE.exists():
    raise FileNotFoundError(
        f"Patlak file not found:\n{PATLAK_FILE}"
    )

patlak = np.load(
    PATLAK_FILE,
    allow_pickle=True
)

print("  Keys:")
print(list(patlak.keys()))


# -----------------------------------------------------------------------------
# Logan
# -----------------------------------------------------------------------------

print()
print("Logan file:")
print(f"  {LOGAN_FILE}")

if not LOGAN_FILE.exists():
    raise FileNotFoundError(
        f"Logan file not found:\n{LOGAN_FILE}"
    )

logan = np.load(
    LOGAN_FILE,
    allow_pickle=True
)

print("  Keys:")
print(list(logan.keys()))


# -----------------------------------------------------------------------------
# Robust 2TCM
# -----------------------------------------------------------------------------

print()
print("Robust 2TCM file:")
print(f"  {TCM_FILE}")

if not TCM_FILE.exists():
    raise FileNotFoundError(
        f"2TCM file not found:\n{TCM_FILE}"
    )

tcm = np.load(
    TCM_FILE,
    allow_pickle=True
)

print("  Keys:")
print(list(tcm.keys()))


# -----------------------------------------------------------------------------
# Validated Hankel-DMD
# -----------------------------------------------------------------------------

print()
print("Validated Hankel-DMD file:")
print(f"  {DMD_FILE}")

if not DMD_FILE.exists():
    raise FileNotFoundError(
        f"DMD file not found:\n{DMD_FILE}"
    )

dmd = np.load(
    DMD_FILE,
    allow_pickle=True
)

print("  Keys:")
print(list(dmd.keys()))


# =============================================================================
# 6. REGIONS
# =============================================================================

regions = [
    "grey_matter",
    "white_matter",
    "lesion"
]

region_labels = {
    "grey_matter": "Grey Matter",
    "white_matter": "White Matter",
    "lesion": "Lesion"
}


# =============================================================================
# 7. EXTRACT ALL RESULTS
# =============================================================================

results = {}


print()
print("=" * 100)
print("EXTRACTING FINAL RESULTS")
print("=" * 100)


for region in regions:

    label = region_labels[region]

    print()
    print("-" * 90)
    print(label.upper())
    print("-" * 90)


    # =========================================================================
    # TRUE VT
    #
    # Ground truth comes from the 2TCM reference stored in Logan results.
    # =========================================================================

    true_vt = get_required(
        logan,
        f"{region}_true_VT_2TCM"
    )


    # =========================================================================
    # PATLAK
    #
    # Use noisy-data Patlak result.
    # =========================================================================

    patlak_ki = get_required(
        patlak,
        f"{region}_Ki_noisy"
    )

    patlak_r2 = get_required(
        patlak,
        f"{region}_R2_noisy"
    )


    # =========================================================================
    # LOGAN
    #
    # Use noisy-data Logan result.
    # =========================================================================

    logan_vt = get_required(
        logan,
        f"{region}_logan_VT_noisy"
    )

    logan_r2 = get_required(
        logan,
        f"{region}_logan_R2_noisy"
    )

    logan_error = percentage_error(
        logan_vt,
        true_vt
    )


    # =========================================================================
    # ROBUST 2TCM
    #
    # Use fitted VT and noisy-data R2.
    # =========================================================================

    tcm_vt = get_required(
        tcm,
        f"{region}_fitted_vt"
    )

    tcm_r2 = get_required(
        tcm,
        f"{region}_r2_noisy"
    )

    tcm_error = percentage_error(
        tcm_vt,
        true_vt
    )


    # =========================================================================
    # VALIDATED HANKEL-DMD
    #
    # IMPORTANT:
    #
    # These values already exist in:
    #
    #     dmd_kinetic_results_validated.npz
    #
    # We therefore use them DIRECTLY.
    #
    # No recalculation from eigenvalues is performed.
    # =========================================================================

    dmd_slow = get_required(
        dmd,
        f"{region}_slow_rate"
    )

    dmd_intermediate = get_required(
        dmd,
        f"{region}_intermediate_rate"
    )

    dmd_fast = get_required(
        dmd,
        f"{region}_fast_rate"
    )

    dmd_r2 = get_required(
        dmd,
        f"{region}_R2"
    )


    # =========================================================================
    # DMD HALF-LIVES
    # =========================================================================

    dmd_slow_half_life = get_required(
        dmd,
        f"{region}_slow_half_life"
    )

    dmd_intermediate_half_life = get_required(
        dmd,
        f"{region}_intermediate_half_life"
    )

    dmd_fast_half_life = get_required(
        dmd,
        f"{region}_fast_half_life"
    )


    # =========================================================================
    # DMD OTHER VALIDATION METRICS
    # =========================================================================

    dmd_rmse = get_required(
        dmd,
        f"{region}_RMSE"
    )

    dmd_nrmse = get_required(
        dmd,
        f"{region}_NRMSE"
    )

    dmd_rank = get_required(
        dmd,
        f"{region}_selected_rank"
    )


    # =========================================================================
    # STORE RESULTS
    # =========================================================================

    results[region] = {

        "label": label,

        # Ground truth
        "true_vt": true_vt,

        # Patlak
        "patlak_ki": patlak_ki,
        "patlak_r2": patlak_r2,

        # Logan
        "logan_vt": logan_vt,
        "logan_r2": logan_r2,
        "logan_error": logan_error,

        # 2TCM
        "tcm_vt": tcm_vt,
        "tcm_r2": tcm_r2,
        "tcm_error": tcm_error,

        # DMD rates
        "dmd_slow": dmd_slow,
        "dmd_intermediate": dmd_intermediate,
        "dmd_fast": dmd_fast,

        # DMD half-lives
        "dmd_slow_half_life": dmd_slow_half_life,
        "dmd_intermediate_half_life": dmd_intermediate_half_life,
        "dmd_fast_half_life": dmd_fast_half_life,

        # DMD quality
        "dmd_r2": dmd_r2,
        "dmd_rmse": dmd_rmse,
        "dmd_nrmse": dmd_nrmse,
        "dmd_rank": dmd_rank
    }


    # =========================================================================
    # PRINT RESULTS
    # =========================================================================

    print()
    print(f"Ground-truth VT = {true_vt:.6f}")

    print()
    print("PATLAK")
    print(f"Ki = {patlak_ki:.6f} 1/min")
    print(f"R² = {patlak_r2:.6f}")

    print()
    print("LOGAN")
    print(f"VT = {logan_vt:.6f}")
    print(f"R² = {logan_r2:.6f}")
    print(f"VT error = {logan_error:.2f}%")

    print()
    print("ROBUST 2TCM")
    print(f"VT = {tcm_vt:.6f}")
    print(f"R² = {tcm_r2:.6f}")
    print(f"VT error = {tcm_error:.2f}%")

    print()
    print("HANKEL-DMD")
    print(f"Rank = {dmd_rank:.0f}")
    print(f"Slow rate         = {dmd_slow:.6f} 1/min")
    print(f"Intermediate rate = {dmd_intermediate:.6f} 1/min")
    print(f"Fast rate          = {dmd_fast:.6f} 1/min")
    print(f"R² = {dmd_r2:.6f}")
    print(f"NRMSE = {dmd_nrmse:.6f}")

    print()
    print("DMD half-lives")
    print(
        f"Slow half-life         = "
        f"{dmd_slow_half_life:.6f} min"
    )
    print(
        f"Intermediate half-life = "
        f"{dmd_intermediate_half_life:.6f} min"
    )
    print(
        f"Fast half-life          = "
        f"{dmd_fast_half_life:.6f} min"
    )


# =============================================================================
# 8. FINAL COMPARISON TABLE
# =============================================================================

print()
print("=" * 155)
print("FINAL COMPARISON TABLE")
print("=" * 155)

header = (
    f"{'Region':<20}"
    f"{'True VT':>11}"
    f"{'Patlak Ki':>14}"
    f"{'Logan VT':>13}"
    f"{'Logan Err%':>14}"
    f"{'2TCM VT':>13}"
    f"{'2TCM Err%':>14}"
    f"{'DMD Slow':>13}"
    f"{'DMD Int.':>13}"
    f"{'DMD Fast':>13}"
)

print(header)
print("-" * len(header))


for region in regions:

    r = results[region]

    print(
        f"{r['label']:<20}"
        f"{r['true_vt']:>11.3f}"
        f"{r['patlak_ki']:>14.5f}"
        f"{r['logan_vt']:>13.3f}"
        f"{r['logan_error']:>14.2f}"
        f"{r['tcm_vt']:>13.3f}"
        f"{r['tcm_error']:>14.2f}"
        f"{r['dmd_slow']:>13.5f}"
        f"{r['dmd_intermediate']:>13.5f}"
        f"{r['dmd_fast']:>13.5f}"
    )


# =============================================================================
# 9. PREPARE ARRAYS FOR SAVING AND PLOTTING
# =============================================================================

labels = [
    region_labels[r]
    for r in regions
]

x = np.arange(
    len(regions)
)


# -----------------------------------------------------------------------------
# VT
# -----------------------------------------------------------------------------

true_vt = np.array([
    results[r]["true_vt"]
    for r in regions
])

logan_vt = np.array([
    results[r]["logan_vt"]
    for r in regions
])

tcm_vt = np.array([
    results[r]["tcm_vt"]
    for r in regions
])


# -----------------------------------------------------------------------------
# VT errors
# -----------------------------------------------------------------------------

logan_error = np.array([
    results[r]["logan_error"]
    for r in regions
])

tcm_error = np.array([
    results[r]["tcm_error"]
    for r in regions
])


# -----------------------------------------------------------------------------
# Patlak
# -----------------------------------------------------------------------------

patlak_ki = np.array([
    results[r]["patlak_ki"]
    for r in regions
])

patlak_r2 = np.array([
    results[r]["patlak_r2"]
    for r in regions
])


# -----------------------------------------------------------------------------
# Logan R2
# -----------------------------------------------------------------------------

logan_r2 = np.array([
    results[r]["logan_r2"]
    for r in regions
])


# -----------------------------------------------------------------------------
# 2TCM R2
# -----------------------------------------------------------------------------

tcm_r2 = np.array([
    results[r]["tcm_r2"]
    for r in regions
])


# -----------------------------------------------------------------------------
# DMD rates
# -----------------------------------------------------------------------------

dmd_slow = np.array([
    results[r]["dmd_slow"]
    for r in regions
])

dmd_intermediate = np.array([
    results[r]["dmd_intermediate"]
    for r in regions
])

dmd_fast = np.array([
    results[r]["dmd_fast"]
    for r in regions
])


# -----------------------------------------------------------------------------
# DMD R2
# -----------------------------------------------------------------------------

dmd_r2 = np.array([
    results[r]["dmd_r2"]
    for r in regions
])


# -----------------------------------------------------------------------------
# DMD NRMSE
# -----------------------------------------------------------------------------

dmd_nrmse = np.array([
    results[r]["dmd_nrmse"]
    for r in regions
])


# -----------------------------------------------------------------------------
# DMD rank
# -----------------------------------------------------------------------------

dmd_rank = np.array([
    results[r]["dmd_rank"]
    for r in regions
])


# =============================================================================
# 10. SAVE FINAL NUMERICAL RESULTS
# =============================================================================

np.savez(
    OUTPUT_NPZ,

    regions=np.array(regions),
    region_labels=np.array(labels),

    # True VT
    true_vt=true_vt,

    # Patlak
    patlak_ki=patlak_ki,
    patlak_r2=patlak_r2,

    # Logan
    logan_vt=logan_vt,
    logan_r2=logan_r2,
    logan_error=logan_error,

    # Robust 2TCM
    tcm_vt=tcm_vt,
    tcm_r2=tcm_r2,
    tcm_error=tcm_error,

    # DMD
    dmd_slow=dmd_slow,
    dmd_intermediate=dmd_intermediate,
    dmd_fast=dmd_fast,

    dmd_r2=dmd_r2,
    dmd_nrmse=dmd_nrmse,
    dmd_rank=dmd_rank,

    # DMD half-lives
    dmd_slow_half_life=np.array([
        results[r]["dmd_slow_half_life"]
        for r in regions
    ]),

    dmd_intermediate_half_life=np.array([
        results[r]["dmd_intermediate_half_life"]
        for r in regions
    ]),

    dmd_fast_half_life=np.array([
        results[r]["dmd_fast_half_life"]
        for r in regions
    ])
)

print()
print("Results saved to:")
print(OUTPUT_NPZ)


# =============================================================================
# 11. SAVE TEXT SUMMARY
# =============================================================================

with open(
    OUTPUT_TXT,
    "w",
    encoding="utf-8"
) as f:

    f.write("=" * 100 + "\n")
    f.write("FINAL PET KINETIC ANALYSIS COMPARISON\n")
    f.write("=" * 100 + "\n\n")

    f.write(
        "Patlak, Logan and Robust 2TCM use stored noisy-data "
        "validation results.\n"
    )

    f.write(
        "Hankel-DMD uses the validated stored characteristic "
        "rates and reconstruction metrics.\n"
    )

    f.write(
        "DMD is not treated as a VT estimator.\n\n"
    )


    for region in regions:

        r = results[region]

        f.write("-" * 90 + "\n")
        f.write(
            f"{r['label'].upper()}\n"
        )
        f.write("-" * 90 + "\n\n")

        f.write(
            f"Ground-truth VT = "
            f"{r['true_vt']:.6f}\n\n"
        )

        f.write("PATLAK\n")
        f.write(
            f"Ki = {r['patlak_ki']:.6f} 1/min\n"
        )
        f.write(
            f"R2 = {r['patlak_r2']:.6f}\n\n"
        )

        f.write("LOGAN\n")
        f.write(
            f"VT = {r['logan_vt']:.6f}\n"
        )
        f.write(
            f"R2 = {r['logan_r2']:.6f}\n"
        )
        f.write(
            f"VT error = {r['logan_error']:.2f}%\n\n"
        )

        f.write("ROBUST 2TCM\n")
        f.write(
            f"VT = {r['tcm_vt']:.6f}\n"
        )
        f.write(
            f"R2 = {r['tcm_r2']:.6f}\n"
        )
        f.write(
            f"VT error = {r['tcm_error']:.2f}%\n\n"
        )

        f.write("VALIDATED HANKEL-DMD\n")
        f.write(
            f"Rank = {r['dmd_rank']:.0f}\n"
        )
        f.write(
            f"Slow rate = "
            f"{r['dmd_slow']:.6f} 1/min\n"
        )
        f.write(
            f"Intermediate rate = "
            f"{r['dmd_intermediate']:.6f} 1/min\n"
        )
        f.write(
            f"Fast rate = "
            f"{r['dmd_fast']:.6f} 1/min\n"
        )
        f.write(
            f"R2 = {r['dmd_r2']:.6f}\n"
        )
        f.write(
            f"NRMSE = {r['dmd_nrmse']:.6f}\n\n"
        )

        f.write("DMD HALF-LIVES\n")
        f.write(
            f"Slow = "
            f"{r['dmd_slow_half_life']:.6f} min\n"
        )
        f.write(
            f"Intermediate = "
            f"{r['dmd_intermediate_half_life']:.6f} min\n"
        )
        f.write(
            f"Fast = "
            f"{r['dmd_fast_half_life']:.6f} min\n\n"
        )


    # -------------------------------------------------------------------------
    # Compact table
    # -------------------------------------------------------------------------

    f.write("\n")
    f.write("=" * 155 + "\n")
    f.write("FINAL COMPARISON TABLE\n")
    f.write("=" * 155 + "\n")

    f.write(header + "\n")
    f.write("-" * len(header) + "\n")

    for region in regions:

        r = results[region]

        f.write(
            f"{r['label']:<20}"
            f"{r['true_vt']:>11.3f}"
            f"{r['patlak_ki']:>14.5f}"
            f"{r['logan_vt']:>13.3f}"
            f"{r['logan_error']:>14.2f}"
            f"{r['tcm_vt']:>13.3f}"
            f"{r['tcm_error']:>14.2f}"
            f"{r['dmd_slow']:>13.5f}"
            f"{r['dmd_intermediate']:>13.5f}"
            f"{r['dmd_fast']:>13.5f}\n"
        )


print()
print("Summary saved to:")
print(OUTPUT_TXT)


# =============================================================================
# 12. ONE FINAL FIGURE — 2 x 2 SUBPLOTS
# =============================================================================
#
# Panel A:
#     VT comparison
#
#     Ground truth
#     Logan
#     Robust 2TCM
#
#     DMD is intentionally NOT shown because it does not estimate VT.
#
# Panel B:
#     VT estimation error
#
#     Logan
#     Robust 2TCM
#
# Panel C:
#     Hankel-DMD characteristic rates
#
#     Slow
#     Intermediate
#     Fast
#
# Panel D:
#     R² comparison
#
#     Patlak
#     Logan
#     Robust 2TCM
#     Hankel-DMD
# =============================================================================

print()
print("=" * 100)
print("GENERATING FINAL COMBINED FIGURE")
print("=" * 100)


fig, axes = plt.subplots(
    2,
    2,
    figsize=(16, 11)
)


# =============================================================================
# PANEL A — VT COMPARISON
# =============================================================================

ax = axes[0, 0]

width_vt = 0.25

ax.bar(
    x - width_vt,
    true_vt,
    width_vt,
    label="Ground truth"
)

ax.bar(
    x,
    logan_vt,
    width_vt,
    label="Logan"
)

ax.bar(
    x + width_vt,
    tcm_vt,
    width_vt,
    label="Robust 2TCM"
)

ax.set_title(
    "(A) Distribution Volume ($V_T$) Comparison",
    fontsize=15,
    fontweight="bold"
)

ax.set_xlabel(
    "Region",
    fontsize=12
)

ax.set_ylabel(
    "$V_T$",
    fontsize=12
)

ax.set_xticks(x)
ax.set_xticklabels(labels)

ax.legend(
    fontsize=10
)

ax.grid(
    axis="y",
    alpha=0.3
)


# =============================================================================
# PANEL B — VT ERROR
# =============================================================================

ax = axes[0, 1]

width_error = 0.32

ax.bar(
    x - width_error / 2,
    logan_error,
    width_error,
    label="Logan"
)

ax.bar(
    x + width_error / 2,
    tcm_error,
    width_error,
    label="Robust 2TCM"
)

ax.set_title(
    "(B) $V_T$ Estimation Error",
    fontsize=15,
    fontweight="bold"
)

ax.set_xlabel(
    "Region",
    fontsize=12
)

ax.set_ylabel(
    "Absolute error (%)",
    fontsize=12
)

ax.set_xticks(x)
ax.set_xticklabels(labels)

ax.legend(
    fontsize=10
)

ax.grid(
    axis="y",
    alpha=0.3
)


# =============================================================================
# PANEL C — DMD RATES
# =============================================================================

ax = axes[1, 0]

width_rate = 0.25

ax.bar(
    x - width_rate,
    dmd_slow,
    width_rate,
    label="Slow"
)

ax.bar(
    x,
    dmd_intermediate,
    width_rate,
    label="Intermediate"
)

ax.bar(
    x + width_rate,
    dmd_fast,
    width_rate,
    label="Fast"
)

ax.set_title(
    "(C) Validated Hankel-DMD Characteristic Rates",
    fontsize=15,
    fontweight="bold"
)

ax.set_xlabel(
    "Region",
    fontsize=12
)

ax.set_ylabel(
    "Decay rate (1/min)",
    fontsize=12
)

ax.set_xticks(x)
ax.set_xticklabels(labels)

ax.legend(
    fontsize=10
)

ax.grid(
    axis="y",
    alpha=0.3
)


# =============================================================================
# PANEL D — R² COMPARISON
# =============================================================================

ax = axes[1, 1]

width_r2 = 0.18

ax.bar(
    x - 1.5 * width_r2,
    patlak_r2,
    width_r2,
    label="Patlak"
)

ax.bar(
    x - 0.5 * width_r2,
    logan_r2,
    width_r2,
    label="Logan"
)

ax.bar(
    x + 0.5 * width_r2,
    tcm_r2,
    width_r2,
    label="Robust 2TCM"
)

ax.bar(
    x + 1.5 * width_r2,
    dmd_r2,
    width_r2,
    label="Hankel-DMD"
)

ax.set_title(
    "(D) Model Fit / Reconstruction Quality",
    fontsize=15,
    fontweight="bold"
)

ax.set_xlabel(
    "Region",
    fontsize=12
)

ax.set_ylabel(
    "$R^2$",
    fontsize=12
)

ax.set_xticks(x)
ax.set_xticklabels(labels)

ax.set_ylim(
    0.0,
    1.05
)

ax.legend(
    fontsize=9
)

ax.grid(
    axis="y",
    alpha=0.3
)


# =============================================================================
# FINAL FIGURE FORMATTING
# =============================================================================

fig.suptitle(
    "Final PET Kinetic Analysis Comparison",
    fontsize=20,
    fontweight="bold"
)

fig.text(
    0.5,
    0.01,
    "Patlak and Logan provide graphical kinetic estimates; "
    "Robust 2TCM provides model-based VT; "
    "Hankel-DMD provides model-independent characteristic decay rates.",
    ha="center",
    fontsize=10
)

fig.tight_layout(
    rect=[
        0,
        0.04,
        1,
        0.95
    ]
)


# =============================================================================
# SAVE ONE FINAL FIGURE
# =============================================================================

plt.savefig(
    FINAL_FIGURE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print()
print(f"Saved final combined figure:")
print(FINAL_FIGURE)


# =============================================================================
# 13. FINAL SUMMARY
# =============================================================================

print()
print("=" * 100)
print("FINAL PET KINETIC COMPARISON COMPLETED")
print("=" * 100)

print()
print("Results saved to:")
print(OUTPUT_NPZ)

print()
print("Summary saved to:")
print(OUTPUT_TXT)

print()
print("Final figure saved to:")
print(FINAL_FIGURE)

print()
print("The final figure contains:")
print("  (A) VT comparison")
print("  (B) VT estimation error")
print("  (C) Hankel-DMD characteristic rates")
print("  (D) R² comparison")

print()
print("The comparison uses:")
print("  ✓ Patlak")
print("  ✓ Logan")
print("  ✓ Robust 2TCM")
print("  ✓ Validated Hankel-DMD")

print()
print("Important:")
print("  ✓ DMD rates are read directly from validated DMD results")
print("  ✓ DMD is not treated as a VT estimator")
print("  ✓ Logan VT error is calculated against ground-truth VT")
print("  ✓ 2TCM VT error is calculated against ground-truth VT")
print("  ✓ All paths are derived from the script location")
print("  ✓ No dependence on current working directory")

print()
print("=" * 100)
