"""
plot_simulated_tacs.py

Visual inspection of the simulated dynamic PET data.

Plots:
1. Arterial input function (AIF)
2. True vs noisy TACs for each tissue region
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PROJECT PATHS
# ============================================================

# PET/
# ├── src/
# │   └── plot_simulated_tacs.py
# └── data/
#     └── simulated_pet_tacs.npz

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"

DATA_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

DATA_FILE = DATA_DIR / "simulated_pet_tacs.npz"

print("=" * 80)
print("PLOTTING SIMULATED PET TACS")
print("=" * 80)
print()
print("Loading:")
print(DATA_FILE)
print()

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"\nSimulation file not found:\n{DATA_FILE}\n\n"
        f"Expected project structure:\n"
        f"{PROJECT_ROOT}\\\n"
        f"  data\\simulated_pet_tacs.npz\n"
        f"  src\\plot_simulated_tacs.py\n"
    )

data = np.load(DATA_FILE)

frame_edges = data["frame_edges"]
t = data["frame_midtimes"]
Cp = data["Cp"]


regions = [
    "grey_matter",
    "white_matter",
    "lesion",
]


# -------------------------------------------------------------
# Print basic information
# -------------------------------------------------------------

print("=" * 60)
print("SIMULATED PET DATA CHECK")
print("=" * 60)

print(f"Number of frames: {len(t)}")
print(f"Start time: {frame_edges[0]:.2f} min")
print(f"End time: {frame_edges[-1]:.2f} min")

print("\nFrame mid-times:")
print(np.round(t, 3))


# -------------------------------------------------------------
# Figure 1: Arterial input function
# -------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    t,
    Cp,
    "o-",
    linewidth=2,
    markersize=4,
)

plt.xlabel("Time (min)")
plt.ylabel("Plasma activity $C_p(t)$")
plt.title("Simulated Arterial Input Function")

plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "figures_aif.png",
    dpi=300,
)

plt.show()


# -------------------------------------------------------------
# Figure 2: True vs noisy TACs
# -------------------------------------------------------------

plt.figure(figsize=(9, 6))

for region in regions:

    true_tac = data[f"{region}_true"]
    noisy_tac = data[f"{region}_noisy"]

    plt.plot(
        t,
        true_tac,
        linewidth=2,
        label=f"{region.replace('_', ' ').title()} — true",
    )

    plt.scatter(
        t,
        noisy_tac,
        s=25,
        alpha=0.7,
        label=f"{region.replace('_', ' ').title()} — noisy",
    )


plt.xlabel("Time (min)")
plt.ylabel("Tissue activity")
plt.title("Simulated Dynamic PET Time-Activity Curves")

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    "figures_tacs.png",
    dpi=300,
)

plt.show()


# -------------------------------------------------------------
# Figure 3: Individual regions
# -------------------------------------------------------------

fig, axes = plt.subplots(
    3,
    1,
    figsize=(9, 10),
    sharex=True,
)

for ax, region in zip(axes, regions):

    true_tac = data[f"{region}_true"]
    noisy_tac = data[f"{region}_noisy"]

    ax.plot(
        t,
        true_tac,
        linewidth=2,
        label="True TAC",
    )

    ax.scatter(
        t,
        noisy_tac,
        s=25,
        alpha=0.7,
        label="Noisy TAC",
    )

    ax.set_ylabel("Activity")

    ax.set_title(
        region.replace("_", " ").title()
    )

    ax.grid(True, alpha=0.3)
    ax.legend()


axes[-1].set_xlabel("Time (min)")

plt.tight_layout()

plt.savefig(
    "figures_individual_tacs.png",
    dpi=300,
)

plt.show()


print("\nPlots saved successfully.")
