# =============================================================================
# dmd_kinetic_analysis4.py
#
# VALIDATED HANKEL-DMD PET KINETIC ANALYSIS
#
# Purpose:
#   1. Load simulated PET TACs
#   2. Interpolate irregular PET frame times onto a uniform grid
#   3. Construct Hankel delay-coordinate matrices
#   4. Test multiple DMD ranks
#   5. Evaluate reconstruction quality using R2 and normalized RMSE
#   6. Select a suitable DMD rank automatically
#   7. Separate reconstruction modes from kinetic characteristic modes
#   8. Extract physically interpretable real decay rates
#   9. Calculate half-lives
#  10. Save all results
#
# Regions:
#   grey_matter
#   white_matter
#   lesion
#
# =============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# USER SETTINGS
# =============================================================================

DATA_FILE = r"C:\Users\Acer\Desktop\PET\data\simulated_pet_tacs.npz"

RESULT_FILE = r"C:\Users\Acer\Desktop\PET\data\dmd_kinetic_results_validated.npz"

FIGURE_DIR = r"C:\Users\Acer\Desktop\PET\figures"

os.makedirs(FIGURE_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# DMD SETTINGS
# -----------------------------------------------------------------------------

# Uniform sampling interval in minutes
DT = 0.25

# Candidate ranks to test
MIN_RANK = 2
MAX_RANK = 12

# Maximum Hankel embedding dimension
MAX_EMBEDDING = 40

# Singular-value energy threshold used only as a diagnostic
ENERGY_THRESHOLD = 0.999

# Reconstruction quality threshold
TARGET_R2 = 0.995

# Characteristic kinetic mode tolerance:
# imaginary part must be very small compared with the eigenvalue
IMAG_TOL = 1e-6

# Eigenvalue must represent a decaying mode:
# 0 < eigenvalue < 1
EIG_MIN = 0.0
EIG_MAX = 1.0

# Minimum amplitude relative to largest DMD amplitude
# used when identifying meaningful characteristic modes
MIN_RELATIVE_AMPLITUDE = 1e-4


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def r2_score(y_true, y_pred):
    """
    Coefficient of determination.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot <= 0:
        return np.nan

    return 1.0 - ss_res / ss_tot


def rmse(y_true, y_pred):
    """
    Root mean square error.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def normalized_rmse(y_true, y_pred):
    """
    RMSE normalized by the dynamic range of the TAC.
    """
    r = rmse(y_true, y_pred)

    scale = np.max(y_true) - np.min(y_true)

    if scale <= 0:
        return np.nan

    return r / scale


def half_life(rate):
    """
    Half-life for a positive first-order decay rate.
    """
    if rate <= 0:
        return np.inf

    return np.log(2.0) / rate


# =============================================================================
# TAC LOADING
# =============================================================================

def get_region_tac(data, region):
    """
    Load the noisy TAC.

    The simulated PET file contains:
        grey_matter_noisy
        white_matter_noisy
        lesion_noisy
    """

    noisy_key = f"{region}_noisy"

    if noisy_key in data:
        return np.asarray(data[noisy_key], dtype=float), noisy_key

    true_key = f"{region}_true"

    if true_key in data:
        print(
            f"WARNING: {noisy_key} not found. "
            f"Using {true_key} instead."
        )
        return np.asarray(data[true_key], dtype=float), true_key

    raise KeyError(
        f"Could not find TAC for {region}.\n"
        f"Available keys:\n{list(data.keys())}"
    )


# =============================================================================
# UNIFORM INTERPOLATION
# =============================================================================

def make_uniform_tac(t, tac, dt):
    """
    Interpolate the PET TAC onto a uniform time grid.

    DMD assumes equally spaced snapshots.
    """

    t = np.asarray(t, dtype=float)
    tac = np.asarray(tac, dtype=float)

    # Remove invalid values
    valid = np.isfinite(t) & np.isfinite(tac)

    t = t[valid]
    tac = tac[valid]

    # Sort
    order = np.argsort(t)
    t = t[order]
    tac = tac[order]

    # Remove duplicate times
    unique_t, unique_idx = np.unique(t, return_index=True)

    t = unique_t
    tac = tac[unique_idx]

    start = t[0]
    end = t[-1]

    uniform_t = np.arange(start, end + 0.5 * dt, dt)

    uniform_t = uniform_t[uniform_t <= end]

    uniform_tac = np.interp(
        uniform_t,
        t,
        tac
    )

    return uniform_t, uniform_tac


# =============================================================================
# HANKEL MATRIX
# =============================================================================

def build_hankel(y, embedding):
    """
    Construct scalar time-series Hankel matrices.

    H0 columns:
        [y0, y1, ..., y(d-1)]^T
        [y1, y2, ..., yd]^T
        ...

    H1 is shifted by one sample.
    """

    y = np.asarray(y, dtype=float)

    n = len(y)

    if embedding >= n:
        raise ValueError(
            f"Embedding dimension {embedding} must be "
            f"smaller than number of samples {n}."
        )

    columns = n - embedding

    H0 = np.zeros((embedding, columns))
    H1 = np.zeros((embedding, columns))

    for j in range(columns):
        H0[:, j] = y[j:j + embedding]
        H1[:, j] = y[j + 1:j + embedding + 1]

    return H0, H1


# =============================================================================
# DMD
# =============================================================================

def exact_dmd(H0, H1, rank):
    """
    Exact DMD using truncated SVD.

    Returns:
        eigenvalues
        DMD modes
        amplitudes
        singular values
    """

    U, S, Vh = np.linalg.svd(
        H0,
        full_matrices=False
    )

    r = min(
        rank,
        len(S),
        U.shape[1],
        Vh.shape[0]
    )

    U_r = U[:, :r]
    S_r = S[:r]
    V_r = Vh.conj().T[:, :r]

    # Avoid division by zero
    safe_S = np.where(
        S_r > np.finfo(float).eps,
        S_r,
        np.finfo(float).eps
    )

    A_tilde = (
        U_r.conj().T
        @ H1
        @ V_r
        @ np.diag(1.0 / safe_S)
    )

    eigenvalues, W = np.linalg.eig(A_tilde)

    Phi = (
        H1
        @ V_r
        @ np.diag(1.0 / safe_S)
        @ W
    )

    # Initial state
    x0 = H0[:, 0]

    # Least-squares amplitudes
    amplitudes, *_ = np.linalg.lstsq(
        Phi,
        x0,
        rcond=None
    )

    return (
        eigenvalues,
        Phi,
        amplitudes,
        S
    )


# =============================================================================
# DMD RECONSTRUCTION
# =============================================================================

def reconstruct_dmd(
    eigenvalues,
    modes,
    amplitudes,
    n_samples,
    dt
):
    """
    Reconstruct the Hankel state trajectory.

    The first component of each delay-coordinate state
    corresponds to the scalar TAC.

    This is the important correction relative to the
    previous reconstruction approach.
    """

    n_modes = len(eigenvalues)

    time_indices = np.arange(n_samples)

    dynamics = np.zeros(
        (n_modes, n_samples),
        dtype=complex
    )

    for i in range(n_modes):

        dynamics[i, :] = (
            amplitudes[i]
            * eigenvalues[i] ** time_indices
        )

    X_dmd = modes @ dynamics

    # First row is the reconstructed scalar TAC
    y_reconstructed = np.real(X_dmd[0, :])

    return y_reconstructed


# =============================================================================
# ALTERNATIVE CONTINUOUS-TIME RECONSTRUCTION
# =============================================================================

def reconstruct_from_rates(
    rates,
    amplitudes,
    n_samples,
    dt
):
    """
    Continuous-time reconstruction.

    Mainly used for diagnostics.
    """

    time = np.arange(n_samples) * dt

    result = np.zeros(n_samples, dtype=complex)

    for rate, amplitude in zip(rates, amplitudes):

        result += (
            amplitude
            * np.exp(rate * time)
        )

    return np.real(result)


# =============================================================================
# SINGULAR VALUE ENERGY
# =============================================================================

def singular_value_energy(S):
    """
    Cumulative normalized singular-value energy.
    """

    energy = S ** 2

    total = np.sum(energy)

    if total <= 0:
        return np.zeros_like(energy)

    cumulative = np.cumsum(energy) / total

    return cumulative


# =============================================================================
# RANK TESTING
# =============================================================================

def test_ranks(
    y,
    dt,
    embedding,
    min_rank,
    max_rank
):
    """
    Test multiple DMD ranks.

    Returns a list of dictionaries.
    """

    H0, H1 = build_hankel(
        y,
        embedding
    )

    U, S, Vh = np.linalg.svd(
        H0,
        full_matrices=False
    )

    max_possible_rank = min(
        max_rank,
        len(S),
        H0.shape[0],
        H0.shape[1]
    )

    results = []

    for rank in range(
        min_rank,
        max_possible_rank + 1
    ):

        try:

            eigvals, modes, amplitudes, singular_values = exact_dmd(
                H0,
                H1,
                rank
            )

            # Reconstruct the complete Hankel state
            n_samples = H0.shape[1]

            y_rec = reconstruct_dmd(
                eigvals,
                modes,
                amplitudes,
                n_samples,
                dt
            )

            # H0 first-row target
            y_target = y[:n_samples]

            R2 = r2_score(
                y_target,
                y_rec
            )

            RMSE = rmse(
                y_target,
                y_rec
            )

            NRMSE = normalized_rmse(
                y_target,
                y_rec
            )

            cumulative_energy = singular_value_energy(
                singular_values
            )

            energy_at_rank = cumulative_energy[
                rank - 1
            ]

            # Number of real decaying modes
            real_decay_count = 0

            for lam in eigvals:

                if (
                    abs(np.imag(lam)) < IMAG_TOL
                    and
                    np.real(lam) > EIG_MIN
                    and
                    np.real(lam) < EIG_MAX
                ):
                    real_decay_count += 1

            results.append({

                "rank": rank,

                "R2": R2,

                "RMSE": RMSE,

                "NRMSE": NRMSE,

                "energy": energy_at_rank,

                "real_decay_count": real_decay_count,

                "eigenvalues": eigvals,

                "modes": modes,

                "amplitudes": amplitudes,

                "singular_values": singular_values,

                "reconstruction": y_rec

            })

        except Exception as exc:

            print(
                f"Rank {rank} failed: {exc}"
            )

    return results


# =============================================================================
# AUTOMATIC RANK SELECTION
# =============================================================================

def select_rank(rank_results):
    """
    Select the smallest rank providing a good reconstruction.

    Priority:
        1. R2 >= TARGET_R2
        2. lowest rank
        3. otherwise highest R2
    """

    if len(rank_results) == 0:
        raise RuntimeError(
            "No valid DMD rank was obtained."
        )

    good = [
        r for r in rank_results
        if (
            np.isfinite(r["R2"])
            and
            r["R2"] >= TARGET_R2
        )
    ]

    if len(good) > 0:

        # Smallest rank with acceptable reconstruction
        selected = sorted(
            good,
            key=lambda x: x["rank"]
        )[0]

        return selected

    # If no rank reaches the target,
    # choose the rank with highest R2
    selected = max(
        rank_results,
        key=lambda x: (
            x["R2"]
            if np.isfinite(x["R2"])
            else -np.inf
        )
    )

    return selected


# =============================================================================
# CHARACTERISTIC RATE EXTRACTION
# =============================================================================

def extract_characteristic_rates(
    eigenvalues,
    amplitudes,
    dt
):
    """
    Extract physically interpretable characteristic
    decay rates.

    Only real, positive, decaying eigenvalues are used.

    For a discrete DMD eigenvalue lambda:

        lambda = exp(mu * dt)

    therefore:

        mu = log(lambda) / dt

    For decay:

        mu < 0

    Characteristic rate:

        k = -mu

    Only modes with negligible imaginary part are
    considered kinetic decay modes.

    Complex modes remain part of the reconstruction,
    but are NOT interpreted as independent kinetic
    compartments.
    """

    candidates = []

    if len(amplitudes) > 0:

        max_amp = np.max(
            np.abs(amplitudes)
        )

    else:
        max_amp = 1.0

    for lam, amp in zip(
        eigenvalues,
        amplitudes
    ):

        lam_real = np.real(lam)
        lam_imag = np.imag(lam)

        # Ignore complex modes for characteristic rates
        if abs(lam_imag) > IMAG_TOL:
            continue

        # Must be a positive decaying discrete eigenvalue
        if not (
            EIG_MIN < lam_real < EIG_MAX
        ):
            continue

        # Avoid numerical issues
        if lam_real <= 0:
            continue

        relative_amp = (
            abs(amp) / max_amp
            if max_amp > 0
            else 0.0
        )

        if relative_amp < MIN_RELATIVE_AMPLITUDE:
            continue

        continuous_rate = (
            -np.log(lam_real) / dt
        )

        if continuous_rate <= 0:
            continue

        candidates.append({

            "eigenvalue": lam_real,

            "rate": continuous_rate,

            "half_life": half_life(
                continuous_rate
            ),

            "amplitude": abs(amp),

            "relative_amplitude": relative_amp

        })

    # Sort from slowest to fastest
    candidates = sorted(
        candidates,
        key=lambda x: x["rate"]
    )

    return candidates


# =============================================================================
# SELECT SLOW / INTERMEDIATE / FAST
# =============================================================================

def select_three_characteristic_rates(
    candidates
):
    """
    Select up to three characteristic rates.

    Slow:
        smallest positive decay rate

    Intermediate:
        middle rate

    Fast:
        largest rate

    If fewer than three physically interpretable
    real modes exist, missing values are NaN.
    """

    selected = candidates.copy()

    # Remove near-duplicate rates
    filtered = []

    for item in selected:

        if len(filtered) == 0:

            filtered.append(item)

        else:

            previous = filtered[-1]["rate"]

            current = item["rate"]

            # Consider rates essentially identical
            # if relative difference is < 1%
            if (
                abs(current - previous)
                / max(previous, 1e-12)
                < 0.01
            ):

                # Keep the stronger mode
                if (
                    item["amplitude"]
                    >
                    filtered[-1]["amplitude"]
                ):
                    filtered[-1] = item

            else:

                filtered.append(item)

    selected = filtered

    # If more than 3 modes exist,
    # retain the slowest, an intermediate,
    # and the fastest.
    if len(selected) > 3:

        selected = [
            selected[0],
            selected[len(selected) // 2],
            selected[-1]
        ]

    output = {

        "slow_rate": np.nan,

        "intermediate_rate": np.nan,

        "fast_rate": np.nan,

        "slow_half_life": np.nan,

        "intermediate_half_life": np.nan,

        "fast_half_life": np.nan

    }

    if len(selected) >= 1:

        output["slow_rate"] = selected[0]["rate"]

        output["slow_half_life"] = selected[0]["half_life"]

    if len(selected) >= 2:

        output["intermediate_rate"] = selected[1]["rate"]

        output["intermediate_half_life"] = selected[1]["half_life"]

    if len(selected) >= 3:

        output["fast_rate"] = selected[2]["rate"]

        output["fast_half_life"] = selected[2]["half_life"]

    return output


# =============================================================================
# PRINT RANK TEST
# =============================================================================

def print_rank_table(
    region,
    rank_results
):

    print()
    print(
        f"RANK TESTING: {region.upper()}"
    )

    print(
        "-" * 75
    )

    print(
        f"{'Rank':>6}"
        f"{'R²':>12}"
        f"{'NRMSE':>12}"
        f"{'Energy':>12}"
        f"{'Real decay modes':>20}"
    )

    print(
        "-" * 75
    )

    for result in rank_results:

        print(
            f"{result['rank']:>6}"
            f"{result['R2']:>12.6f}"
            f"{result['NRMSE']:>12.6f}"
            f"{result['energy']:>12.6f}"
            f"{result['real_decay_count']:>20}"
        )


# =============================================================================
# PLOT RANK VALIDATION
# =============================================================================

def plot_rank_validation(
    region,
    rank_results,
    selected_rank
):

    ranks = [
        r["rank"]
        for r in rank_results
    ]

    R2 = [
        r["R2"]
        for r in rank_results
    ]

    NRMSE = [
        r["NRMSE"]
        for r in rank_results
    ]

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        ranks,
        R2,
        marker="o",
        linewidth=2,
        label=r"$R^2$"
    )

    plt.axhline(
        TARGET_R2,
        linestyle="--",
        linewidth=1.5,
        label=f"Target R² = {TARGET_R2}"
    )

    plt.axvline(
        selected_rank,
        linestyle=":",
        linewidth=2,
        label=f"Selected rank = {selected_rank}"
    )

    plt.xlabel(
        "Hankel-DMD rank"
    )

    plt.ylabel(
        r"$R^2$"
    )

    plt.title(
        f"{region.replace('_', ' ').title()} — Rank Validation"
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    filename = os.path.join(
        FIGURE_DIR,
        f"dmd_rank_validation_{region}.png"
    )

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    # NRMSE
    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        ranks,
        NRMSE,
        marker="o",
        linewidth=2
    )

    plt.axvline(
        selected_rank,
        linestyle=":",
        linewidth=2,
        label=f"Selected rank = {selected_rank}"
    )

    plt.xlabel(
        "Hankel-DMD rank"
    )

    plt.ylabel(
        "Normalized RMSE"
    )

    plt.title(
        f"{region.replace('_', ' ').title()} — Reconstruction Error"
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    filename = os.path.join(
        FIGURE_DIR,
        f"dmd_rank_nrmse_{region}.png"
    )

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# =============================================================================
# PLOT SINGULAR VALUES
# =============================================================================

def plot_singular_values(
    singular_results,
    region_names
):

    plt.figure(
        figsize=(12, 8)
    )

    for region in region_names:

        S = singular_results[region]

        x = np.arange(
            1,
            len(S) + 1
        )

        plt.semilogy(
            x,
            S,
            marker="o",
            linewidth=2,
            label=region.replace(
                "_",
                " "
            ).title()
        )

    plt.xlabel(
        "Singular-value index"
    )

    plt.ylabel(
        "Singular value"
    )

    plt.title(
        "Hankel Singular Values"
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    filename = os.path.join(
        FIGURE_DIR,
        "dmd_validated_singular_values.png"
    )

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# =============================================================================
# PLOT TAC RECONSTRUCTIONS
# =============================================================================

def plot_reconstructions(
    results,
    region_names
):

    plt.figure(
        figsize=(12, 8)
    )

    for region in region_names:

        result = results[region]

        t = result["uniform_time"]

        y = result["uniform_tac"]

        y_rec = result["reconstruction"]

        plt.plot(
            t,
            y,
            linewidth=2,
            label=f"{region.replace('_', ' ').title()} TAC"
        )

        plt.plot(
            t[:len(y_rec)],
            y_rec,
            linestyle="--",
            linewidth=2,
            label=f"{region.replace('_', ' ').title()} DMD"
        )

    plt.xlabel(
        "Time (min)"
    )

    plt.ylabel(
        "Activity"
    )

    plt.title(
        "PET TACs and Validated Hankel-DMD Reconstruction"
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    filename = os.path.join(
        FIGURE_DIR,
        "dmd_validated_pet_tacs.png"
    )

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# =============================================================================
# PLOT CHARACTERISTIC RATES
# =============================================================================

def plot_characteristic_rates(
    results,
    region_names
):

    slow = []
    intermediate = []
    fast = []

    labels = []

    for region in region_names:

        labels.append(
            region.replace(
                "_",
                " "
            ).title()
        )

        slow.append(
            results[region]["slow_rate"]
        )

        intermediate.append(
            results[region]["intermediate_rate"]
        )

        fast.append(
            results[region]["fast_rate"]
        )

    x = np.arange(
        len(labels)
    )

    width = 0.25

    plt.figure(
        figsize=(10, 7)
    )

    plt.bar(
        x - width,
        slow,
        width,
        label="Slow"
    )

    plt.bar(
        x,
        intermediate,
        width,
        label="Intermediate"
    )

    plt.bar(
        x + width,
        fast,
        width,
        label="Fast"
    )

    plt.xticks(
        x,
        labels
    )

    plt.ylabel(
        "Characteristic rate (1/min)"
    )

    plt.title(
        "Validated Hankel-DMD Characteristic Rates"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    filename = os.path.join(
        FIGURE_DIR,
        "dmd_validated_characteristic_rates.png"
    )

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# =============================================================================
# MAIN PROGRAM
# =============================================================================

if __name__ == "__main__":

    print("=" * 90)
    print(
        "VALIDATED HANKEL-DMD PET KINETIC ANALYSIS"
    )
    print("=" * 90)

    print()
    print(
        "Loading:"
    )

    print(
        DATA_FILE
    )

    # -------------------------------------------------------------------------
    # LOAD DATA
    # -------------------------------------------------------------------------

    data = np.load(
        DATA_FILE,
        allow_pickle=True
    )

    print()
    print(
        "Files loaded successfully."
    )

    print()
    print(
        "Available input keys:"
    )

    print(
        list(data.keys())
    )

    # -------------------------------------------------------------------------
    # TIME INFORMATION
    # -------------------------------------------------------------------------

    if "frame_midtimes" not in data:
        raise KeyError(
            "frame_midtimes not found in input file."
        )

    if "Cp" not in data:
        raise KeyError(
            "Cp not found in input file."
        )

    frame_midtimes = np.asarray(
        data["frame_midtimes"],
        dtype=float
    )

    Cp = np.asarray(
        data["Cp"],
        dtype=float
    )

    print()
    print(
        "PET frame midtimes:"
    )

    print(
        f"Number of frames : {len(frame_midtimes)}"
    )

    print(
        f"Time range       : "
        f"{frame_midtimes[0]:.3f} - "
        f"{frame_midtimes[-1]:.3f} min"
    )

    # -------------------------------------------------------------------------
    # UNIFORM GRID
    # -------------------------------------------------------------------------

    region_names = [
        "grey_matter",
        "white_matter",
        "lesion"
    ]

    region_tacs = {}

    uniform_times = {}

    uniform_tacs = {}

    singular_results = {}

    final_results = {}

    # -------------------------------------------------------------------------
    # PROCESS EACH REGION
    # -------------------------------------------------------------------------

    for region in region_names:

        print()
        print("-" * 90)

        print(
            region.upper()
        )

        print("-" * 90)

        # Load TAC
        tac, tac_key = get_region_tac(
            data,
            region
        )

        print(
            f"TAC array       : {tac_key}"
        )

        print(
            f"TAC length      : {len(tac)}"
        )

        # -------------------------------------------------------------
        # Uniform interpolation
        # -------------------------------------------------------------

        t_uniform, y_uniform = make_uniform_tac(
            frame_midtimes,
            tac,
            DT
        )

        print()
        print(
            "Uniform DMD grid:"
        )

        print(
            f"Sampling interval : {DT:.6f} min"
        )

        print(
            f"Number of samples : {len(y_uniform)}"
        )

        print(
            f"Time range        : "
            f"{t_uniform[0]:.3f} - "
            f"{t_uniform[-1]:.3f} min"
        )

        region_tacs[region] = tac

        uniform_times[region] = t_uniform

        uniform_tacs[region] = y_uniform

        # -------------------------------------------------------------
        # Choose embedding
        # -------------------------------------------------------------

        embedding = min(
            MAX_EMBEDDING,
            max(
                10,
                len(y_uniform) // 5
            )
        )

        # Must remain smaller than number of samples
        embedding = min(
            embedding,
            len(y_uniform) - 2
        )

        print()
        print(
            f"Embedding dimension : {embedding}"
        )

        # -------------------------------------------------------------
        # Build Hankel matrix for singular values
        # -------------------------------------------------------------

        H0, H1 = build_hankel(
            y_uniform,
            embedding
        )

        _, S, _ = np.linalg.svd(
            H0,
            full_matrices=False
        )

        singular_results[region] = S

        # -------------------------------------------------------------
        # Rank testing
        # -------------------------------------------------------------

        rank_results = test_ranks(
            y_uniform,
            DT,
            embedding,
            MIN_RANK,
            MAX_RANK
        )

        print_rank_table(
            region,
            rank_results
        )

        # -------------------------------------------------------------
        # Select rank
        # -------------------------------------------------------------

        selected = select_rank(
            rank_results
        )

        selected_rank = selected["rank"]

        print()
        print(
            f"SELECTED DMD RANK = {selected_rank}"
        )

        print(
            f"Reconstruction R² = {selected['R2']:.6f}"
        )

        print(
            f"Reconstruction NRMSE = "
            f"{selected['NRMSE']:.6f}"
        )

        print(
            f"Captured singular-value energy = "
            f"{selected['energy']:.6f}"
        )

        # -------------------------------------------------------------
        # Eigenvalues
        # -------------------------------------------------------------

        eigenvalues = selected[
            "eigenvalues"
        ]

        amplitudes = selected[
            "amplitudes"
        ]

        print()
        print(
            "Selected DMD eigenvalues:"
        )

        for i, lam in enumerate(
            eigenvalues
        ):

            print(
                f"Mode {i + 1:2d}: "
                f"{lam.real:+.6f}"
                f"{lam.imag:+.6f}i"
            )

        # -------------------------------------------------------------
        # Continuous-time rates
        # -------------------------------------------------------------

        continuous_rates = np.log(
            eigenvalues.astype(complex)
        ) / DT

        print()
        print(
            "Continuous-time DMD rates (1/min):"
        )

        for i, rate in enumerate(
            continuous_rates
        ):

            print(
                f"Mode {i + 1:2d}: "
                f"{rate.real:+.6f}"
                f"{rate.imag:+.6f}i"
            )

        # -------------------------------------------------------------
        # Characteristic kinetic rates
        # -------------------------------------------------------------

        candidates = extract_characteristic_rates(
            eigenvalues,
            amplitudes,
            DT
        )

        kinetic = select_three_characteristic_rates(
            candidates
        )

        print()
        print(
            "PHYSICALLY INTERPRETABLE REAL DECAY MODES:"
        )

        if len(candidates) == 0:

            print(
                "No suitable real decay modes found."
            )

        else:

            for i, item in enumerate(
                candidates
            ):

                print(
                    f"Mode {i + 1}: "
                    f"lambda = "
                    f"{item['eigenvalue']:.8f}, "
                    f"rate = "
                    f"{item['rate']:.6f} 1/min, "
                    f"half-life = "
                    f"{item['half_life']:.3f} min, "
                    f"relative amplitude = "
                    f"{item['relative_amplitude']:.6e}"
                )

        print()
        print(
            "CHARACTERISTIC RATES"
        )

        print(
            f"Slow rate         : "
            f"{kinetic['slow_rate']:.6f} 1/min"
        )

        print(
            f"Slow half-life    : "
            f"{kinetic['slow_half_life']:.3f} min"
        )

        print(
            f"Intermediate rate : "
            f"{kinetic['intermediate_rate']:.6f} 1/min"
        )

        print(
            f"Intermediate half-life : "
            f"{kinetic['intermediate_half_life']:.3f} min"
        )

        print(
            f"Fast rate          : "
            f"{kinetic['fast_rate']:.6f} 1/min"
        )

        print(
            f"Fast half-life     : "
            f"{kinetic['fast_half_life']:.3f} min"
        )

        # -------------------------------------------------------------
        # Store
        # -------------------------------------------------------------

        final_results[region] = {

            "tac_key": tac_key,

            "original_tac": tac,

            "original_time": frame_midtimes,

            "uniform_time": t_uniform,

            "uniform_tac": y_uniform,

            "embedding": embedding,

            "selected_rank": selected_rank,

            "R2": selected["R2"],

            "RMSE": selected["RMSE"],

            "NRMSE": selected["NRMSE"],

            "energy": selected["energy"],

            "singular_values": selected[
                "singular_values"
            ],

            "eigenvalues": eigenvalues,

            "continuous_rates": continuous_rates,

            "amplitudes": amplitudes,

            "reconstruction": selected[
                "reconstruction"
            ],

            "candidate_rates": np.array(
                [
                    x["rate"]
                    for x in candidates
                ]
            ),

            "candidate_half_lives": np.array(
                [
                    x["half_life"]
                    for x in candidates
                ]
            ),

            "slow_rate": kinetic[
                "slow_rate"
            ],

            "intermediate_rate": kinetic[
                "intermediate_rate"
            ],

            "fast_rate": kinetic[
                "fast_rate"
            ],

            "slow_half_life": kinetic[
                "slow_half_life"
            ],

            "intermediate_half_life": kinetic[
                "intermediate_half_life"
            ],

            "fast_half_life": kinetic[
                "fast_half_life"
            ]

        }

        # -------------------------------------------------------------
        # Rank validation plots
        # -------------------------------------------------------------

        plot_rank_validation(
            region,
            rank_results,
            selected_rank
        )

    # =========================================================================
    # GLOBAL PLOTS
    # =========================================================================

    print()
    print("=" * 90)
    print(
        "GENERATING FINAL VALIDATION FIGURES"
    )
    print("=" * 90)

    plot_singular_values(
        singular_results,
        region_names
    )

    plot_reconstructions(
        final_results,
        region_names
    )

    plot_characteristic_rates(
        final_results,
        region_names
    )

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    print()
    print("=" * 100)

    print(
        "FINAL VALIDATED DMD SUMMARY"
    )

    print("=" * 100)

    print(
        f"{'Region':<18}"
        f"{'Rank':>8}"
        f"{'R²':>12}"
        f"{'NRMSE':>12}"
        f"{'Slow':>14}"
        f"{'Intermediate':>16}"
        f"{'Fast':>14}"
    )

    print(
        "-" * 100
    )

    for region in region_names:

        r = final_results[region]

        print(
            f"{region:<18}"
            f"{r['selected_rank']:>8}"
            f"{r['R2']:>12.6f}"
            f"{r['NRMSE']:>12.6f}"
            f"{r['slow_rate']:>14.6f}"
            f"{r['intermediate_rate']:>16.6f}"
            f"{r['fast_rate']:>14.6f}"
        )

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================

    save_dict = {}

    for region in region_names:

        r = final_results[region]

        prefix = region

        save_dict[
            f"{prefix}_tac"
        ] = r["original_tac"]

        save_dict[
            f"{prefix}_time"
        ] = r["original_time"]

        save_dict[
            f"{prefix}_uniform_time"
        ] = r["uniform_time"]

        save_dict[
            f"{prefix}_uniform_tac"
        ] = r["uniform_tac"]

        save_dict[
            f"{prefix}_reconstruction"
        ] = r["reconstruction"]

        save_dict[
            f"{prefix}_singular_values"
        ] = r["singular_values"]

        save_dict[
            f"{prefix}_eigenvalues"
        ] = r["eigenvalues"]

        save_dict[
            f"{prefix}_continuous_rates"
        ] = r["continuous_rates"]

        save_dict[
            f"{prefix}_amplitudes"
        ] = r["amplitudes"]

        save_dict[
            f"{prefix}_selected_rank"
        ] = r["selected_rank"]

        save_dict[
            f"{prefix}_embedding"
        ] = r["embedding"]

        save_dict[
            f"{prefix}_R2"
        ] = r["R2"]

        save_dict[
            f"{prefix}_RMSE"
        ] = r["RMSE"]

        save_dict[
            f"{prefix}_NRMSE"
        ] = r["NRMSE"]

        save_dict[
            f"{prefix}_energy"
        ] = r["energy"]

        save_dict[
            f"{prefix}_slow_rate"
        ] = r["slow_rate"]

        save_dict[
            f"{prefix}_intermediate_rate"
        ] = r["intermediate_rate"]

        save_dict[
            f"{prefix}_fast_rate"
        ] = r["fast_rate"]

        save_dict[
            f"{prefix}_slow_half_life"
        ] = r["slow_half_life"]

        save_dict[
            f"{prefix}_intermediate_half_life"
        ] = r["intermediate_half_life"]

        save_dict[
            f"{prefix}_fast_half_life"
        ] = r["fast_half_life"]

        save_dict[
            f"{prefix}_candidate_rates"
        ] = r["candidate_rates"]

        save_dict[
            f"{prefix}_candidate_half_lives"
        ] = r["candidate_half_lives"]

    np.savez(
        RESULT_FILE,
        **save_dict
    )

    # =========================================================================
    # FINISHED
    # =========================================================================

    print()
    print("=" * 90)

    print(
        "VALIDATED HANKEL-DMD ANALYSIS COMPLETED"
    )

    print("=" * 90)

    print()
    print(
        "Results saved to:"
    )

    print(
        RESULT_FILE
    )

    print()
    print(
        "Figures saved to:"
    )

    print(
        FIGURE_DIR
    )

    print()
    print(
        "Important:"
    )

    print(
        "DMD rank was selected using reconstruction quality."
    )

    print(
        "Complex modes are retained for reconstruction "
        "but are not automatically interpreted as kinetic "
        "characteristic rates."
    )

    print(
        "Characteristic rates are extracted only from "
        "real, positive, decaying DMD eigenvalues."
    )

    print()
    print("=" * 90)
