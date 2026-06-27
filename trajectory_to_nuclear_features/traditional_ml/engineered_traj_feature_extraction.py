"""
Engineered trajectory feature extraction pipeline.

Expected folder layout (all paths are relative to this script's directory):

    traditional_ml/
    ├── engineered_traj_feature_extraction.py   ← this script
    ├── data/
    │   ├── traj_copied/            trajectory CSVs with Nuc# prefixes + Nuc_number_mapping.csv
    │   │                           (produced externally, e.g. by trajectory_process.py)
    │   ├── master_nucleus_features.csv         nucleus morphology table (from segmentation)
    │   └── master_locus_features_clean.csv     per-locus spatial features (from detection)
    └── results/
        ├── combined_extracted_features.csv
        ├── combined_extracted_features_normalized.csv
        ├── combined_loci_model_input_output.csv
        │       ↑ inspect and manually clean outliers/bad rows, then save as:
        └── combined_loci_model_input_output_clean.csv

Steps run in order:
    1. extract_trajectory_features   — compute motion features per locus; save combined CSV
    2. normalize_features            — merge with nucleus area; add size-normalized features
    3. merge_locus_features          — join with per-locus spatial measurements
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import welch
from scipy.spatial import ConvexHull

try:
    import pywt
except ImportError:
    pywt = None

# ---------------------------------------------------------------------------
# Paths — edit HERE if your layout differs; everything else is derived below
# ---------------------------------------------------------------------------
HERE = Path(__file__).parent

TRAJ_DIR             = HERE / "data" / "traj_copied"        # renamed trajectory CSVs + mapping
RESULTS_DIR          = HERE / "results"
NUCLEUS_FEATURES_CSV = HERE / "data" / "master_nucleus_features.csv"
LOCUS_FEATURES_CSV   = HERE / "data" / "master_locus_features_clean.csv"

LOCUS_PATTERN  = "*G_loci*"   # glob pattern for per-locus CSVs in traj_copied
FRAME_INTERVAL = 1.0           # seconds per frame; used for PSD frequency axis


# ===========================================================================
# Feature extraction helpers
# ===========================================================================

def safe_var(x):
    return np.var(x, ddof=1) if len(x) > 1 else np.nan


def convex_hull_area(xy):
    # convex hull area measures the spatial territory explored by the DNA locus
    if len(xy) < 3:
        return np.nan
    try:
        return ConvexHull(xy).volume  # in 2D, "volume" == area
    except Exception:
        return np.nan


def radius_of_gyration(xy):
    center = xy.mean(axis=0)
    return np.sqrt(np.mean(np.sum((xy - center) ** 2, axis=1)))


def turning_angles(steps):
    if len(steps) < 2:
        return np.array([])
    angles = []
    for a, b in zip(steps[:-1], steps[1:]):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            continue
        angles.append(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1, 1)))
    return np.array(angles)


def autocorrelation_1d(signal, max_lag=None):
    signal = np.asarray(signal, dtype=float)
    signal = signal[~np.isnan(signal)]
    if len(signal) < 3:
        return np.array([])
    signal = signal - np.mean(signal)
    denom = np.dot(signal, signal)
    if denom == 0:
        return np.array([])
    if max_lag is None:
        max_lag = len(signal) // 2
    acf = []
    for lag in range(1, max_lag + 1):
        acf.append(np.dot(signal[:-lag], signal[lag:]) / denom)
    return np.array(acf)


def step_vector_autocorrelation(steps, max_lag=None):
    steps = np.asarray(steps, dtype=float)
    if len(steps) < 3:
        return np.array([])
    norms = np.linalg.norm(steps, axis=1)
    valid = norms > 0
    unit_steps = steps[valid] / norms[valid, None]
    if len(unit_steps) < 3:
        return np.array([])
    if max_lag is None:
        max_lag = len(unit_steps) // 2
    acf = []
    for lag in range(1, max_lag + 1):
        dots = np.sum(unit_steps[:-lag] * unit_steps[lag:], axis=1)
        acf.append(np.mean(dots))
    return np.array(acf)


def estimate_autocorr_decay(acf):
    if len(acf) < 3:
        return np.nan
    lags = np.arange(1, len(acf) + 1)
    valid = acf > 0
    if np.sum(valid) < 3:
        return np.nan
    try:
        def exp_decay(x, tau):
            return np.exp(-x / tau)
        popt, _ = curve_fit(exp_decay, lags[valid], acf[valid], p0=[2.0])
        return popt[0]
    except Exception:
        return np.nan


def estimate_persistence_length(steps):
    acf = step_vector_autocorrelation(steps)
    if len(acf) < 3:
        return np.nan
    step_sizes = np.linalg.norm(steps, axis=1)
    mean_step = np.nanmean(step_sizes)
    if mean_step == 0 or np.isnan(mean_step):
        return np.nan
    lags = np.arange(1, len(acf) + 1)
    s = lags * mean_step
    valid = acf > 0
    if np.sum(valid) < 3:
        return np.nan
    try:
        def exp_decay(x, Lp):
            return np.exp(-x / Lp)
        popt, _ = curve_fit(exp_decay, s[valid], acf[valid], p0=[mean_step])
        return popt[0]
    except Exception:
        return np.nan


def frequency_features(signal, frame_interval=1.0):
    signal = np.asarray(signal, dtype=float)
    signal = signal[~np.isnan(signal)]
    if len(signal) < 4:
        return {"psd_total_power": np.nan, "psd_peak_frequency": np.nan, "psd_spectral_centroid": np.nan}
    fs = 1.0 / frame_interval
    nperseg = min(len(signal), 8)
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    total_power = np.trapz(psd, freqs)
    peak_freq = freqs[np.argmax(psd)] if len(psd) > 0 else np.nan
    spectral_centroid = np.sum(freqs * psd) / np.sum(psd) if np.sum(psd) > 0 else np.nan
    return {"psd_total_power": total_power, "psd_peak_frequency": peak_freq, "psd_spectral_centroid": spectral_centroid}


def wavelet_features(signal, wavelet="db2", level=2):
    empty = {"wavelet_energy_total": np.nan, "wavelet_energy_approx": np.nan, "wavelet_energy_detail": np.nan}
    if pywt is None:
        return empty
    signal = np.asarray(signal, dtype=float)
    signal = signal[~np.isnan(signal)]
    if len(signal) < 4:
        return empty
    max_level = pywt.dwt_max_level(len(signal), pywt.Wavelet(wavelet).dec_len)
    level = min(level, max_level)
    if level < 1:
        return empty
    coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level)
    energies = [np.sum(c ** 2) for c in coeffs]
    return {
        "wavelet_energy_total": np.sum(energies),
        "wavelet_energy_approx": energies[0],
        "wavelet_energy_detail": np.sum(energies[1:]),
    }


def extract_features_one_trajectory(df, frame_interval=FRAME_INTERVAL):
    df = df.sort_values("frame").copy()
    xy = df[["x_nm", "y_nm"]].to_numpy(float)
    x = xy[:, 0]
    y = xy[:, 1]
    steps = np.diff(xy, axis=0)
    step_sizes = np.linalg.norm(steps, axis=1)

    total_path_length = np.sum(step_sizes)
    net_displacement = np.linalg.norm(xy[-1] - xy[0]) if len(xy) > 1 else np.nan
    straightness_index = net_displacement / total_path_length if total_path_length > 0 else np.nan

    angles = turning_angles(steps)
    speed_acf = autocorrelation_1d(step_sizes)
    vector_acf = step_vector_autocorrelation(steps)
    angle_acf = autocorrelation_1d(angles)
    angle_acf_tau = estimate_autocorr_decay(angle_acf)

    angle_freq_features = frequency_features(angles, frame_interval=frame_interval)
    angle_wavelet_features = wavelet_features(angles)

    angle_acf = autocorrelation_1d(angles)
    angle_acf_tau = estimate_autocorr_decay(angle_acf)

    features = {
        "n_frames": len(df),
        "x_variance_nm2": safe_var(x),
        "y_variance_nm2": safe_var(y),
        "convex_hull_area_nm2": convex_hull_area(xy),
        "radius_of_gyration_nm": radius_of_gyration(xy),
        "mean_step_size_nm": np.mean(step_sizes) if len(step_sizes) else np.nan,
        "max_step_size_nm": np.max(step_sizes) if len(step_sizes) else np.nan,
        "displacement_variance_nm2": safe_var(step_sizes),
        "total_path_length_nm": total_path_length,
        "net_displacement_nm": net_displacement,
        "straightness_index": straightness_index,
        "turning_angle_mean": np.mean(angles) if len(angles) else np.nan,
        "turning_angle_var": safe_var(angles),
        "turning_angle_median": np.median(angles) if len(angles) else np.nan,
        "turning_angle_acf_lag1": angle_acf[0] if len(angle_acf) >= 1 else np.nan,
        "turning_angle_acf_lag2": angle_acf[1] if len(angle_acf) >= 2 else np.nan,
        "turning_angle_acf_decay_tau": angle_acf_tau,
        "speed_autocorr_lag1": speed_acf[0] if len(speed_acf) >= 1 else np.nan,
        "vector_autocorr_lag1": vector_acf[0] if len(vector_acf) >= 1 else np.nan,
        "autocorr_decay_tau": estimate_autocorr_decay(speed_acf),
        "persistence_length": estimate_persistence_length(steps),
        "turning_angle_acf_lag1": angle_acf[0] if len(angle_acf) >= 1 else np.nan,
        "turning_angle_acf_lag2": angle_acf[1] if len(angle_acf) >= 2 else np.nan,
        "turning_angle_autocorr_decay_tau": angle_acf_tau,
    }

    step_freq_features = frequency_features(step_sizes, frame_interval=frame_interval)
    features.update({f"step_size_{k}": v for k, v in step_freq_features.items()})

    step_wavelet_features = wavelet_features(step_sizes)
    features.update({f"step_size_{k}": v for k, v in step_wavelet_features.items()})

    features.update({f"turning_angle_{k}": v for k, v in angle_freq_features.items()})
    features.update({f"turning_angle_{k}": v for k, v in angle_wavelet_features.items()})

    return pd.Series(features)


def get_nucleus_id(csv_path, mapping_df):
    filename = Path(csv_path).name
    matches = mapping_df.loc[mapping_df["new_filename"] == filename, "source_subsub_directory"]
    return matches.iloc[0] if len(matches) else np.nan


def get_locus_id(csv_path):
    filename = Path(csv_path).name
    match = re.search(r"(G_loci\d+)", filename)
    return match.group(1) if match else np.nan


def extract_features_from_file(csv_path, output_path=None, frame_interval=FRAME_INTERVAL,
                                track_col=None, mapping_df=None):
    df = pd.read_csv(csv_path)
    if track_col is not None and track_col in df.columns:
        features = (
            df.groupby(track_col)
            .apply(lambda g: extract_features_one_trajectory(g, frame_interval))
            .reset_index()
        )
    else:
        features = extract_features_one_trajectory(df, frame_interval).to_frame().T
    nucleus_id = get_nucleus_id(csv_path, mapping_df) if mapping_df is not None else np.nan
    locus_id = get_locus_id(csv_path)
    if isinstance(features, pd.Series):
        features = features.to_frame().T
    features.insert(0, "locus_id", locus_id)
    features.insert(0, "nucleus_id", nucleus_id)
    if output_path is not None:
        features.to_csv(output_path, index=False)
    return features


# ===========================================================================
# Step 1: extract features from all locus CSVs
# ===========================================================================

def extract_trajectory_features(
    traj_dir=TRAJ_DIR,
    results_dir=RESULTS_DIR,
    locus_pattern=LOCUS_PATTERN,
    frame_interval=FRAME_INTERVAL,
):
    results_dir.mkdir(parents=True, exist_ok=True)
    mapping_df = pd.read_csv(traj_dir / "Nuc_number_mapping.csv")
    combined_feature_rows = []

    for csv_path in sorted(traj_dir.rglob(locus_pattern)):
        if not csv_path.is_file() or csv_path.name.startswith("."):
            continue
        features = extract_features_from_file(
            csv_path, output_path=None, frame_interval=frame_interval,
            track_col=None, mapping_df=mapping_df,
        )
        combined_feature_rows.append(features)
        print(f"Queued {csv_path.name} ({len(features)} rows)")

    combined_features = pd.concat(combined_feature_rows, ignore_index=True)
    out = results_dir / "combined_extracted_features.csv"
    combined_features.to_csv(out, index=False)
    print(f"Saved combined features to: {out} ({len(combined_features)} rows)")
    return combined_features


# ===========================================================================
# Step 2: normalize by nucleus size
# ===========================================================================

def normalize_features(
    results_dir=RESULTS_DIR,
    nucleus_features_csv=NUCLEUS_FEATURES_CSV,
):
    traj_feature = pd.read_csv(results_dir / "combined_extracted_features.csv")
    nuc_feature  = pd.read_csv(nucleus_features_csv)

    nuc_feature_uniq = (
        nuc_feature.loc[nuc_feature.groupby("nucleus_id")["frame"].idxmin()]
        .reset_index(drop=True)
    )

    combined_normalized = traj_feature.merge(
        nuc_feature_uniq, on="nucleus_id", how="left", suffixes=("", "_nuc")
    )

    combined_normalized["convex_hull_area_normalized_nuc_area_10_neg_6"] = (
        combined_normalized["convex_hull_area_nm2"] / combined_normalized["area_um2"]
    )
    combined_normalized["convex_hull_area_normalized_nuc_area_frames_10_neg_6"] = (
        combined_normalized["convex_hull_area_normalized_nuc_area_10_neg_6"] / combined_normalized["n_frames"]
    )
    combined_normalized["radius_of_gyration_normalized_area_10_neg_3"] = (
        combined_normalized["radius_of_gyration_nm"] / np.sqrt(combined_normalized["area_um2"])
    )

    out = results_dir / "combined_extracted_features_normalized.csv"
    combined_normalized.to_csv(out, index=False)
    print(f"Wrote combined_extracted_features_normalized.csv with {len(combined_normalized)} rows")
    return combined_normalized


# ===========================================================================
# Step 3: merge per-locus spatial features
# ===========================================================================

def merge_locus_features(
    results_dir=RESULTS_DIR,
    locus_features_csv=LOCUS_FEATURES_CSV,
    normalized_csv_name="combined_extracted_features_normalized_clean.csv",
):
    """
    Reads the manually cleaned normalized CSV, joins with per-locus spatial
    measurements, and writes combined_loci_model_input_output.csv.

    NOTE: before running this step, inspect
      results/combined_extracted_features_normalized.csv,
    remove bad rows / outliers, and save the result as
      results/combined_extracted_features_normalized_clean.csv.
    """
    traj_norm     = pd.read_csv(results_dir / normalized_csv_name)
    locus_feature = pd.read_csv(locus_features_csv)

    locus_feature_agg = (
        locus_feature
        .groupby(["nucleus_id", "locus_id"], as_index=False)
        .mean(numeric_only=True)
    )

    combined_loci = traj_norm.merge(
        locus_feature_agg, on=["nucleus_id", "locus_id"], how="left", suffixes=("", "_locus")
    )

    out = results_dir / "combined_loci_model_input_output.csv"
    combined_loci.to_csv(out, index=False)
    print(f"Wrote combined_loci_model_input_output.csv with {len(combined_loci)} rows")
    return combined_loci


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    print("=== Step 1: extract trajectory features ===")
    extract_trajectory_features()

    print("\n=== Step 2: normalize by nucleus area ===")
    normalize_features()

    # --- manual step ---
    # Inspect results/combined_extracted_features_normalized.csv,
    # remove bad rows / outliers, and save as:
    #   results/combined_extracted_features_normalized_clean.csv
    # before running step 3.

    print("\n=== Step 3: merge locus spatial features ===")
    merge_locus_features()
