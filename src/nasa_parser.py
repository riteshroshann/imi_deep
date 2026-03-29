"""
nasa_parser.py — NASA PCoE CFRP Composites Full Dataset Parser
==============================================================
Parses the real NASA dataset from local disk into structured parquet/CSV files.
Run this once before training:

    python -m src.nasa_parser

Outputs in data/parsed/:
    pzt_waveforms.parquet       — Feature vectors per actuator-sensor path
    pzt_signals_raw.parquet     — Downsampled raw waveforms (subset)
    strain_data.parquet         — Strain gauge time-series
    experiment_log.parquet      — Excel log records
    xray_inventory.csv          — X-ray image inventory
    dataset_summary.json        — Parse statistics

Dataset citation:
    Saxena A., Goebel K., Larrosa C.C., Chang F-K.,
    NASA Ames PCoE, 2008–2013.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.io as sio

warnings.filterwarnings("ignore")

log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

DOWNSAMPLE_FACTOR = 4          # raw waveform storage downsampling
MAX_RAW_PATHS = 252            # all actuator-sensor paths

LAYUP_DIRS = ["Layup1", "Layup2", "Layup3"]

BOUNDARY_LABELS = {
    0: "baseline",
    1: "loaded",
    2: "clamped",
    3: "traction_free",
}

# ── Filename parser ──────────────────────────────────────────────────────────

def parse_pzt_filename(filename: str) -> Optional[Dict[str, Any]]:
    """
    Decode PZT .mat filename into structured metadata.

    Convention: L{layup}S{specimen}_{cycles}_{boundary_code}[_{rep}].mat
    where boundary_code:  0=baseline  1=loaded  2=clamped  3=traction-free
    """
    name = filename.replace(".mat", "")
    m = re.match(r"^L(\d)S(\d+)_(\d+)_(\d+)(?:_(\d+))?$", name)
    if not m:
        return None
    return {
        "layup": int(m.group(1)),
        "specimen": int(m.group(2)),
        "cycles": int(m.group(3)),
        "boundary_code": int(m.group(4)),
        "repetition": int(m.group(5)) if m.group(5) else 0,
    }


# ── .mat loader ──────────────────────────────────────────────────────────────

def safe_loadmat(filepath: str) -> Optional[dict]:
    """Load .mat file; falls back to h5py for v7.3 format."""
    try:
        return sio.loadmat(filepath, squeeze_me=False)
    except NotImplementedError:
        try:
            import h5py
            data: Dict[str, Any] = {}
            with h5py.File(filepath, "r") as f:
                for key in f.keys():
                    data[key] = np.array(f[key])
            return data
        except Exception as exc:
            log.debug("h5py load failed for %s: %s", filepath, exc)
            return None
    except Exception as exc:
        log.debug("scipy load failed for %s: %s", filepath, exc)
        return None


# ── Signal feature extraction ────────────────────────────────────────────────

def extract_signal_features(signal: np.ndarray) -> Dict[str, float]:
    """
    Time-domain + frequency-domain + Hilbert envelope features
    from a single Lamb wave sensor signal.
    """
    sig = signal.flatten().astype(np.float64)
    n = len(sig)
    if n == 0:
        return {}

    feats: Dict[str, float] = {}

    # ── Time domain ──
    feats["amplitude_max"] = float(np.max(sig))
    feats["amplitude_min"] = float(np.min(sig))
    feats["amplitude_pp"] = feats["amplitude_max"] - feats["amplitude_min"]
    feats["rms"] = float(np.sqrt(np.mean(sig ** 2)))
    feats["energy"] = float(np.sum(sig ** 2))
    feats["mean"] = float(np.mean(sig))
    feats["std"] = float(np.std(sig))
    feats["variance"] = float(np.var(sig))

    std = feats["std"]
    if std > 1e-12:
        centered = sig - feats["mean"]
        feats["skewness"] = float(np.mean(centered ** 3) / std ** 3)
        feats["kurtosis"] = float(np.mean(centered ** 4) / std ** 4)
    else:
        feats["skewness"] = 0.0
        feats["kurtosis"] = 0.0

    feats["zero_crossing_rate"] = float(
        np.sum(np.diff(np.sign(sig)) != 0)
    ) / max(n - 1, 1)

    # ── Frequency domain ──
    fft_mag = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(n)
    total_power = float(np.sum(fft_mag ** 2))

    if total_power > 1e-12:
        feats["dominant_frequency"] = float(freqs[np.argmax(fft_mag[1:]) + 1])
        feats["spectral_centroid"] = float(
            np.sum(freqs * fft_mag ** 2) / total_power
        )
        feats["spectral_bandwidth"] = float(
            np.sqrt(
                np.sum(
                    ((freqs - feats["spectral_centroid"]) ** 2) * fft_mag ** 2
                )
                / total_power
            )
        )
    else:
        feats["dominant_frequency"] = 0.0
        feats["spectral_centroid"] = 0.0
        feats["spectral_bandwidth"] = 0.0

    # ── Hilbert envelope ──
    try:
        from scipy.signal import hilbert
        envelope = np.abs(hilbert(sig))
        feats["envelope_max"] = float(np.max(envelope))
        feats["envelope_mean"] = float(np.mean(envelope))
    except Exception:
        feats["envelope_max"] = feats["amplitude_pp"]
        feats["envelope_mean"] = feats["rms"]

    # ── Time of arrival (10 % threshold) ──
    threshold = 0.1 * max(abs(feats["amplitude_max"]), 1e-12)
    crossings = np.where(np.abs(sig) > threshold)[0]
    feats["toa_index"] = int(crossings[0]) if len(crossings) > 0 else n
    feats["signal_length"] = n

    return feats


# ── PZT file parser ──────────────────────────────────────────────────────────

def parse_pzt_file(
    filepath: Path, meta: Dict[str, Any]
) -> Tuple[List[dict], List[dict]]:
    """
    Parse one PZT .mat file.

    The file contains a `coupon` struct:
        cycles, load, condition, comment,
        path_data (1 × N_paths) struct:
            actuator, sensor, amplitude, frequency, gain,
            signal_actuator, signal_sensor, sampling_rate

    Returns
    -------
    feature_rows : one dict per actuator-sensor path
    raw_rows     : downsampled waveform dicts (memory limited)
    """
    data = safe_loadmat(str(filepath))
    if data is None:
        return [], []

    feature_rows: List[dict] = []
    raw_rows: List[dict] = []

    try:
        coupon = data["coupon"]

        cycles = int(coupon["cycles"].flat[0].flat[0])
        load_val = float(coupon["load"].flat[0].flat[0])

        cond = coupon["condition"].flat[0]
        condition = str(cond.flat[0] if isinstance(cond, np.ndarray) else cond)

        try:
            c = coupon["comment"].flat[0]
            comment = str(c.flat[0] if isinstance(c, np.ndarray) else c)
        except Exception:
            comment = ""

        path_data = coupon["path_data"].flat[0]
        n_paths = path_data.shape[1] if path_data.ndim > 1 else path_data.shape[0]

        for pidx in range(n_paths):
            try:
                path = (
                    path_data[0, pidx] if path_data.ndim > 1 else path_data[pidx]
                )

                def _extract_val(k):
                    v = path[k]
                    if v.dtype == object and v.size == 1:
                        v = v.flat[0]
                    return np.asarray(v).flatten()

                actuator = int(_extract_val("actuator")[0])
                sensor = int(_extract_val("sensor")[0])
                amplitude = float(_extract_val("amplitude")[0])
                frequency = float(_extract_val("frequency")[0])
                gain = float(_extract_val("gain")[0])
                sampling_rate = float(_extract_val("sampling_rate")[0])
                sig_sensor = _extract_val("signal_sensor")

                feats = extract_signal_features(sig_sensor)

                row = {
                    "layup": meta["layup"],
                    "specimen": meta["specimen"],
                    "cycles": cycles,
                    "load_kips": load_val,
                    "condition": condition,
                    "boundary_code": meta["boundary_code"],
                    "boundary_label": BOUNDARY_LABELS.get(
                        meta["boundary_code"], f"unknown_{meta['boundary_code']}"
                    ),
                    "repetition": meta["repetition"],
                    "actuator": actuator,
                    "sensor": sensor,
                    "path_id": f"{actuator}-{sensor}",
                    "excitation_amplitude": amplitude,
                    "excitation_frequency_khz": frequency,
                    "gain": gain,
                    "sampling_rate_mhz": sampling_rate,
                    "filename": filepath.name,
                }
                row.update(feats)
                feature_rows.append(row)

                if pidx < MAX_RAW_PATHS:
                    sig_ds = sig_sensor[::DOWNSAMPLE_FACTOR]
                    raw_rows.append(
                        {
                            "layup": meta["layup"],
                            "specimen": meta["specimen"],
                            "cycles": cycles,
                            "actuator": actuator,
                            "sensor": sensor,
                            "path_id": f"{actuator}-{sensor}",
                            "frequency_khz": frequency,
                            "n_samples_original": len(sig_sensor),
                            "n_samples_downsampled": len(sig_ds),
                            "waveform": sig_ds.tolist(),
                        }
                    )

            except Exception as exc:
                log.debug("Path %d parse error in %s: %s", pidx, filepath.name, exc)

    except Exception as exc:
        log.warning("Coupon parse failed in %s: %s", filepath.name, exc)

    return feature_rows, raw_rows


# ── Strain parser ─────────────────────────────────────────────────────────────

def parse_strain_file(
    filepath: Path, specimen_id: str
) -> Optional[pd.DataFrame]:
    """Parse a strain-gauge .mat file."""
    data = safe_loadmat(str(filepath))
    if data is None:
        return None

    fname = filepath.stem
    parts = fname.split("_")
    if len(parts) < 3:
        return None

    layup_str, specimen_str, fatigue_step = parts[0], parts[1], parts[2]

    strain_type = "unknown"
    if "STRAIN_A" in fname:
        strain_type = "axial"
    elif "STRAIN_M" in fname:
        strain_type = "midplane"
    elif "STRAIN_S" in fname:
        strain_type = "surface"
    elif fname.endswith("_DAT"):
        strain_type = "mts_data"

    channels: Dict[str, np.ndarray] = {}
    for key, val in data.items():
        if key.startswith("__"):
            continue
        if isinstance(val, np.ndarray) and val.ndim <= 2:
            channels[key] = val.flatten()

    if not channels:
        return None

    max_len = max(len(v) for v in channels.values())
    aligned: Dict[str, np.ndarray] = {}
    for k, v in channels.items():
        if len(v) == max_len:
            aligned[k] = v
        else:
            padded = np.full(max_len, np.nan)
            padded[: len(v)] = v
            aligned[k] = padded

    df = pd.DataFrame(aligned)
    try:
        df["layup"] = int(layup_str[1])
        df["specimen"] = int(specimen_str[1:])
    except (IndexError, ValueError):
        df["layup"] = -1
        df["specimen"] = -1

    df["specimen_id"] = specimen_id
    df["fatigue_step"] = fatigue_step
    df["strain_type"] = strain_type
    df["filename"] = filepath.name
    df["sample_index"] = range(len(df))
    return df


# ── Experiment log parser ─────────────────────────────────────────────────────

def parse_experiment_log(
    xlsx_path: Path, specimen_id: str
) -> Optional[pd.DataFrame]:
    """Parse Excel log file into a uniform DataFrame."""
    try:
        xl = pd.ExcelFile(str(xlsx_path), engine="openpyxl")
    except Exception as exc:
        log.warning("Cannot open %s: %s", xlsx_path, exc)
        return None

    all_dfs: List[pd.DataFrame] = []
    for sheet in xl.sheet_names:
        try:
            df = xl.parse(sheet)
            if df.empty:
                continue

            if "Unnamed: 0" in df.columns:
                df.columns = [str(c).strip() for c in df.iloc[0].values]
                df = df.iloc[1:].reset_index(drop=True)

            col_map: Dict[str, str] = {}
            for col in df.columns:
                cl = str(col).lower().strip()
                if "date" in cl:
                    col_map[col] = "date"
                elif cl == "time" or ("time" in cl and "file" not in cl):
                    col_map[col] = "time"
                elif "record" in cl:
                    col_map[col] = "record_info"
                elif "cycle" in cl:
                    col_map[col] = "cycles"
                elif "load" in cl and "file" not in cl:
                    col_map[col] = "load_kips"
                elif "boundary" in cl or "condition" in cl:
                    col_map[col] = "boundary_condition"
                elif "data file" in cl:
                    col_map[col] = "data_filename"
                elif "mts" in cl:
                    col_map[col] = "mts_filename"
                elif "remark" in cl:
                    col_map[col] = "remarks"

            df = df.rename(columns=col_map)
            df["specimen_id"] = specimen_id
            df["sheet_name"] = sheet

            parts = specimen_id.split("_")
            try:
                df["layup"] = int(parts[0][1])
                df["specimen"] = int(parts[1][1:])
            except (IndexError, ValueError):
                df["layup"] = -1
                df["specimen"] = -1

            for col in ["cycles", "load_kips", "record_info"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            all_dfs.append(df)
        except Exception as exc:
            log.debug("Sheet '%s' skip: %s", sheet, exc)

    return pd.concat(all_dfs, ignore_index=True) if all_dfs else None


# ── X-ray inventory ───────────────────────────────────────────────────────────

def build_xray_inventory(
    specimen_dir: Path, specimen_id: str
) -> List[dict]:
    """Enumerate X-ray images and extract associated cycle counts."""
    xray_dir = specimen_dir / "XRay"
    if not xray_dir.exists():
        return []

    parts = specimen_id.split("_")
    try:
        layup = int(parts[0][1])
        specimen = int(parts[1][1:])
    except (IndexError, ValueError):
        layup, specimen = -1, -1

    rows: List[dict] = []
    for f in sorted(os.listdir(str(xray_dir))):
        if not f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
            continue
        fpath = xray_dir / f
        cycle_str = f.split(".")[0].split("_")[-1]
        if cycle_str.lower() == "baseline":
            cycles = 0
        elif cycle_str.isdigit():
            cycles = int(cycle_str)
        else:
            cycles = -1

        rows.append(
            {
                "layup": layup,
                "specimen": specimen,
                "specimen_id": specimen_id,
                "cycles": cycles,
                "filename": f,
                "filepath": str(fpath),
                "file_size_kb": os.path.getsize(str(fpath)) / 1024,
            }
        )
    return rows


# ── Main parse pipeline ───────────────────────────────────────────────────────

def run_parse(
    base_dir: Path,
    output_dir: Path,
    raw_store_every: int = 5,
) -> Dict[str, Any]:
    """
    Full parse of the NASA CFRP dataset.

    Parameters
    ----------
    base_dir        Path to "2. Composites" root
    output_dir      Directory to write parquet/CSV outputs
    raw_store_every Store raw waveforms for every N-th PZT file
                    (controls output size)

    Returns
    -------
    stats dict with parse summary
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    all_pzt_features: List[dict] = []
    all_pzt_raw: List[dict] = []
    all_strain: List[pd.DataFrame] = []
    all_logs: List[pd.DataFrame] = []
    all_xray: List[dict] = []

    stats: Dict[str, Any] = {
        "layups": {},
        "total_pzt_files": 0,
        "total_strain_files": 0,
        "total_xray_images": 0,
        "total_specimens": 0,
        "total_paths_parsed": 0,
        "parse_errors": 0,
    }

    for layup_name in LAYUP_DIRS:
        layup_dir = base_dir / layup_name
        if not layup_dir.exists():
            log.warning("%s not found, skipping.", layup_name)
            continue

        log.info("─" * 60)
        log.info("Processing %s", layup_name)

        specimens = sorted(
            d
            for d in os.listdir(str(layup_dir))
            if (layup_dir / d).is_dir() and d.startswith("L")
        )

        layup_stats: Dict[str, Any] = {
            "specimens": len(specimens),
            "pzt_files": 0,
            "strain_files": 0,
            "xray_images": 0,
        }

        for specimen_id in specimens:
            specimen_dir = layup_dir / specimen_id
            log.info("  Specimen: %s", specimen_id)
            stats["total_specimens"] += 1

            # ── PZT ──────────────────────────────────────────────────
            pzt_dir = specimen_dir / "PZT-data"
            if pzt_dir.exists():
                mat_files = sorted(
                    f
                    for f in os.listdir(str(pzt_dir))
                    if f.endswith(".mat") and not f.startswith(".")
                )
                log.info("    PZT: %d files", len(mat_files))
                layup_stats["pzt_files"] += len(mat_files)
                stats["total_pzt_files"] += len(mat_files)

                for i, fname in enumerate(mat_files):
                    meta = parse_pzt_filename(fname)
                    if meta is None:
                        log.debug("    Unparseable filename: %s", fname)
                        stats["parse_errors"] += 1
                        continue

                    feat_rows, raw_rows = parse_pzt_file(pzt_dir / fname, meta)
                    all_pzt_features.extend(feat_rows)

                    # Store raw waveform for baseline measurements, every
                    # raw_store_every-th file, and first/last few per specimen
                    if (
                        i % raw_store_every == 0
                        or meta["boundary_code"] == 0
                        or i < 3
                    ):
                        all_pzt_raw.extend(raw_rows)

                    stats["total_paths_parsed"] += len(feat_rows)

                    if (i + 1) % 50 == 0:
                        log.info(
                            "      %d/%d PZT files processed",
                            i + 1,
                            len(mat_files),
                        )

                log.info("    ✓ PZT done")

            # ── Strain ───────────────────────────────────────────────
            strain_dir = specimen_dir / "StrainData"
            if strain_dir.exists():
                strain_files = sorted(
                    f
                    for f in os.listdir(str(strain_dir))
                    if f.endswith(".mat") and not f.startswith(".")
                )
                log.info("    Strain: %d files", len(strain_files))
                layup_stats["strain_files"] += len(strain_files)
                stats["total_strain_files"] += len(strain_files)

                for fname in strain_files:
                    df = parse_strain_file(strain_dir / fname, specimen_id)
                    if df is not None:
                        all_strain.append(df)
                log.info("    ✓ Strain done")

            # ── Experiment log ────────────────────────────────────────
            for xf in os.listdir(str(specimen_dir)):
                if xf.endswith(".xlsx") and not xf.startswith("."):
                    df = parse_experiment_log(specimen_dir / xf, specimen_id)
                    if df is not None:
                        all_logs.append(df)
                        log.info(
                            "    ✓ Log: %d records (%s)", len(df), xf
                        )

            # ── X-ray ────────────────────────────────────────────────
            xray_rows = build_xray_inventory(specimen_dir, specimen_id)
            all_xray.extend(xray_rows)
            layup_stats["xray_images"] += len(xray_rows)
            stats["total_xray_images"] += len(xray_rows)
            if xray_rows:
                log.info("    ✓ X-Ray: %d images", len(xray_rows))

        stats["layups"][layup_name] = layup_stats

    # ── Persist outputs ───────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Assembling output files …")

    if all_pzt_features:
        df_pzt = (
            pd.DataFrame(all_pzt_features)
            .sort_values(["layup", "specimen", "cycles", "actuator", "sensor"])
            .reset_index(drop=True)
        )
        df_pzt.to_parquet(str(output_dir / "pzt_waveforms.parquet"), index=False)
        df_pzt.head(2000).to_csv(str(output_dir / "pzt_waveforms_sample.csv"), index=False)
        stats["pzt_features_shape"] = list(df_pzt.shape)
        stats["pzt_feature_columns"] = list(df_pzt.columns)
        stats["pzt_unique_cycles"] = sorted(
            int(v) for v in df_pzt["cycles"].unique()
        )
        log.info("  ✓ pzt_waveforms.parquet  %s", df_pzt.shape)

    if all_pzt_raw:
        df_raw = pd.DataFrame(all_pzt_raw)
        df_raw.to_parquet(str(output_dir / "pzt_signals_raw.parquet"), index=False)
        stats["pzt_raw_shape"] = list(df_raw.shape)
        log.info("  ✓ pzt_signals_raw.parquet  %s", df_raw.shape)

    if all_strain:
        df_strain = pd.concat(all_strain, ignore_index=True)
        df_strain.to_parquet(str(output_dir / "strain_data.parquet"), index=False)
        df_strain.head(2000).to_csv(str(output_dir / "strain_data_sample.csv"), index=False)
        stats["strain_shape"] = list(df_strain.shape)
        log.info("  ✓ strain_data.parquet  %s", df_strain.shape)

    if all_logs:
        df_logs = pd.concat(all_logs, ignore_index=True)
        # Coerce all object columns to str so PyArrow can serialise mixed types
        for col in df_logs.select_dtypes(include="object").columns:
            df_logs[col] = df_logs[col].astype(str)
        try:
            df_logs.to_parquet(str(output_dir / "experiment_log.parquet"), index=False)
        except Exception as exc:
            log.warning("experiment_log parquet write failed (%s); CSV only.", exc)
        df_logs.to_csv(str(output_dir / "experiment_log.csv"), index=False)
        stats["experiment_log_shape"] = list(df_logs.shape)
        log.info("  ✓ experiment_log  %s", df_logs.shape)

    if all_xray:
        df_xray = pd.DataFrame(all_xray)
        df_xray.to_csv(str(output_dir / "xray_inventory.csv"), index=False)
        stats["xray_inventory_shape"] = list(df_xray.shape)
        log.info("  ✓ xray_inventory.csv  %s", df_xray.shape)

    with open(str(output_dir / "dataset_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, default=str)

    log.info("=" * 60)
    log.info("PARSE COMPLETE")
    log.info("  Specimens      : %d", stats["total_specimens"])
    log.info("  PZT files      : %d", stats["total_pzt_files"])
    log.info("  Strain files   : %d", stats["total_strain_files"])
    log.info("  X-Ray images   : %d", stats["total_xray_images"])
    log.info("  Paths parsed   : %d", stats["total_paths_parsed"])
    log.info("  Parse errors   : %d", stats["parse_errors"])
    log.info("  Output dir     : %s", output_dir)
    return stats


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    parser = argparse.ArgumentParser(description="NASA CFRP dataset parser")
    parser.add_argument(
        "--base_dir",
        type=str,
        default=r"C:\Users\rites\Downloads\2.+Composites\2. Composites",
        help="Path to '2. Composites' root directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/parsed",
        help="Output directory for parquet/CSV files",
    )
    parser.add_argument(
        "--raw_every",
        type=int,
        default=5,
        help="Store raw waveform for every N-th PZT file (controls disk usage)",
    )
    args = parser.parse_args()

    run_parse(
        base_dir=Path(args.base_dir),
        output_dir=Path(args.output_dir),
        raw_store_every=args.raw_every,
    )


if __name__ == "__main__":
    main()
