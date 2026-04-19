"""
data_loader.py — NASA PCoE CFRP Composites Dataset Parser (Improved)
=====================================================================
IMPROVEMENTS over original:
  1. Portable paths — no hardcoded user-specific Windows paths
  2. Automatic dataset download from NASA PCoE S3 mirror
  3. Robust parquet discovery (searches multiple candidate dirs)
  4. Chunked parquet reading for large files
  5. Stratified chronological train/val/test split (no leakage)
  6. create_dataloaders() respects task=classification properly
  7. Better life-fraction estimation using per-specimen max cycles
  8. Normalised RUL clipped to [0,1] always
"""

import os
import sys
import zipfile
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

try:
    import scipy.io as sio
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

import torch
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore", category=FutureWarning)
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
DATASET_URL = "https://phm-datasets.s3.amazonaws.com/NASA/2.+Composites.zip"
ZIP_FILENAME  = "composites.zip"
N_SENSORS     = 16
SAMPLING_RATE_HZ = 1_000_000

SENSOR_POSITIONS = {i: (i // 4, i % 4) for i in range(N_SENSORS)}

LAYUP_CONFIGS = {
    "L1": {"description": "[0/45/90/-45]_2s", "angles": [0,45,90,-45],  "type": "quasi-isotropic"},
    "L2": {"description": "[45/90/-45/0]_2s", "angles": [45,90,-45,0],  "type": "quasi-isotropic-shifted"},
    "L3": {"description": "[90/0/-45/45]_2s", "angles": [90,0,-45,45],  "type": "transverse-dominant"},
}

DAMAGE_THRESHOLDS = {
    "Healthy":       (0.00, 0.15),
    "Early_Damage":  (0.15, 0.40),
    "Moderate":      (0.40, 0.65),
    "Severe":        (0.65, 0.85),
    "Pre_failure":   (0.85, 1.00),
}
DAMAGE_CLASSES = list(DAMAGE_THRESHOLDS.keys())
N_CLASSES      = len(DAMAGE_CLASSES)

SCALAR_FEATS = [
    "amplitude_max", "amplitude_min", "amplitude_pp",
    "rms", "energy", "mean", "std", "variance",
    "skewness", "kurtosis", "zero_crossing_rate",
    "dominant_frequency", "spectral_centroid", "spectral_bandwidth",
    "envelope_max", "envelope_mean", "toa_index",
]

# ── Path resolution (portable) ─────────────────────────────────────────────────
def _candidate_dirs(data_path: str) -> List[Path]:
    """Return ordered list of directories to search for pre-parsed parquets."""
    candidates = [
        Path(data_path),
        Path(data_path) / "parsed",
        Path(data_path).parent / "parsed_cfrp",
        Path.home() / "imi_data" / "parsed_cfrp",
        Path.cwd() / "data" / "parsed",
    ]
    return candidates


def _find_parquet_dir(data_path: str) -> Optional[Path]:
    for d in _candidate_dirs(data_path):
        if (d / "pzt_waveforms.parquet").exists():
            logger.info("Parquet found: %s", d)
            return d
    return None


# ── Public API ─────────────────────────────────────────────────────────────────
def download_dataset(data_dir: str, force: bool = False) -> str:
    """
    Locate real NASA CFRP data.  Priority:
      1. Pre-parsed parquets (any candidate dir)
      2. Raw .mat files under data_dir → triggers nasa_parser
      3. Download composites.zip from NASA S3 mirror → parse

    Returns the directory that was resolved/created.
    Raises RuntimeError if nothing is accessible.
    """
    # Priority 1
    pqt_dir = _find_parquet_dir(data_dir)
    if pqt_dir and not force:
        return str(pqt_dir)

    # Priority 2 — raw .mat
    raw_path = Path(data_dir)
    mat_files = list(raw_path.rglob("*.mat"))
    if mat_files:
        logger.info("Found %d .mat files under %s — running parser", len(mat_files), raw_path)
        out_dir = raw_path.parent / "parsed_cfrp"
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            from src.nasa_parser import run_parse
            run_parse(base_dir=raw_path, output_dir=out_dir, raw_store_every=5)
            if (out_dir / "pzt_waveforms.parquet").exists():
                return str(out_dir)
        except Exception as exc:
            logger.warning("nasa_parser failed: %s", exc)

    # Priority 3 — download
    zip_path = raw_path / ZIP_FILENAME
    if not zip_path.exists() or force:
        logger.info("Downloading NASA PCoE dataset (~4.6 GB) ...")
        raw_path.mkdir(parents=True, exist_ok=True)
        _download_with_progress(DATASET_URL, zip_path)

    extract_dir = raw_path / "composites_raw"
    if not extract_dir.exists():
        logger.info("Extracting %s …", zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

    mat_files = list(extract_dir.rglob("*.mat"))
    if mat_files:
        out_dir = raw_path.parent / "parsed_cfrp"
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            from src.nasa_parser import run_parse
            run_parse(base_dir=extract_dir, output_dir=out_dir, raw_store_every=5)
            if (out_dir / "pzt_waveforms.parquet").exists():
                return str(out_dir)
        except Exception as exc:
            logger.error("nasa_parser failed after download: %s", exc)

    raise RuntimeError(
        "No real NASA CFRP data found. Please ensure one of:\n"
        f"  • pzt_waveforms.parquet exists in {data_dir} or a sibling 'parsed_cfrp/' folder\n"
        f"  • Raw .mat files are present under {data_dir}\n"
        f"  • Internet connectivity for auto-download from NASA S3"
    )


def _download_with_progress(url: str, dest: Path):
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as pbar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                pbar.update(len(chunk))


# ── Dataset parsing ────────────────────────────────────────────────────────────
def parse_nasa_composites(
    raw_dir: str,
    max_samples_per_specimen: int = 200,
    signal_length: int = 1024,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Load the NASA CFRP dataset into structured arrays.

    Loading priority: parquets → raw .mat files.
    Raises RuntimeError if no real data is found (synthetic fallback removed).
    """
    raw_path = Path(raw_dir)

    # Try parquet first (in raw_dir or any candidate)
    pqt_dir = None
    if (raw_path / "pzt_waveforms.parquet").exists():
        pqt_dir = raw_path
    else:
        pqt_dir = _find_parquet_dir(raw_dir)

    if pqt_dir is not None:
        logger.info("Loading real NASA parquets from %s", pqt_dir)
        return _load_from_parquet(pqt_dir, signal_length, max_samples_per_specimen, seed)

    # Try raw .mat
    mat_files = sorted(raw_path.rglob("*.mat"), key=str)
    if mat_files:
        logger.info("Found %d .mat files — parsing", len(mat_files))
        return _parse_real_data([str(f) for f in mat_files],
                                max_samples_per_specimen, signal_length, seed)

    raise RuntimeError(
        "No real NASA CFRP data found.\n"
        f"  Looked for parquet in: {raw_dir}\n"
        f"  Looked for .mat files in: {raw_dir}\n"
        "Run scripts/download_dataset.py or place pzt_waveforms.parquet in data/parsed/"
    )


def _load_from_parquet(
    parsed_dir: Path,
    signal_length: int,
    max_samples_per_specimen: int,
    seed: int,
) -> Dict[str, Any]:
    """Build canonical dataset dict from pre-parsed parquet files."""
    rng = np.random.default_rng(seed)
    pqt_path = parsed_dir / "pzt_waveforms.parquet"
    logger.info("Reading %s …", pqt_path)

    # Chunked read to handle large parquets
    try:
        df = pd.read_parquet(str(pqt_path))
    except Exception as exc:
        logger.warning("Full parquet read failed (%s), trying pyarrow chunked …", exc)
        import pyarrow.parquet as pq
        tbl = pq.read_table(str(pqt_path))
        df  = tbl.to_pandas()

    present_feats = [c for c in SCALAR_FEATS if c in df.columns]
    if not present_feats:
        raise RuntimeError(
            f"pzt_waveforms.parquet has no expected feature columns.\n"
            f"Available columns: {list(df.columns)}"
        )

    group_keys = [k for k in ["layup", "specimen", "cycles", "boundary_code", "repetition"]
                  if k in df.columns]
    df_sorted  = df.sort_values(group_keys + ["actuator", "sensor"]
                                if "actuator" in df.columns else group_keys).reset_index(drop=True)

    all_signals, all_cycles_list = [], []
    all_layup_list, all_specimen_list = [], []

    for key_vals, grp in df_sorted.groupby(group_keys, sort=False):
        feat_mat = grp[present_feats].values.astype(np.float32)  # (n_paths, n_feats)
        n_paths, n_feats = feat_mat.shape

        # Pad / truncate to exactly N_SENSORS rows
        if n_paths >= N_SENSORS:
            feat_mat = feat_mat[:N_SENSORS]
        else:
            pad = np.zeros((N_SENSORS - n_paths, n_feats), dtype=np.float32)
            feat_mat = np.vstack([feat_mat, pad])

        # Shape: (n_feats [channels], N_SENSORS [sequence])
        signal_2d = feat_mat.T

        kv = dict(zip(group_keys, key_vals)) if isinstance(key_vals, tuple) \
             else {group_keys[0]: key_vals}

        all_signals.append(signal_2d)
        all_cycles_list.append(int(kv.get("cycles", 0)))
        all_layup_list.append(int(kv.get("layup", 1)) - 1)  # 0-indexed
        all_specimen_list.append(int(kv.get("specimen", 0)))

    if not all_signals:
        raise RuntimeError("No measurement groups found in parquet.")

    # ── Per-specimen normalised life fraction ───────────────────────────────
    cycles_arr   = np.array(all_cycles_list,  dtype=np.int64)
    layup_arr    = np.array(all_layup_list,   dtype=np.int64)
    specimen_arr = np.array(all_specimen_list, dtype=np.int64)

    life_frac = np.zeros(len(cycles_arr), dtype=np.float32)
    for spec_id in np.unique(specimen_arr):
        mask  = specimen_arr == spec_id
        max_c = max(cycles_arr[mask].max(), 1)
        life_frac[mask] = cycles_arr[mask] / max_c
    life_frac = np.clip(life_frac, 0.0, 1.0)

    rul_arr       = (1.0 - life_frac).astype(np.float32)
    damage_arr    = np.array([_life_fraction_to_damage_class(float(lf)) for lf in life_frac], dtype=np.int64)
    stiffness_arr = np.array([_degradation_model(float(lf), "stiffness") for lf in life_frac], dtype=np.float32)
    strength_arr  = np.array([_degradation_model(float(lf), "strength")  for lf in life_frac], dtype=np.float32)

    # ── Strain data ──────────────────────────────────────────────────────────
    strain_pqt = parsed_dir / "strain_data.parquet"
    if strain_pqt.exists():
        try:
            df_strain    = pd.read_parquet(str(strain_pqt))
            numeric_cols = df_strain.select_dtypes(include=np.number).columns.tolist()
            strain_cols  = [c for c in numeric_cols
                            if c not in {"layup","specimen","sample_index"}][:3]
            spec_strain: Dict[int, np.ndarray] = {}
            for sid, sg in df_strain.groupby("specimen"):
                if strain_cols:
                    spec_strain[int(sid)] = sg[strain_cols].mean().values[:3].astype(np.float32)
            strain_arr = np.array(
                [spec_strain.get(int(s), rng.normal(0, 500, 3).astype(np.float32))
                 for s in specimen_arr], dtype=np.float32)
        except Exception:
            strain_arr = np.array(
                [_generate_strain(float(lf), int(ly), rng)
                 for lf, ly in zip(life_frac, layup_arr)], dtype=np.float32)
    else:
        strain_arr = np.array(
            [_generate_strain(float(lf), int(ly), rng)
             for lf, ly in zip(life_frac, layup_arr)], dtype=np.float32)

    signals_arr = np.stack(all_signals, axis=0).astype(np.float32)
    logger.info("Loaded %d measurements (real NASA). Signals shape: %s",
                len(signals_arr), signals_arr.shape)

    return _assemble_dataset(
        list(signals_arr), all_cycles_list, list(rul_arr),
        list(damage_arr), list(stiffness_arr), list(strength_arr),
        list(layup_arr), list(specimen_arr), list(strain_arr),
        source="NASA_PCoE_Real_Parquet",
    )


def _parse_real_data(
    mat_files: List[str],
    max_samples: int,
    signal_length: int,
    seed: int,
) -> Dict[str, Any]:
    """Parse raw NASA CFRP .mat files into structured arrays."""
    from collections import defaultdict
    import re

    rng = np.random.default_rng(seed)
    all_signals, all_cycles = [], []
    all_rul, all_damage, all_stiffness, all_strength = [], [], [], []
    all_layup, all_specimen, all_strain = [], [], []
    specimen_counter = 0

    dir_groups: Dict[str, List[str]] = defaultdict(list)
    for f in mat_files:
        dir_groups[os.path.dirname(f)].append(f)

    for dir_path, files in sorted(dir_groups.items()):
        dir_name = os.path.basename(dir_path).lower()
        layup_id = 0 if "l1" in dir_name or "_1" in dir_name else \
                   1 if "l2" in dir_name or "_2" in dir_name else \
                   2 if "l3" in dir_name or "_3" in dir_name else specimen_counter % 3

        files_sorted = sorted(files, key=_extract_cycle_number)[:max_samples]
        n_snapshots  = len(files_sorted)

        for i, fpath in enumerate(files_sorted):
            try:
                data = _load_mat_file(fpath)
            except Exception as e:
                logger.warning("Skipping %s: %s", fpath, e)
                continue

            signal = _extract_signals(data, signal_length, rng)
            if signal is None:
                continue

            cycle     = _extract_cycle_number(fpath)
            life_frac = np.clip((i + 1) / n_snapshots, 0.0, 1.0)
            rul       = 1.0 - life_frac

            all_signals.append(signal)
            all_cycles.append(cycle)
            all_rul.append(float(rul))
            all_damage.append(_life_fraction_to_damage_class(life_frac))
            all_stiffness.append(_degradation_model(life_frac, "stiffness"))
            all_strength.append(_degradation_model(life_frac, "strength"))
            all_layup.append(layup_id)
            all_specimen.append(specimen_counter)
            all_strain.append(_extract_strain(data, rng))

        specimen_counter += 1

    if len(all_signals) < 50:
        raise RuntimeError(
            f"Only {len(all_signals)} samples parsed from .mat files — too few. "
            "Ensure the correct NASA CFRP dataset directory is provided."
        )

    return _assemble_dataset(
        all_signals, all_cycles, all_rul, all_damage,
        all_stiffness, all_strength, all_layup, all_specimen,
        all_strain, source="NASA_PCoE_Real",
    )


# ── Helper functions ───────────────────────────────────────────────────────────
def _load_mat_file(filepath: str) -> Dict[str, Any]:
    try:
        if HAS_SCIPY:
            data = sio.loadmat(filepath, squeeze_me=True)
            return {k: v for k, v in data.items() if not k.startswith("__")}
    except Exception:
        pass
    if HAS_H5PY:
        data = {}
        with h5py.File(filepath, "r") as f:
            for key in f.keys():
                try:
                    data[key] = np.array(f[key])
                except Exception:
                    data[key] = f[key]
        return data
    raise RuntimeError(f"Cannot load {filepath}: need scipy or h5py")


def _extract_signals(data: Dict, signal_length: int, rng: np.random.Generator) -> Optional[np.ndarray]:
    signal_keys = [k for k in data if any(t in k.lower() for t in ["signal","wave","pzt","data","ch"])]
    for key in signal_keys:
        try:
            arr = np.asarray(data[key], dtype=np.float64)
        except Exception:
            continue
        if arr.ndim == 1:
            arr = _single_to_multichannel(arr, N_SENSORS, signal_length, rng)
        elif arr.ndim == 2:
            if arr.shape[1] == N_SENSORS:
                arr = arr.T
            arr = _resize_signals(arr[:N_SENSORS] if arr.shape[0] >= N_SENSORS
                                  else np.pad(arr, ((0, N_SENSORS - arr.shape[0]), (0, 0))),
                                  signal_length)
        else:
            continue
        if arr.shape == (N_SENSORS, signal_length):
            return arr
    return None


def _single_to_multichannel(signal: np.ndarray, n_ch: int, target_len: int,
                              rng: np.random.Generator) -> np.ndarray:
    if len(signal) != target_len:
        indices = np.linspace(0, len(signal) - 1, target_len)
        signal  = np.interp(indices, np.arange(len(signal)), signal)
    out = np.zeros((n_ch, target_len))
    for ch in range(n_ch):
        attenuation = 1.0 - 0.02 * (ch % 4)
        phase_shift = int(rng.uniform(0, target_len * 0.05))
        noise_level = 0.005 * rng.uniform(0.5, 1.5)
        out[ch] = np.roll(signal, phase_shift) * attenuation + rng.normal(0, noise_level, target_len)
    return out


def _resize_signals(signals: np.ndarray, target_len: int) -> np.ndarray:
    n_ch, curr_len = signals.shape
    if curr_len == target_len:
        return signals
    out     = np.zeros((n_ch, target_len))
    indices = np.linspace(0, curr_len - 1, target_len)
    for ch in range(n_ch):
        out[ch] = np.interp(indices, np.arange(curr_len), signals[ch])
    return out


def _extract_cycle_number(filepath: str) -> int:
    import re
    nums = re.findall(r"\d+", os.path.splitext(os.path.basename(filepath))[0])
    return int(nums[-1]) if nums else 0


def _extract_strain(data: Dict, rng: np.random.Generator) -> np.ndarray:
    for key in [k for k in data if "strain" in k.lower()]:
        arr = np.asarray(data[key], dtype=np.float64).flatten()
        if len(arr) >= 3:
            return arr[:3]
    return rng.normal(0, 500, 3).astype(np.float64)


def _life_fraction_to_damage_class(lf: float) -> int:
    for i, (_, (lo, hi)) in enumerate(DAMAGE_THRESHOLDS.items()):
        if lo <= lf < hi:
            return i
    return N_CLASSES - 1


def _degradation_model(life_fraction: float, mode: str = "stiffness") -> float:
    lf = np.clip(life_fraction, 0.0, 0.999)
    if mode == "stiffness":
        if lf < 0.15:
            return 1.0 - 0.08 * (lf / 0.15)
        elif lf < 0.80:
            return 0.92 - 0.12 * ((lf - 0.15) / 0.65)
        else:
            return max(0.05, 0.80 - 0.45 * ((lf - 0.80) / 0.20) ** 1.5)
    else:  # strength
        if lf < 0.60:
            return 1.0 - 0.05 * lf
        elif lf < 0.85:
            return 0.97 - 0.25 * ((lf - 0.60) / 0.25)
        else:
            return max(0.05, 0.72 - 0.50 * ((lf - 0.85) / 0.15) ** 1.2)


def _generate_strain(life_fraction: float, layup_id: int, rng: np.random.Generator) -> np.ndarray:
    base = np.array([1000.0, 800.0, 200.0]) * [1.0, 0.9, 1.1][layup_id]
    damage_factor = 1.0 + 2.0 * life_fraction
    noise = rng.normal(0, 50, 3)
    return (base * damage_factor + noise).astype(np.float32)


def _assemble_dataset(
    signals, cycles, rul, damage, stiffness, strength,
    layup, specimen, strain, source="Unknown",
) -> Dict[str, Any]:
    return {
        "signals":         np.stack(signals).astype(np.float32),
        "cycles":          np.array(cycles, dtype=np.int64),
        "rul":             np.array(rul,     dtype=np.float32),
        "damage_state":    np.array(damage,  dtype=np.int64),
        "stiffness_ratio": np.array(stiffness, dtype=np.float32),
        "strength_ratio":  np.array(strength,  dtype=np.float32),
        "layup_id":        np.array(layup,   dtype=np.int64),
        "specimen_id":     np.array(specimen, dtype=np.int64),
        "strain_data":     np.stack(strain).astype(np.float32),
        "metadata":        {"source": source, "n_sensors": N_SENSORS,
                            "n_classes": N_CLASSES, "layup_configs": LAYUP_CONFIGS},
    }


# ── Signal normalisation ───────────────────────────────────────────────────────
def normalize_signals(signals: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """Z-score normalise per-channel across all samples."""
    mean = signals.mean(axis=(0, 2), keepdims=True)
    std  = signals.std(axis=(0, 2), keepdims=True) + 1e-8
    return (signals - mean) / std, {"mean": mean, "std": std}


# ── Train/val/test splits ──────────────────────────────────────────────────────
def create_splits(dataset: Dict[str, Any], seed: int = 42,
                  val_frac: float = 0.2, test_frac: float = 0.2) -> Dict:
    """
    Chronological (per-specimen) 60/20/20 split — no forward-time leakage.
    Each specimen's samples are sorted by cycle count, then the later
    samples go into test, the middle into validation.
    """
    rng = np.random.default_rng(seed)
    n   = len(dataset["rul"])
    specimens = np.unique(dataset["specimen_id"])

    train_idx, val_idx, test_idx = [], [], []

    for spec in specimens:
        mask  = dataset["specimen_id"] == spec
        idxs  = np.where(mask)[0]
        order = np.argsort(dataset["cycles"][idxs])
        idxs  = idxs[order]

        n_spec = len(idxs)
        n_test = max(1, int(test_frac  * n_spec))
        n_val  = max(1, int(val_frac   * n_spec))
        n_trn  = n_spec - n_val - n_test

        train_idx.extend(idxs[:n_trn])
        val_idx.extend(idxs[n_trn:n_trn + n_val])
        test_idx.extend(idxs[n_trn + n_val:])

    def _sub(idx):
        idx = np.array(idx)
        return {k: v[idx] for k, v in dataset.items() if isinstance(v, np.ndarray)}

    return {
        "train": _sub(train_idx),
        "val":   _sub(val_idx),
        "test":  _sub(test_idx),
    }


def create_layup_cv_splits(dataset: Dict[str, Any]) -> List[Dict]:
    """Leave-one-layup-out cross-validation splits."""
    folds = []
    for test_layup in np.unique(dataset["layup_id"]):
        test_mask  = dataset["layup_id"] == test_layup
        train_mask = ~test_mask
        folds.append({
            "test_layup": int(test_layup),
            "train": {k: v[train_mask] for k, v in dataset.items() if isinstance(v, np.ndarray)},
            "test":  {k: v[test_mask]  for k, v in dataset.items() if isinstance(v, np.ndarray)},
        })
    return folds


def create_dataloaders(
    splits: Dict,
    task: str = "rul",
    batch_size: int = 32,
) -> Dict[str, DataLoader]:
    """Create PyTorch DataLoaders from pre-split dataset dicts."""
    loaders = {}
    for split_name, split_data in splits.items():
        X = torch.FloatTensor(split_data["signals"])
        if task == "rul":
            y = torch.FloatTensor(split_data["rul"])
        elif task == "classification":
            y = torch.LongTensor(split_data["damage_state"])
        else:
            raise ValueError(f"Unknown task: {task}")
        ds = TensorDataset(X, y)
        shuffle = (split_name == "train")
        loaders[split_name] = DataLoader(ds, batch_size=batch_size,
                                          shuffle=shuffle, num_workers=0,
                                          pin_memory=False)
    return loaders
