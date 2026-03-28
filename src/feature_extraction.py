"""
feature_extraction.py — Multi-Modal Feature Engineering from Lamb Wave Signals
================================================================================
Implements comprehensive signal processing for CFRP SHM data:
  - Time-of-Flight (ToF) extraction across sensor pairs
  - Discrete Wavelet Transform (DWT) decomposition (Daubechies-4, 5 levels)
  - Signal energy, amplitude attenuation, cross-correlation features  
  - Damage Index (DI) construction with baseline normalization
  - Dimensionality reduction (PCA, t-SNE, UMAP) visualization

Reference:
    Su Z., Ye L., "Identification of Damage Using Lamb Waves", 
    Springer, 2009.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pywt
from scipy import signal as sp_signal
from scipy.stats import kurtosis, skew
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# TIME-OF-FLIGHT (ToF) EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════


def extract_tof(
    signals: np.ndarray,
    sampling_rate: float = 1_000_000,
    method: str = "envelope",
) -> np.ndarray:
    """Extract Time-of-Flight features across all sensor pairs.

    Computes ToF for all unique sensor pairs (16 choose 2 = 120 pairs)
    using Hilbert envelope peak detection or cross-correlation.

    Args:
        signals: Array of shape (N, 16, T) — N samples, 16 sensors, T timesteps.
        sampling_rate: Sampling frequency in Hz.
        method: "envelope" (Hilbert transform) or "xcorr" (cross-correlation).

    Returns:
        Array of shape (N, 120) — ToF in microseconds for each sensor pair.
    """
    n_samples, n_sensors, sig_len = signals.shape
    n_pairs = n_sensors * (n_sensors - 1) // 2
    tof_features = np.zeros((n_samples, n_pairs), dtype=np.float32)

    dt = 1.0 / sampling_rate

    for i in range(n_samples):
        pair_idx = 0
        for s1 in range(n_sensors):
            for s2 in range(s1 + 1, n_sensors):
                if method == "envelope":
                    tof_features[i, pair_idx] = _tof_hilbert(
                        signals[i, s1], signals[i, s2], dt
                    )
                else:
                    tof_features[i, pair_idx] = _tof_xcorr(
                        signals[i, s1], signals[i, s2], dt
                    )
                pair_idx += 1

    return tof_features


def _tof_hilbert(sig1: np.ndarray, sig2: np.ndarray, dt: float) -> float:
    """Compute ToF between two signals using Hilbert envelope peak lag.

    Args:
        sig1: First sensor signal.
        sig2: Second sensor signal.
        dt: Time step in seconds.

    Returns:
        Time-of-flight difference in microseconds.
    """
    env1 = np.abs(sp_signal.hilbert(sig1))
    env2 = np.abs(sp_signal.hilbert(sig2))
    peak1 = np.argmax(env1)
    peak2 = np.argmax(env2)
    return abs(peak2 - peak1) * dt * 1e6  # Convert to microseconds


def _tof_xcorr(sig1: np.ndarray, sig2: np.ndarray, dt: float) -> float:
    """Compute ToF via cross-correlation peak lag.

    Args:
        sig1: First sensor signal.
        sig2: Second sensor signal.
        dt: Time step in seconds.

    Returns:
        Time-of-flight difference in microseconds.
    """
    xcorr = np.correlate(sig1, sig2, mode="full")
    lag = np.argmax(np.abs(xcorr)) - len(sig1) + 1
    return abs(lag) * dt * 1e6


# ═══════════════════════════════════════════════════════════════════════════════
# DISCRETE WAVELET TRANSFORM (DWT) DECOMPOSITION
# ═══════════════════════════════════════════════════════════════════════════════


def extract_dwt_features(
    signals: np.ndarray,
    wavelet: str = "db4",
    max_level: int = 5,
    features_per_band: int = 6,
) -> np.ndarray:
    """Extract DWT decomposition features from multi-channel signals.

    Performs Daubechies-4 wavelet decomposition at 5 levels and extracts
    statistical features from each subband: energy, entropy, RMS,
    kurtosis, skewness, and max amplitude.

    The wavelet decomposition of signal s(t) at scale j is:
        s(t) = Σ_k c_{J,k} φ_{J,k}(t) + Σ_{j=1}^{J} Σ_k d_{j,k} ψ_{j,k}(t)

    where ψ_{j,k} is the mother wavelet at scale j, position k.

    Args:
        signals: Array of shape (N, 16, T).
        wavelet: Wavelet family name (default: 'db4' = Daubechies-4).
        max_level: Maximum decomposition level.
        features_per_band: Statistical features per subband.

    Returns:
        Array of shape (N, 16 * (max_level + 1) * features_per_band).
    """
    n_samples, n_sensors, _ = signals.shape
    n_bands = max_level + 1  # Approximation + detail coefficients
    n_features_total = n_sensors * n_bands * features_per_band
    dwt_feats = np.zeros((n_samples, n_features_total), dtype=np.float32)

    for i in range(n_samples):
        feat_idx = 0
        for ch in range(n_sensors):
            sig = signals[i, ch]

            # Multi-level wavelet decomposition
            coeffs = pywt.wavedec(sig, wavelet, level=max_level)

            for band_coeffs in coeffs:
                band_features = _subband_features(band_coeffs)
                n_f = len(band_features)
                dwt_feats[i, feat_idx:feat_idx + n_f] = band_features
                feat_idx += n_f

    return dwt_feats


def _subband_features(coeffs: np.ndarray) -> np.ndarray:
    """Extract statistical features from a single DWT subband.

    Args:
        coeffs: Wavelet coefficients for one subband.

    Returns:
        Array of 6 features: [energy, entropy, rms, kurtosis, skewness, max_amp].
    """
    c = coeffs.astype(np.float64)
    if len(c) == 0:
        return np.zeros(6, dtype=np.float32)

    # Energy: E = Σ |c_k|²
    energy = np.sum(c ** 2)

    # Shannon entropy: H = -Σ p_k log₂(p_k)
    c_abs = np.abs(c) + 1e-12
    p = c_abs / c_abs.sum()
    entropy = -np.sum(p * np.log2(p + 1e-12))

    # Root mean square
    rms = np.sqrt(np.mean(c ** 2))

    # Higher-order statistics
    kurt = kurtosis(c, fisher=True)  # Excess kurtosis
    skewness = skew(c)

    # Maximum absolute amplitude
    max_amp = np.max(np.abs(c))

    return np.array([energy, entropy, rms, kurt, skewness, max_amp],
                    dtype=np.float32)


def compute_dwt_scalogram(
    signal: np.ndarray,
    wavelet: str = "cmor1.5-1.0",
    scales: Optional[np.ndarray] = None,
    sampling_rate: float = 1_000_000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute continuous wavelet transform scalogram for visualization.

    Args:
        signal: 1D signal array.
        wavelet: CWT wavelet name.
        scales: Array of wavelet scales. If None, auto-computed.
        sampling_rate: Sampling frequency in Hz.

    Returns:
        Tuple of (coefficients, frequencies, time_axis).
    """
    if scales is None:
        scales = np.arange(1, 128)

    coefficients, frequencies = pywt.cwt(
        signal, scales, wavelet, 1.0 / sampling_rate
    )
    time_axis = np.arange(len(signal)) / sampling_rate * 1e6  # microseconds

    return np.abs(coefficients) ** 2, frequencies, time_axis


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL ENERGY & AMPLITUDE FEATURES
# ═══════════════════════════════════════════════════════════════════════════════


def extract_signal_features(
    signals: np.ndarray,
    sampling_rate: float = 1_000_000,
) -> np.ndarray:
    """Extract time-domain and frequency-domain signal features.

    For each sensor channel, computes:
        - Peak amplitude, peak-to-peak amplitude
        - RMS amplitude
        - Total energy (Parseval's theorem in time domain)
        - Envelope area (Hilbert transform)
        - Spectral centroid and bandwidth
        - Dominant frequency
        - Crest factor
        - Zero-crossing rate

    Args:
        signals: Array of shape (N, 16, T).
        sampling_rate: Sampling frequency in Hz.

    Returns:
        Array of shape (N, 16 * 10) — 10 features per sensor.
    """
    n_samples, n_sensors, sig_len = signals.shape
    features_per_sensor = 10
    feats = np.zeros((n_samples, n_sensors * features_per_sensor), dtype=np.float32)

    freqs = np.fft.rfftfreq(sig_len, d=1.0 / sampling_rate)

    for i in range(n_samples):
        for ch in range(n_sensors):
            sig = signals[i, ch].astype(np.float64)
            offset = ch * features_per_sensor

            # Time-domain features
            feats[i, offset] = np.max(np.abs(sig))                     # Peak amplitude
            feats[i, offset + 1] = np.max(sig) - np.min(sig)           # Peak-to-peak
            feats[i, offset + 2] = np.sqrt(np.mean(sig ** 2))          # RMS
            feats[i, offset + 3] = np.sum(sig ** 2) / sig_len          # Energy density

            # Hilbert envelope area
            envelope = np.abs(sp_signal.hilbert(sig))
            feats[i, offset + 4] = np.trapezoid(envelope)

            # Frequency-domain features
            fft_mag = np.abs(np.fft.rfft(sig))
            fft_power = fft_mag ** 2

            # Spectral centroid: f_c = Σ(f * P(f)) / Σ(P(f))
            total_power = fft_power.sum() + 1e-12
            feats[i, offset + 5] = np.sum(freqs * fft_power) / total_power

            # Spectral bandwidth
            centroid = feats[i, offset + 5]
            feats[i, offset + 6] = np.sqrt(
                np.sum(((freqs - centroid) ** 2) * fft_power) / total_power
            )

            # Dominant frequency
            feats[i, offset + 7] = freqs[np.argmax(fft_power[1:]) + 1]

            # Crest factor: peak / RMS
            rms = feats[i, offset + 2]
            feats[i, offset + 8] = feats[i, offset] / (rms + 1e-12)

            # Zero-crossing rate
            feats[i, offset + 9] = np.sum(np.abs(np.diff(np.sign(sig)))) / (2 * sig_len)

    return feats


# ═══════════════════════════════════════════════════════════════════════════════
# CROSS-CORRELATION FEATURES
# ═══════════════════════════════════════════════════════════════════════════════


def extract_cross_correlation_features(
    signals: np.ndarray,
    sensor_pairs: Optional[List[Tuple[int, int]]] = None,
) -> np.ndarray:
    """Extract cross-correlation features between sensor pairs.

    For each selected sensor pair, computes:
        - Maximum cross-correlation coefficient
        - Lag at maximum correlation
        - Correlation decorrelation width (half-power bandwidth)

    Args:
        signals: Array of shape (N, 16, T).
        sensor_pairs: List of (i, j) sensor pairs. If None, uses adjacent pairs.

    Returns:
        Array of shape (N, n_pairs * 3).
    """
    if sensor_pairs is None:
        # Adjacent sensor pairs in the 4×4 grid
        sensor_pairs = []
        for r in range(4):
            for c in range(4):
                idx = r * 4 + c
                if c < 3:
                    sensor_pairs.append((idx, idx + 1))
                if r < 3:
                    sensor_pairs.append((idx, idx + 4))

    n_samples = signals.shape[0]
    n_pairs = len(sensor_pairs)
    feats = np.zeros((n_samples, n_pairs * 3), dtype=np.float32)

    for i in range(n_samples):
        for j, (s1, s2) in enumerate(sensor_pairs):
            sig1 = signals[i, s1].astype(np.float64)
            sig2 = signals[i, s2].astype(np.float64)

            # Normalized cross-correlation
            norm1 = np.linalg.norm(sig1) + 1e-12
            norm2 = np.linalg.norm(sig2) + 1e-12
            xcorr = np.correlate(sig1 / norm1, sig2 / norm2, mode="full")

            offset = j * 3
            feats[i, offset] = np.max(xcorr)                          # Max correlation
            feats[i, offset + 1] = np.argmax(xcorr) - len(sig1) + 1   # Lag
            # Half-power width
            half_max = feats[i, offset] / 2
            above_half = xcorr > half_max
            feats[i, offset + 2] = np.sum(above_half)                 # Width

    return feats


# ═══════════════════════════════════════════════════════════════════════════════
# DAMAGE INDEX (DI) CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════


def compute_damage_index(
    signals: np.ndarray,
    baseline_indices: Optional[np.ndarray] = None,
    method: str = "correlation",
) -> np.ndarray:
    """Compute Damage Index with baseline normalization.

    DI = (S_damaged - S_baseline) / S_baseline

    Supports multiple DI formulations:
        - "correlation": 1 - max(R_xy) between current and baseline signal
        - "energy": |E_current - E_baseline| / E_baseline
        - "amplitude": |A_current - A_baseline| / A_baseline

    Args:
        signals: Array of shape (N, 16, T).
        baseline_indices: Indices of baseline (healthy) samples.
                         If None, uses first 10% of samples.
        method: DI computation method.

    Returns:
        Array of shape (N, 16) — DI per sensor per sample.
    """
    n_samples, n_sensors, sig_len = signals.shape

    if baseline_indices is None:
        n_baseline = max(1, int(0.10 * n_samples))
        baseline_indices = np.arange(n_baseline)

    # Compute baseline reference per sensor
    baseline_signals = signals[baseline_indices].mean(axis=0)  # (16, T)

    di = np.zeros((n_samples, n_sensors), dtype=np.float32)

    for i in range(n_samples):
        for ch in range(n_sensors):
            current = signals[i, ch].astype(np.float64)
            baseline = baseline_signals[ch].astype(np.float64)

            if method == "correlation":
                # Correlation-based DI
                norm_c = np.linalg.norm(current) + 1e-12
                norm_b = np.linalg.norm(baseline) + 1e-12
                corr = np.max(np.correlate(
                    current / norm_c, baseline / norm_b, mode="full"
                ))
                di[i, ch] = 1.0 - corr

            elif method == "energy":
                e_current = np.sum(current ** 2)
                e_baseline = np.sum(baseline ** 2)
                di[i, ch] = abs(e_current - e_baseline) / (e_baseline + 1e-12)

            elif method == "amplitude":
                a_current = np.max(np.abs(current))
                a_baseline = np.max(np.abs(baseline))
                di[i, ch] = abs(a_current - a_baseline) / (a_baseline + 1e-12)

    return di


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE AGGREGATION & DIMENSIONALITY REDUCTION
# ═══════════════════════════════════════════════════════════════════════════════


def build_feature_matrix(
    dataset: Dict[str, np.ndarray],
    include_tof: bool = True,
    include_dwt: bool = True,
    include_signal: bool = True,
    include_xcorr: bool = True,
    include_di: bool = True,
    include_strain: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """Build a comprehensive feature matrix from the CFRP dataset.

    Concatenates all engineered feature groups into a single matrix
    suitable for ML model training.

    Args:
        dataset: Dataset dictionary from data_loader.
        include_tof: Include Time-of-Flight features.
        include_dwt: Include DWT decomposition features.
        include_signal: Include time/freq domain signal features.
        include_xcorr: Include cross-correlation features.
        include_di: Include Damage Index features.
        include_strain: Include strain gage features.

    Returns:
        Tuple of (feature_matrix, feature_names).
    """
    signals = dataset["signals"]
    n_samples = signals.shape[0]
    feature_groups = []
    feature_names = []

    logger.info("Building feature matrix for %d samples...", n_samples)

    if include_tof:
        logger.info("  Extracting ToF features...")
        tof = extract_tof(signals, method="envelope")
        feature_groups.append(tof)
        feature_names.extend([f"ToF_pair_{j}" for j in range(tof.shape[1])])
        logger.info("    -> %d ToF features", tof.shape[1])

    if include_dwt:
        logger.info("  Extracting DWT features (db4, 5 levels)...")
        dwt = extract_dwt_features(signals, wavelet="db4", max_level=5)
        feature_groups.append(dwt)
        sensors = range(16)
        bands = ["A5", "D5", "D4", "D3", "D2", "D1"]
        stats = ["energy", "entropy", "rms", "kurtosis", "skewness", "max_amp"]
        for s in sensors:
            for b in bands:
                for st in stats:
                    feature_names.append(f"DWT_s{s}_{b}_{st}")
        logger.info("    -> %d DWT features", dwt.shape[1])

    if include_signal:
        logger.info("  Extracting signal features...")
        sig_feats = extract_signal_features(signals)
        feature_groups.append(sig_feats)
        sig_names = ["peak_amp", "p2p_amp", "rms", "energy", "env_area",
                     "spec_centroid", "spec_bandwidth", "dom_freq",
                     "crest_factor", "zcr"]
        for s in range(16):
            for name in sig_names:
                feature_names.append(f"Sig_s{s}_{name}")
        logger.info("    -> %d signal features", sig_feats.shape[1])

    if include_xcorr:
        logger.info("  Extracting cross-correlation features...")
        xcorr = extract_cross_correlation_features(signals)
        feature_groups.append(xcorr)
        feature_names.extend([f"XCorr_{j}" for j in range(xcorr.shape[1])])
        logger.info("    -> %d xcorr features", xcorr.shape[1])

    if include_di:
        logger.info("  Computing Damage Index...")
        di = compute_damage_index(signals, method="correlation")
        feature_groups.append(di)
        feature_names.extend([f"DI_s{s}" for s in range(16)])
        logger.info("    -> %d DI features", di.shape[1])

    if include_strain and "strain_data" in dataset:
        logger.info("  Adding strain features...")
        strain = dataset["strain_data"]
        feature_groups.append(strain)
        feature_names.extend(["strain_x", "strain_y", "strain_xy"])
        logger.info("    -> %d strain features", strain.shape[1])

    # Concatenate all feature groups
    X = np.concatenate(feature_groups, axis=1)

    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    logger.info("Feature matrix: %s | Names: %d", X.shape, len(feature_names))

    return X, feature_names


def reduce_dimensions(
    X: np.ndarray,
    method: str = "umap",
    n_components: int = 2,
    seed: int = 42,
    **kwargs,
) -> np.ndarray:
    """Reduce feature dimensions for visualization.

    Args:
        X: Feature matrix of shape (N, D).
        method: "pca", "tsne", or "umap".
        n_components: Target dimensionality.
        seed: Random seed.
        **kwargs: Additional parameters for the reducer.

    Returns:
        Array of shape (N, n_components).
    """
    # Standardize features first
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if method == "pca":
        reducer = PCA(n_components=n_components, random_state=seed)
        return reducer.fit_transform(X_scaled)

    elif method == "tsne":
        from sklearn.manifold import TSNE
        # First reduce with PCA if high-dimensional
        if X_scaled.shape[1] > 50:
            X_pca = PCA(n_components=50, random_state=seed).fit_transform(X_scaled)
        else:
            X_pca = X_scaled
        tsne = TSNE(
            n_components=n_components,
            perplexity=kwargs.get("perplexity", 30),
            random_state=seed,
            n_iter=1000,
        )
        return tsne.fit_transform(X_pca)

    elif method == "umap":
        try:
            import umap
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=kwargs.get("n_neighbors", 15),
                min_dist=kwargs.get("min_dist", 0.1),
                metric=kwargs.get("metric", "euclidean"),
                random_state=seed,
            )
            return reducer.fit_transform(X_scaled)
        except ImportError:
            logger.warning("UMAP not available. Falling back to t-SNE.")
            return reduce_dimensions(X, method="tsne", n_components=n_components, seed=seed)

    raise ValueError(f"Unknown method: {method}")


def get_feature_importance_names() -> Dict[str, str]:
    """Return human-readable descriptions for feature name prefixes.

    Returns:
        Dictionary mapping feature prefix to description.
    """
    return {
        "ToF": "Time-of-Flight (μs)",
        "DWT": "Discrete Wavelet Transform coefficient",
        "Sig": "Time/frequency domain signal feature",
        "XCorr": "Cross-correlation metric",
        "DI": "Damage Index (correlation-based)",
        "strain": "Triaxial strain gage (microstrain)",
    }
