"""Shared 561-feature extraction for UCI HAR training and live phone windows."""

from functools import lru_cache

import numpy as np


SAMPLE_RATE_HZ = 50.0
WINDOW_SIZE = 128
FEATURE_COUNT = 561
BASE_AXES = ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")


def _as_window(values):
    arr = np.asarray(values if values is not None else [], dtype=float).reshape(-1)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    if arr.size == 0:
        return np.zeros(WINDOW_SIZE, dtype=float)
    if arr.size >= WINDOW_SIZE:
        return arr[-WINDOW_SIZE:].astype(float)

    pad = np.full(WINDOW_SIZE - arr.size, arr[0], dtype=float)
    return np.concatenate([pad, arr]).astype(float)


def _corr(a, b):
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _stats(values):
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    std = float(np.std(values))
    centered = values - mean
    rms = float(np.sqrt(np.mean(values**2)))
    mad = float(np.mean(np.abs(centered)))
    q25, q75 = np.percentile(values, [25, 75])
    denom = std + 1e-9

    return [
        mean,
        std,
        float(np.min(values)),
        float(np.max(values)),
        float(np.median(values)),
        float(q25),
        float(q75),
        float(q75 - q25),
        mad,
        rms,
        float(np.sum(np.abs(values)) / values.size),
        float(np.mean(values**2)),
        float(np.mean(np.abs(values))),
        float(np.max(np.abs(values))),
        float(np.ptp(values)),
        float(np.mean(np.diff(np.signbit(values)) != 0)),
        float(np.mean(np.diff(np.signbit(centered)) != 0)),
        float(np.mean((centered / denom) ** 3)),
        float(np.mean((centered / denom) ** 4)),
        float((values[-1] - values[0]) / values.size),
        float(values[0]),
        float(values[-1]),
        float(np.ptp(values) / (rms + 1e-9)),
        float(np.mean(values > 0)),
        float(np.mean(values < 0)),
    ]


def _fft_stats(values):
    spectrum = np.abs(np.fft.rfft(values))
    if spectrum.size <= 1:
        spectrum = np.array([0.0], dtype=float)
    else:
        spectrum = spectrum[1:]

    power = spectrum**2
    total_power = float(np.sum(power)) + 1e-9
    probs = power / total_power
    low = spectrum[: max(1, spectrum.size // 5)]
    mid = spectrum[spectrum.size // 5 : max(spectrum.size // 5 + 1, spectrum.size // 2)]

    return [
        float(np.mean(spectrum)),
        float(np.std(spectrum)),
        float(np.max(spectrum)),
        float(total_power / spectrum.size),
        float(-np.sum(probs * np.log2(probs + 1e-12))),
        float(np.argmax(spectrum) / max(1, spectrum.size - 1)),
        float(np.sum(low**2) / total_power),
        float(np.sum(mid**2) / total_power),
    ]


def _build_signals(window):
    acc_x = _as_window(window.get("acc_x"))
    acc_y = _as_window(window.get("acc_y"))
    acc_z = _as_window(window.get("acc_z"))
    gyro_x = _as_window(window.get("gyro_x"))
    gyro_y = _as_window(window.get("gyro_y"))
    gyro_z = _as_window(window.get("gyro_z"))

    acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    gyro_mag = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)
    acc_jerk_x = np.diff(acc_x, prepend=acc_x[0]) * SAMPLE_RATE_HZ
    acc_jerk_y = np.diff(acc_y, prepend=acc_y[0]) * SAMPLE_RATE_HZ
    acc_jerk_z = np.diff(acc_z, prepend=acc_z[0]) * SAMPLE_RATE_HZ
    gyro_jerk_x = np.diff(gyro_x, prepend=gyro_x[0]) * SAMPLE_RATE_HZ
    gyro_jerk_y = np.diff(gyro_y, prepend=gyro_y[0]) * SAMPLE_RATE_HZ
    gyro_jerk_z = np.diff(gyro_z, prepend=gyro_z[0]) * SAMPLE_RATE_HZ
    acc_jerk_mag = np.sqrt(acc_jerk_x**2 + acc_jerk_y**2 + acc_jerk_z**2)
    gyro_jerk_mag = np.sqrt(gyro_jerk_x**2 + gyro_jerk_y**2 + gyro_jerk_z**2)

    return {
        "body_acc_x": acc_x,
        "body_acc_y": acc_y,
        "body_acc_z": acc_z,
        "body_gyro_x": gyro_x,
        "body_gyro_y": gyro_y,
        "body_gyro_z": gyro_z,
        "body_acc_mag": acc_mag,
        "body_gyro_mag": gyro_mag,
        "body_acc_jerk_x": acc_jerk_x,
        "body_acc_jerk_y": acc_jerk_y,
        "body_acc_jerk_z": acc_jerk_z,
        "body_gyro_jerk_x": gyro_jerk_x,
        "body_gyro_jerk_y": gyro_jerk_y,
        "body_gyro_jerk_z": gyro_jerk_z,
        "body_acc_jerk_mag": acc_jerk_mag,
        "body_gyro_jerk_mag": gyro_jerk_mag,
    }


@lru_cache(maxsize=1)
def get_feature_names():
    names = []
    stat_names = [
        "mean",
        "std",
        "min",
        "max",
        "median",
        "q25",
        "q75",
        "iqr",
        "mad",
        "rms",
        "sma",
        "energy",
        "abs_mean",
        "abs_max",
        "range",
        "zero_cross_rate",
        "mean_cross_rate",
        "skew",
        "kurtosis",
        "slope",
        "first",
        "last",
        "range_to_rms",
        "positive_ratio",
        "negative_ratio",
    ]
    fft_names = [
        "fft_mean",
        "fft_std",
        "fft_max",
        "fft_energy",
        "fft_entropy",
        "fft_dom_bin",
        "fft_low_ratio",
        "fft_mid_ratio",
    ]

    signal_names = list(_build_signals({}).keys())
    for signal_name in signal_names:
        names.extend([f"{signal_name}_{name}" for name in stat_names])
        names.extend([f"{signal_name}_{name}" for name in fft_names])

    corr_pairs = [
        ("body_acc_x", "body_acc_y"),
        ("body_acc_x", "body_acc_z"),
        ("body_acc_y", "body_acc_z"),
        ("body_gyro_x", "body_gyro_y"),
        ("body_gyro_x", "body_gyro_z"),
        ("body_gyro_y", "body_gyro_z"),
        ("body_acc_jerk_x", "body_acc_jerk_y"),
        ("body_acc_jerk_x", "body_acc_jerk_z"),
        ("body_acc_jerk_y", "body_acc_jerk_z"),
        ("body_gyro_jerk_x", "body_gyro_jerk_y"),
        ("body_gyro_jerk_x", "body_gyro_jerk_z"),
        ("body_gyro_jerk_y", "body_gyro_jerk_z"),
        ("body_acc_x", "body_gyro_x"),
        ("body_acc_y", "body_gyro_y"),
        ("body_acc_z", "body_gyro_z"),
    ]
    names.extend([f"corr_{left}_{right}" for left, right in corr_pairs])
    names.extend(
        [
            "window_acc_mag_mean",
            "window_acc_mag_std",
            "window_acc_mag_max",
            "window_gyro_mag_mean",
            "window_gyro_mag_std",
            "window_gyro_mag_max",
            "window_acc_jerk_peak",
            "window_gyro_jerk_peak",
            "window_acc_energy_total",
            "window_gyro_energy_total",
            "window_acc_axes_sma",
            "window_gyro_axes_sma",
            "window_stillness_score",
            "window_dynamic_score",
            "window_impact_like_score",
            "window_sample_count_ratio",
            "window_acc_gyro_energy_ratio",
            "window_jerk_ratio",
        ]
    )

    if len(names) != FEATURE_COUNT:
        raise RuntimeError(f"Expected {FEATURE_COUNT} features, got {len(names)}")
    return names


def extract_window_features(window):
    signals = _build_signals(window)
    features = []

    for signal in signals.values():
        features.extend(_stats(signal))
        features.extend(_fft_stats(signal))

    for left, right in [
        ("body_acc_x", "body_acc_y"),
        ("body_acc_x", "body_acc_z"),
        ("body_acc_y", "body_acc_z"),
        ("body_gyro_x", "body_gyro_y"),
        ("body_gyro_x", "body_gyro_z"),
        ("body_gyro_y", "body_gyro_z"),
        ("body_acc_jerk_x", "body_acc_jerk_y"),
        ("body_acc_jerk_x", "body_acc_jerk_z"),
        ("body_acc_jerk_y", "body_acc_jerk_z"),
        ("body_gyro_jerk_x", "body_gyro_jerk_y"),
        ("body_gyro_jerk_x", "body_gyro_jerk_z"),
        ("body_gyro_jerk_y", "body_gyro_jerk_z"),
        ("body_acc_x", "body_gyro_x"),
        ("body_acc_y", "body_gyro_y"),
        ("body_acc_z", "body_gyro_z"),
    ]:
        features.append(_corr(signals[left], signals[right]))

    acc_mag = signals["body_acc_mag"]
    gyro_mag = signals["body_gyro_mag"]
    acc_jerk = signals["body_acc_jerk_mag"]
    gyro_jerk = signals["body_gyro_jerk_mag"]
    acc_energy = float(np.mean(acc_mag**2))
    gyro_energy = float(np.mean(gyro_mag**2))

    features.extend(
        [
            float(np.mean(acc_mag)),
            float(np.std(acc_mag)),
            float(np.max(acc_mag)),
            float(np.mean(gyro_mag)),
            float(np.std(gyro_mag)),
            float(np.max(gyro_mag)),
            float(np.max(acc_jerk)),
            float(np.max(gyro_jerk)),
            acc_energy,
            gyro_energy,
            float(np.mean(np.abs(signals["body_acc_x"]) + np.abs(signals["body_acc_y"]) + np.abs(signals["body_acc_z"]))),
            float(np.mean(np.abs(signals["body_gyro_x"]) + np.abs(signals["body_gyro_y"]) + np.abs(signals["body_gyro_z"]))),
            float(np.mean((acc_mag < 0.14) & (gyro_mag < 0.18))),
            float(np.mean((acc_mag >= 0.14) | (gyro_mag >= 0.18))),
            float(max(np.max(acc_mag) / 1.7, max(np.max(acc_mag) - 1.0, 0.0) / 1.6)),
            float(min(1.0, len(_as_window(window.get("acc_x"))) / WINDOW_SIZE)),
            float(acc_energy / (gyro_energy + 1e-9)),
            float(np.max(acc_jerk) / (np.max(gyro_jerk) + 1e-9)),
        ]
    )

    return np.asarray(features, dtype=float)


def extract_feature_matrix(windows):
    return np.vstack([extract_window_features(window) for window in windows])


def window_from_values(acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z):
    values = {
        "acc_x": float(acc_x),
        "acc_y": float(acc_y),
        "acc_z": float(acc_z),
        "gyro_x": float(gyro_x),
        "gyro_y": float(gyro_y),
        "gyro_z": float(gyro_z),
    }
    dynamic = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2) > 0.12 or np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2) > 0.08
    phase = np.linspace(0, 4 * np.pi, WINDOW_SIZE)
    envelope = 0.25 * np.sin(phase) if dynamic else np.zeros(WINDOW_SIZE)

    return {
        key: np.full(WINDOW_SIZE, value, dtype=float) + envelope * (0.35 if key.startswith("acc") else 0.12)
        for key, value in values.items()
    }


def window_from_motion(motion, fallback_values):
    raw_window = motion.get("window") if isinstance(motion, dict) else None
    if isinstance(raw_window, dict):
        return {key: _as_window(raw_window.get(key)) for key in BASE_AXES}
    return window_from_values(*fallback_values)
