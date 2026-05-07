"""
Train the HAR classifier on actual UCI HAR raw inertial windows.

This model intentionally uses only the compact features that the live phone
collector sends to Streamlit. That keeps live prediction fast while still using
the real UCI HAR dataset rather than synthetic data.
"""

from pathlib import Path
import pickle
import urllib.request
import zipfile

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler


DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ZIP_PATH = DATA_DIR / "UCI_HAR_Dataset.zip"
EXTRACTED_DIR = DATA_DIR / "UCI HAR Dataset"
MODEL_PATH = BASE_DIR / "model.pkl"
SAMPLE_RATE_HZ = 50.0
N_ESTIMATORS = 500

ACTIVITIES = [
    "Walking",
    "Walking Upstairs",
    "Walking Downstairs",
    "Sitting",
    "Standing",
    "Laying",
]

FEATURE_NAMES = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "acc_x_std",
    "acc_y_std",
    "acc_z_std",
    "gyro_x_std",
    "gyro_y_std",
    "gyro_z_std",
    "acc_rms_g",
    "gyro_rms",
    "peak_body_g",
    "peak_total_g",
    "peak_jerk_gs",
    "stillness_score",
]


def download_dataset():
    """Download and extract the actual UCI HAR dataset if it is missing."""
    DATA_DIR.mkdir(exist_ok=True)

    if not ZIP_PATH.exists():
        print(f"Downloading UCI HAR dataset from {DATASET_URL}...")
        urllib.request.urlretrieve(DATASET_URL, ZIP_PATH)

    if not EXTRACTED_DIR.exists():
        print("Extracting UCI HAR dataset...")
        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            zf.extractall(DATA_DIR)


def load_signal(split, signal_name):
    path = EXTRACTED_DIR / split / "Inertial Signals" / f"{signal_name}_{split}.txt"
    return np.loadtxt(path)


def load_labels(split):
    path = EXTRACTED_DIR / split / f"y_{split}.txt"
    return np.loadtxt(path, dtype=int) - 1


def build_live_feature_matrix(acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z):
    """Build the same compact feature schema emitted by the phone collector."""
    acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    gyro_mag = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)
    acc_jerk = np.abs(np.diff(acc_mag, axis=1, prepend=acc_mag[:, :1])) * SAMPLE_RATE_HZ
    total_mag = np.sqrt((acc_x + 0.0) ** 2 + (acc_y + 0.0) ** 2 + (acc_z + 1.0) ** 2)
    stillness = np.mean((acc_mag < 0.14) & (gyro_mag < 0.18), axis=1)

    return np.column_stack(
        [
            acc_x.mean(axis=1),
            acc_y.mean(axis=1),
            acc_z.mean(axis=1),
            gyro_x.mean(axis=1),
            gyro_y.mean(axis=1),
            gyro_z.mean(axis=1),
            acc_x.std(axis=1),
            acc_y.std(axis=1),
            acc_z.std(axis=1),
            gyro_x.std(axis=1),
            gyro_y.std(axis=1),
            gyro_z.std(axis=1),
            np.sqrt(np.mean(acc_mag**2, axis=1)),
            np.sqrt(np.mean(gyro_mag**2, axis=1)),
            acc_mag.max(axis=1),
            total_mag.max(axis=1),
            acc_jerk.max(axis=1),
            stillness,
        ]
    )


def load_uci_har_split(split):
    acc_x = load_signal(split, "body_acc_x")
    acc_y = load_signal(split, "body_acc_y")
    acc_z = load_signal(split, "body_acc_z")
    gyro_x = load_signal(split, "body_gyro_x")
    gyro_y = load_signal(split, "body_gyro_y")
    gyro_z = load_signal(split, "body_gyro_z")
    labels = load_labels(split)

    features = build_live_feature_matrix(acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z)
    return features, labels


def load_uci_har_data():
    download_dataset()
    X_train, y_train = load_uci_har_split("train")
    X_test, y_test = load_uci_har_split("test")
    return X_train, X_test, y_train, y_test


def save_artifact(artifact):
    """Write a compressed joblib artifact while keeping the existing model.pkl path."""
    try:
        joblib.dump(artifact, MODEL_PATH, compress=3)
    except Exception:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(artifact, f)


def build_activity_presets(X_train, y_train, model, X_train_scaled):
    """Pick one high-confidence compact feature row for each manual simulator preset."""
    probabilities = model.predict_proba(X_train_scaled)
    presets = {}

    for label, activity in enumerate(ACTIVITIES):
        class_indices = np.flatnonzero(y_train == label)
        best_index = class_indices[np.argmax(probabilities[class_indices, label])]
        row = X_train[int(best_index)]
        features = {name: float(value) for name, value in zip(FEATURE_NAMES, row)}
        presets[activity] = {
            "inputs": {key: features[key] for key in ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")},
            "features": features,
        }

    return presets


def train_and_save():
    print("Loading actual UCI HAR raw inertial signal dataset...")
    X_train, X_test, y_train, y_test = load_uci_har_data()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Training Random Forest model with {N_ESTIMATORS} trees on {len(FEATURE_NAMES)} live features...")
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=28,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {acc:.4f} ({acc * 100:.2f}%)")
    print(classification_report(y_test, y_pred, target_names=ACTIVITIES))

    artifact = {
        "model": model,
        "scaler": scaler,
        "accuracy": acc,
        "activities": ACTIVITIES,
        "feature_names": FEATURE_NAMES,
        "dataset": "UCI HAR Dataset",
        "training_source": "actual UCI HAR raw inertial signal windows",
        "feature_pipeline": "compact live sensor features",
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "n_estimators": N_ESTIMATORS,
        "activity_preset_features": build_activity_presets(X_train, y_train, model, X_train_scaled),
    }

    save_artifact(artifact)
    print(f"Model saved to {MODEL_PATH}")
    return acc


if __name__ == "__main__":
    train_and_save()
