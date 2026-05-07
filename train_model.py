"""
Train a Human Activity Recognition model on the actual UCI HAR Dataset.

The original UCI HAR dataset provides precomputed 561-feature vectors and raw
128-sample inertial signal windows. This app predicts from live phone summaries,
so training uses the real raw inertial windows and extracts the same compact
runtime-compatible features that the Streamlit app can compute from live data.
"""

from pathlib import Path
import pickle
import urllib.request
import zipfile

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ZIP_PATH = DATA_DIR / "UCI_HAR_Dataset.zip"
EXTRACTED_DIR = DATA_DIR / "UCI HAR Dataset"
MODEL_PATH = BASE_DIR / "model.pkl"
SAMPLE_RATE_HZ = 50.0

ACTIVITIES = [
    "Walking",
    "Walking Upstairs",
    "Walking Downstairs",
    "Sitting",
    "Standing",
    "Laying",
]

FEATURE_NAMES = [
    "acc_x_mean",
    "acc_y_mean",
    "acc_z_mean",
    "gyro_x_mean",
    "gyro_y_mean",
    "gyro_z_mean",
    "acc_x_std",
    "acc_y_std",
    "acc_z_std",
    "gyro_x_std",
    "gyro_y_std",
    "gyro_z_std",
    "acc_rms_g",
    "gyro_rms",
    "peak_body_g",
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


def build_window_features(acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z):
    """
    Build compact features from real 128-sample UCI HAR inertial windows.

    The feature schema mirrors what the live phone collector can provide:
    channel means, magnitude RMS values, impact peak, jerk peak, and stillness.
    """
    acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    gyro_mag = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)
    acc_jerk = np.abs(np.diff(acc_mag, axis=1, prepend=acc_mag[:, :1])) * SAMPLE_RATE_HZ
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

    features = build_window_features(acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z)
    return features, labels


def load_uci_har_data():
    download_dataset()
    X_train, y_train = load_uci_har_split("train")
    X_test, y_test = load_uci_har_split("test")
    return X_train, X_test, y_train, y_test


def train_and_save():
    print("Loading actual UCI HAR raw inertial signal dataset...")
    X_train, X_test, y_train, y_test = load_uci_har_data()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training Random Forest model on actual UCI HAR windows...")
    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=24,
        min_samples_split=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {acc:.4f} ({acc * 100:.2f}%)")

    artifact = {
        "model": model,
        "scaler": scaler,
        "accuracy": acc,
        "activities": ACTIVITIES,
        "feature_names": FEATURE_NAMES,
        "dataset": "UCI HAR Dataset",
        "training_source": "actual UCI HAR raw inertial signal windows",
        "sample_rate_hz": SAMPLE_RATE_HZ,
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artifact, f)

    print(f"Model saved to {MODEL_PATH}")
    return acc


if __name__ == "__main__":
    train_and_save()
