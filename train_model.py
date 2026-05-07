"""
Train the HAR classifier on the actual UCI HAR raw inertial windows.

This version uses the full 128-sample accelerometer and gyroscope windows to
build a 561-feature vector for every UCI HAR example. The live phone collector
sends the same rolling window shape, so the trained model sees richer motion
patterns than simple mean/std summaries.
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

from har_features import FEATURE_COUNT, SAMPLE_RATE_HZ, extract_feature_matrix, get_feature_names


DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ZIP_PATH = DATA_DIR / "UCI_HAR_Dataset.zip"
EXTRACTED_DIR = DATA_DIR / "UCI HAR Dataset"
MODEL_PATH = BASE_DIR / "model.pkl"
N_ESTIMATORS = 500

ACTIVITIES = [
    "Walking",
    "Walking Upstairs",
    "Walking Downstairs",
    "Sitting",
    "Standing",
    "Laying",
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


def load_uci_windows(split):
    acc_x = load_signal(split, "body_acc_x")
    acc_y = load_signal(split, "body_acc_y")
    acc_z = load_signal(split, "body_acc_z")
    gyro_x = load_signal(split, "body_gyro_x")
    gyro_y = load_signal(split, "body_gyro_y")
    gyro_z = load_signal(split, "body_gyro_z")

    windows = []
    for index in range(acc_x.shape[0]):
        windows.append(
            {
                "acc_x": acc_x[index],
                "acc_y": acc_y[index],
                "acc_z": acc_z[index],
                "gyro_x": gyro_x[index],
                "gyro_y": gyro_y[index],
                "gyro_z": gyro_z[index],
            }
        )
    return windows, load_labels(split)


def load_uci_har_data():
    download_dataset()
    train_windows, y_train = load_uci_windows("train")
    test_windows, y_test = load_uci_windows("test")

    print(f"Extracting {FEATURE_COUNT} features from train windows...")
    X_train = extract_feature_matrix(train_windows)
    print(f"Extracting {FEATURE_COUNT} features from test windows...")
    X_test = extract_feature_matrix(test_windows)

    return X_train, X_test, y_train, y_test


def save_artifact(artifact):
    """Write a compressed joblib artifact while keeping the existing model.pkl path."""
    try:
        joblib.dump(artifact, MODEL_PATH, compress=3)
    except Exception:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(artifact, f)


def train_and_save():
    print("Loading actual UCI HAR raw inertial signal dataset...")
    X_train, X_test, y_train, y_test = load_uci_har_data()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Training Random Forest model with {N_ESTIMATORS} trees on full 561-feature windows...")
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
        "feature_names": get_feature_names(),
        "dataset": "UCI HAR Dataset",
        "training_source": "actual UCI HAR raw inertial signal windows",
        "feature_pipeline": "shared 561-feature rolling-window extractor",
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "n_estimators": N_ESTIMATORS,
    }

    save_artifact(artifact)
    print(f"Model saved to {MODEL_PATH}")
    return acc


if __name__ == "__main__":
    train_and_save()
