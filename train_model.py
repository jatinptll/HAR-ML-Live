"""
Train a Human Activity Recognition model using synthetic UCI HAR-like data.
In production, replace with actual UCI HAR dataset.
"""
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

ACTIVITIES = [
    "Walking",
    "Walking Upstairs",
    "Walking Downstairs",
    "Sitting",
    "Standing",
    "Laying"
]

ACTIVITY_LABELS = {i: act for i, act in enumerate(ACTIVITIES)}

def generate_synthetic_har_data(n_samples=3000):
    """
    Generate synthetic sensor data mimicking UCI HAR dataset characteristics.
    Features: body_acc_x, body_acc_y, body_acc_z, gyro_x, gyro_y, gyro_z
    + 555 additional engineered features (mean, std, correlation, etc.)
    """
    np.random.seed(42)
    n_per_class = n_samples // 6
    
    X_list, y_list = [], []
    
    # Activity-specific sensor profiles
    profiles = {
        0: {"acc_mean": [0.15, -0.01, 0.02], "acc_std": [0.25, 0.15, 0.20], 
            "gyro_mean": [0.0, 0.0, 0.0], "gyro_std": [0.35, 0.25, 0.20]},   # Walking
        1: {"acc_mean": [0.12, 0.05, 0.03], "acc_std": [0.30, 0.25, 0.25],
            "gyro_mean": [0.05, 0.0, 0.0], "gyro_std": [0.45, 0.35, 0.25]},  # Upstairs
        2: {"acc_mean": [0.12, -0.06, 0.03], "acc_std": [0.28, 0.22, 0.23],
            "gyro_mean": [-0.05, 0.0, 0.0], "gyro_std": [0.42, 0.32, 0.23]}, # Downstairs
        3: {"acc_mean": [0.0, 0.0, -0.08], "acc_std": [0.03, 0.03, 0.03],
            "gyro_mean": [0.0, 0.0, 0.0], "gyro_std": [0.03, 0.03, 0.03]},   # Sitting
        4: {"acc_mean": [0.0, 0.0, -0.07], "acc_std": [0.02, 0.02, 0.04],
            "gyro_mean": [0.0, 0.0, 0.0], "gyro_std": [0.02, 0.02, 0.02]},   # Standing
        5: {"acc_mean": [-0.01, 0.0, 0.97], "acc_std": [0.01, 0.01, 0.01],
            "gyro_mean": [0.0, 0.0, 0.0], "gyro_std": [0.01, 0.01, 0.01]},   # Laying
    }
    
    for label, profile in profiles.items():
        n = n_per_class + (n_samples % 6 if label == 0 else 0)
        
        acc_x = np.random.normal(profile["acc_mean"][0], profile["acc_std"][0], n)
        acc_y = np.random.normal(profile["acc_mean"][1], profile["acc_std"][1], n)
        acc_z = np.random.normal(profile["acc_mean"][2], profile["acc_std"][2], n)
        gyro_x = np.random.normal(profile["gyro_mean"][0], profile["gyro_std"][0], n)
        gyro_y = np.random.normal(profile["gyro_mean"][1], profile["gyro_std"][1], n)
        gyro_z = np.random.normal(profile["gyro_mean"][2], profile["gyro_std"][2], n)
        
        # Build feature set (6 core + 555 engineered to mimic 561 UCI features)
        core = np.column_stack([acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z])
        
        # Engineered features: magnitude, jerk, statistical features
        acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2).reshape(-1, 1)
        gyro_mag = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2).reshape(-1, 1)
        
        # Additional 553 noise features correlated with activity
        extra = np.random.randn(n, 553) * 0.1
        extra += np.array(profile["acc_mean"] * 90 + profile["gyro_mean"] * 90 + [0] * 13)
        
        features = np.hstack([core, acc_mag, gyro_mag, extra])
        X_list.append(features)
        y_list.extend([label] * n)
    
    X = np.vstack(X_list)
    y = np.array(y_list)
    
    # Shuffle
    idx = np.random.permutation(len(y))
    return X[idx], y[idx]


def train_and_save():
    print("Generating synthetic HAR data...")
    X, y = generate_synthetic_har_data(3000)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {acc:.4f} ({acc*100:.2f}%)")
    
    with open("model.pkl", "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "accuracy": acc}, f)
    print("Model saved to model.pkl")
    return acc


if __name__ == "__main__":
    train_and_save()