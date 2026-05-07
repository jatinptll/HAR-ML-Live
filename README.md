# HAR-ML-Live

Machine learning based Human Activity Recognition and fall/abnormal motion detection using smartphone IMU sensor data.

The project uses accelerometer and gyroscope readings from a phone to classify human activity in real time. It also includes a rule-based safety layer for detecting sudden abnormal motion patterns such as possible falls.

## Features

- Live human activity recognition from phone sensor data
- Manual simulator mode with preset sensor values
- Six activity classes:
  - Walking
  - Walking Upstairs
  - Walking Downstairs
  - Sitting
  - Standing
  - Laying
- Fall and abnormal motion detection
- Live phone sensor collector for Android/mobile browsers
- Streamlit dashboard with visual predictions
- Confidence chart for all activity classes
- Sensor signal profile visualization
- Feature importance chart
- Local Git and GitHub ready project structure

## Demo

The Streamlit dashboard provides two modes:

- **Live Phone Sensor**: reads sensor data streamed from a phone.
- **Manual Simulator**: lets users test predictions using sliders and quick presets.

The phone collector page runs separately and sends live IMU data to the Streamlit app through a local bridge server.

## Tech Stack

| Area | Technology |
| --- | --- |
| Dashboard | Streamlit |
| Machine learning | Scikit-learn |
| Model | Random Forest Classifier |
| Data processing | NumPy, Pandas |
| Visualization | Plotly |
| Phone sensor capture | JavaScript DeviceMotion API |
| Phone-to-dashboard bridge | Python HTTP server |
| Model storage | Pickle |

## Project Structure

```text
.
├── app.py
├── train_model.py
├── phone_sensor_bridge.py
├── model.pkl
├── requirements.txt
├── README.md
└── components/
    ├── live_refresh/
    │   └── index.html
    └── phone_motion/
        └── index.html
```

## How It Works

```text
Phone IMU sensors
        |
        v
Phone collector page
        |
        v
Python sensor bridge
        |
        v
Latest sensor JSON
        |
        v
Streamlit dashboard
        |
        v
Feature engineering
        |
        v
Scaler + Random Forest model
        |
        v
Activity prediction + fall detection
```

## Machine Learning Pipeline

1. Download the actual UCI HAR Dataset.
2. Load the raw 128-sample inertial signal windows.
3. Extract compact live-compatible features from real accelerometer and gyroscope signals.
4. Scale features using `StandardScaler`.
5. Train a 500-tree `RandomForestClassifier`.
6. Save the trained model, scaler, feature names, and accuracy in `model.pkl`.
7. Load the saved model inside the Streamlit app.
8. Convert live phone sensor readings into the same compact feature format.
9. Predict the current activity and display class probabilities.

## Model Details

The current model is a Random Forest classifier trained on the actual UCI HAR Dataset.

The model input starts from six core sensor channels:

```text
acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
```

The app expands those readings into the same compact live-feature schema used during training:

- Core accelerometer and gyroscope means
- Rolling standard deviation for each sensor axis
- Acceleration and gyroscope magnitude
- Impact peak and jerk peak
- Stillness score

The model predicts one of six human activity classes.

## Fall and Abnormal Motion Detection

Fall detection is implemented as a separate rule-based layer because the activity classifier is not trained on fall data.

The safety layer checks:

- Impact peak
- Jerk peak
- Stillness after movement
- Time since impact
- Whether movement has stabilized

Possible states:

| State | Meaning |
| --- | --- |
| Normal | No abnormal motion pattern detected |
| Abnormal | Recent unusual motion spike or unstable movement |
| Fall likely | Strong impact followed by stillness |

This makes the project more practical than a basic activity classifier while keeping the ML model focused on HAR classification.

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Train the Model

If `model.pkl` already exists, this step is optional.

```bash
python train_model.py
```

This downloads the actual UCI HAR Dataset, extracts compact live-compatible inertial features, trains the 500-tree Random Forest model, and saves the compressed model artifact.

## Run the Streamlit App

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Or with the local virtual environment:

```bash
.venv/bin/python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Open the dashboard:

```text
http://127.0.0.1:8501
```

## Run the Phone Sensor Bridge

In a second terminal:

```bash
.venv/bin/python phone_sensor_bridge.py
```

The phone collector runs on:

```text
http://<mac-ip>:8765
```

Example:

```text
http://10.96.212.95:8765
```

Open that URL on your phone, tap **Start Streaming**, and keep the phone on your body while moving.

## Phone Browser Notes

Mobile browsers may block motion sensors on plain HTTP pages.

For local Android Chrome testing, enable this Chrome flag:

```text
chrome://flags/#unsafely-treat-insecure-origin-as-secure
```

Add your local collector origin:

```text
http://<mac-ip>:8765
```

Then relaunch Chrome and reopen the collector page.

For production deployment, use HTTPS.

## Requirements

```text
streamlit>=1.32.0
scikit-learn>=1.4.0
numpy>=1.26.0
pandas>=2.0.0
plotly>=5.18.0
```

## Limitations

- The current model is trained on the actual UCI HAR Dataset using compact features that match the live phone payload.
- Real-world accuracy depends on phone placement and sensor quality.
- Fall detection is rule-based and not trained on a dedicated fall dataset.
- Browser sensor permissions vary across devices and browsers.
- This is an educational project and should not be used as a medical safety system.

## Future Improvements

- Train using the real UCI HAR dataset
- Collect custom labeled phone sensor data
- Add CSV upload and activity timeline analysis
- Add session analytics and activity duration charts
- Train a time-series model such as LSTM, GRU, or 1D CNN
- Add a real fall detection dataset
- Deploy the dashboard and collector over HTTPS
- Build a mobile app version

## Repository

```text
https://github.com/jatinptll/HAR-ML-Live
```
