# Human Activity Recognition with Live Phone Sensor and Fall Detection

This project is a Human Activity Recognition (HAR) system that predicts a person's physical activity using smartphone IMU sensor data. It started as a manual sensor input demo and was upgraded into a more practical live monitoring system.

The final project supports:

- Manual sensor simulation using sliders
- Live phone accelerometer and gyroscope input
- ML-based activity prediction
- Fall and abnormal motion detection
- Streamlit dashboard for visualization
- Separate phone sensor collector page for Android Chrome

The project is built for demonstrating how smartphone motion sensors can be used in fitness tracking, elderly monitoring, wearable devices, fall detection systems, and IoT-based health applications.

## Project Summary

The system classifies human activity into six classes:

1. Walking
2. Walking Upstairs
3. Walking Downstairs
4. Sitting
5. Standing
6. Laying

The application takes accelerometer and gyroscope readings, converts them into model-compatible features, scales them, and predicts the activity using a trained Random Forest classifier.

On top of the activity classifier, the project also has a fall detection layer. This layer checks for sudden impact, high jerk, and post-impact stillness to decide whether the current motion is normal, abnormal, or likely a fall.

## Why This Project Is Useful

Human activity recognition is used in many real systems:

- Smartwatches and fitness bands
- Mobile health monitoring
- Elderly care systems
- Fall detection systems
- Sports and rehabilitation tracking
- IoT wearable devices
- Workplace safety monitoring

Instead of only changing sensor sliders manually, this project can read real phone motion data and send it to the ML model for live prediction.

## Tech Stack

| Part | Technology |
| --- | --- |
| Frontend dashboard | Streamlit |
| ML model | Scikit-learn Random Forest |
| Numerical processing | NumPy |
| Visualization | Plotly |
| Model storage | Pickle |
| Phone sensor page | HTML, CSS, JavaScript |
| Phone-to-Mac bridge | Python HTTP server |
| Sensor source | Smartphone accelerometer and gyroscope |

## Project Files

```text
Activity/
  app.py
  train_model.py
  model.pkl
  phone_sensor_bridge.py
  phone_motion_latest.json
  requirements.txt
  README.md
  components/
    phone_motion/
      index.html
    live_refresh/
      index.html
```

### File Explanation

`app.py`

Main Streamlit dashboard. It loads the model, reads sensor input, predicts activity, shows confidence charts, displays sensor metrics, and performs fall detection.

`train_model.py`

Generates synthetic UCI HAR-like training data, trains a Random Forest classifier, evaluates accuracy, and saves the model to `model.pkl`.

`model.pkl`

Serialized model artifact. It contains:

- Trained Random Forest model
- StandardScaler
- Test accuracy

`phone_sensor_bridge.py`

Runs a local phone sensor receiver on port `8765`. The phone opens this page, sends live motion data to the Mac, and the bridge writes the latest sensor window to `phone_motion_latest.json`.

`phone_motion_latest.json`

Temporary live sensor data file. Streamlit reads this file continuously to get the latest phone readings.

`components/phone_motion/index.html`

Streamlit embedded browser sensor component. It can collect motion directly if the dashboard is opened on a phone.

`components/live_refresh/index.html`

A small Streamlit component that triggers periodic refreshes so live sensor predictions update automatically.

`requirements.txt`

Python dependencies.

## Complete Pipeline

This is the full zero-to-hero pipeline of the project.

### 1. Problem Definition

The problem is to recognize human physical activity from smartphone sensor data.

Input:

- Accelerometer readings
- Gyroscope readings

Output:

- Predicted activity class
- Prediction confidence
- Fall or abnormal motion status

### 2. Sensor Data

The project uses IMU data.

IMU means Inertial Measurement Unit. A smartphone IMU usually contains:

- Accelerometer
- Gyroscope
- Sometimes magnetometer

This project uses:

| Sensor | Meaning |
| --- | --- |
| Accelerometer | Measures linear acceleration |
| Gyroscope | Measures angular velocity |

The six core input values are:

```text
acc_x
acc_y
acc_z
gyro_x
gyro_y
gyro_z
```

Accelerometer values are treated as g-force values. Gyroscope values are treated as rad/s.

### 3. Dataset Concept

The project is inspired by the UCI HAR dataset.

UCI HAR stands for University of California Irvine Human Activity Recognition dataset. It is a well-known dataset collected from smartphone accelerometer and gyroscope readings.

In this implementation, `train_model.py` generates synthetic UCI HAR-like data instead of downloading the full real dataset. This keeps the project lightweight and easy to run in viva/demo environments.

Synthetic data is generated with different motion profiles for each class.

Example:

- Walking has higher acceleration variation.
- Sitting and standing have low movement.
- Laying has a different gravity/body orientation pattern.
- Upstairs and downstairs have different acceleration and gyroscope patterns.

### 4. Data Generation

The script `train_model.py` generates 3000 samples.

It creates six activity profiles:

```python
Walking
Walking Upstairs
Walking Downstairs
Sitting
Standing
Laying
```

For each activity, it defines:

- Mean acceleration
- Standard deviation of acceleration
- Mean gyroscope value
- Standard deviation of gyroscope value

Then random samples are generated using normal distribution.

### 5. Feature Engineering

The model expects 561 features, similar to the UCI HAR dataset feature structure.

Feature vector structure:

```text
6 core sensor features
2 magnitude features
553 synthetic engineered features
= 561 total features
```

The six core features are:

```text
acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
```

Two magnitude features are calculated:

```text
acc_mag = sqrt(acc_x^2 + acc_y^2 + acc_z^2)
gyro_mag = sqrt(gyro_x^2 + gyro_y^2 + gyro_z^2)
```

The remaining 553 features simulate the kind of statistical and frequency-domain features used in the original UCI HAR dataset.

This allows the project to demonstrate a realistic HAR pipeline while keeping the code simple.

### 6. Data Preprocessing

Before training, the data is split into:

- Training set: 80 percent
- Testing set: 20 percent

Then `StandardScaler` is applied.

Standardization formula:

```text
z = (x - mean) / standard_deviation
```

Why scaling is important:

- Different features may have different ranges.
- Scaling improves model stability.
- It ensures large-valued features do not dominate small-valued features.

The scaler is saved with the model so the same transformation is applied during prediction.

### 7. Model Training

The model used is Random Forest Classifier.

Configuration:

```python
RandomForestClassifier(
    n_estimators=150,
    max_depth=20,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
```

Random Forest was chosen because:

- It works well for tabular sensor features.
- It is robust to noise.
- It handles nonlinear decision boundaries.
- It gives good accuracy without heavy tuning.
- It is explainable compared to deep learning.

### 8. Model Evaluation

After training, the model predicts on the test set.

Accuracy is calculated using:

```text
accuracy = correct_predictions / total_predictions
```

The Streamlit sidebar displays the saved test accuracy.

Current model accuracy shown in the app is around 88 percent.

### 9. Model Saving

The trained model is stored in `model.pkl`.

The pickle file contains:

```python
{
    "model": model,
    "scaler": scaler,
    "accuracy": acc
}
```

This avoids retraining every time the app starts.

### 10. Streamlit App Flow

When `app.py` starts:

1. Streamlit page is configured.
2. Custom CSS is loaded.
3. `model.pkl` is loaded.
4. If the model is missing, `train_model.py` automatically trains one.
5. User selects input mode.
6. Sensor readings are collected.
7. Features are generated.
8. Features are scaled.
9. Model predicts activity.
10. Confidence values are displayed.
11. Fall risk is calculated.
12. Charts and metrics are updated.

## Input Modes

The app has two input modes.

### Manual Simulator Mode

This mode uses Streamlit sliders.

It is useful for:

- Demonstrating how the model reacts to different sensor values
- Debugging
- Classroom explanation
- Running without a phone

The sidebar has quick presets for:

- Walking
- Walking Upstairs
- Walking Downstairs
- Sitting
- Standing
- Laying

### Live Phone Sensor Mode

This mode uses real phone motion data.

The phone opens the collector page:

```text
http://<mac-ip>:8765
```

The Streamlit dashboard runs at:

```text
http://<mac-ip>:8501
```

The phone sends data to the Mac, and the Streamlit dashboard reads the latest motion window.

## Live Phone Sensor Pipeline

This is the live sensor pipeline:

```text
Samsung phone sensors
    |
    v
Chrome DeviceMotion API
    |
    v
Phone collector web page
    |
    v
HTTP POST to Mac on port 8765
    |
    v
phone_sensor_bridge.py
    |
    v
phone_motion_latest.json
    |
    v
Streamlit app.py
    |
    v
Feature engineering
    |
    v
Scaler
    |
    v
Random Forest model
    |
    v
Activity prediction and fall detection
```

## Phone Sensor Collector

The collector page uses JavaScript `DeviceMotionEvent`.

It reads:

- `event.acceleration`
- `event.accelerationIncludingGravity`
- `event.rotationRate`

The collector calculates:

- Rolling acceleration
- Rolling gyroscope values
- Sample rate
- Body acceleration peak
- Total impact peak
- Jerk peak
- Stillness score

The collector sends the processed values to:

```text
POST /motion
```

The Python bridge receives the JSON and writes it to:

```text
phone_motion_latest.json
```

## Important Sensor Terms

### Acceleration

Acceleration measures change in velocity. In this project, it is used to understand body movement intensity.

### Gyroscope

Gyroscope measures angular velocity. It tells how much the phone is rotating around each axis.

### Magnitude

Magnitude combines x, y, and z axes into one value.

Formula:

```text
magnitude = sqrt(x^2 + y^2 + z^2)
```

### Jerk

Jerk is the rate of change of acceleration.

High jerk means a sudden movement or impact.

Formula:

```text
jerk = change_in_acceleration / change_in_time
```

### Stillness Score

Stillness score estimates whether the phone became stable after movement.

It is useful for fall detection because a common fall pattern is:

```text
normal movement -> sudden impact -> low movement
```

## Fall and Abnormal Motion Detection

The Random Forest model predicts activity. Fall detection is handled by a separate rule-based layer.

This is intentional because the model is trained on six HAR activity classes, not on fall data. Since fall is not one of the six classes, fall detection should not be forced into the classifier.

The fall layer checks:

- Impact peak
- Jerk peak
- Stillness score
- Time since impact
- Whether current motion has stabilized

The possible states are:

| State | Meaning |
| --- | --- |
| Normal | No dangerous motion pattern |
| Abnormal | Recent unusual spike or unstable motion |
| Fall likely | Hard impact followed by stillness |

### Fall Logic

The system marks `Fall likely` when:

- Impact is strong
- Jerk is high
- Impact is recent
- Phone becomes still after impact

The system marks `Abnormal` when:

- There is a recent impact spike, or
- Motion remains unstable after a spike

The system returns to `Normal` when:

- The spike is old, or
- Movement has stabilized

This prevents the app from staying stuck on `Abnormal` after the user stops moving.

## Why Use a Separate Fall Detection Layer?

The activity model only knows six classes. It does not know a fall class.

If we used only the Random Forest classifier, a fall might be incorrectly classified as walking, laying, or standing. Therefore, fall detection is implemented as an additional safety layer using motion physics.

This is a good engineering decision because:

- It avoids pretending the classifier was trained on fall data.
- It keeps the activity model focused on HAR.
- It allows real-time safety rules to be tuned separately.

## Visualizations in the Dashboard

The Streamlit dashboard shows:

- Current detected activity
- Prediction confidence
- Probability bar chart
- Live model inputs
- Acceleration magnitude
- Gyroscope magnitude
- Motion type: Static or Dynamic
- Fall risk card
- Sensor radar chart
- Activity probability heatmap
- Feature importance chart

## How to Run the Project

### 1. Create Virtual Environment

```bash
python3 -m venv .venv
```

### 2. Activate Virtual Environment

On macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train Model

This step is optional if `model.pkl` already exists.

```bash
python train_model.py
```

### 5. Start Streamlit Dashboard

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Or with the local virtual environment:

```bash
.venv/bin/python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

### 6. Start Phone Sensor Bridge

Open another terminal:

```bash
.venv/bin/python phone_sensor_bridge.py
```

The bridge runs on:

```text
http://0.0.0.0:8765
```

### 7. Open on Mac

Open Streamlit:

```text
http://127.0.0.1:8501
```

### 8. Open on Phone

Find the Mac IP shown in the Streamlit dashboard or using:

```bash
ifconfig
```

Then open this on the phone:

```text
http://<mac-ip>:8765
```

Example:

```text
http://10.96.212.95:8765
```

Tap `Start Streaming` on the phone collector page.

## Demo Flow for Viva

Use this sequence during your viva:

1. Open Streamlit dashboard.
2. Explain the six activity classes.
3. Show manual simulator mode.
4. Click quick presets and show predictions changing.
5. Switch to live phone sensor mode.
6. Open phone collector page on the phone.
7. Tap `Start Streaming`.
8. Move with the phone and show live values updating.
9. Show prediction confidence chart.
10. Explain fall detection card.
11. Create a sudden movement carefully and show abnormal detection.
12. Keep phone stable and explain why it returns to normal.
13. Explain limitations and future improvements.

## What to Say in Viva

### Short Project Explanation

This project recognizes human physical activity using smartphone accelerometer and gyroscope data. I trained a Random Forest model on synthetic UCI HAR-like features and deployed it in a Streamlit dashboard. The system supports both manual sensor simulation and live phone sensor streaming. I also added a rule-based fall detection layer using impact, jerk, and stillness after impact.

### One-Minute Technical Explanation

The phone provides six IMU readings: three acceleration axes and three gyroscope axes. These are converted into a 561-dimensional feature vector similar to the UCI HAR dataset. The feature vector is standardized using a saved StandardScaler and passed to a Random Forest classifier. The classifier outputs one of six activities and class probabilities. For fall detection, I calculate impact magnitude, jerk, stillness score, and time since impact. If there is a strong impact followed by stillness, the system marks it as fall likely. If there is only a recent unstable spike, it marks abnormal. Once motion stabilizes, it returns to normal.

## Expected Viva Questions and Answers

### 1. What is HAR?

HAR means Human Activity Recognition. It is the task of identifying a person's activity using sensor data such as accelerometer and gyroscope readings.

### 2. What sensors are used?

The project uses accelerometer and gyroscope data from a smartphone.

### 3. Why use accelerometer?

Accelerometer data helps detect body movement, direction, vibration, and impact.

### 4. Why use gyroscope?

Gyroscope data helps detect rotation and orientation changes, which improves recognition of movements like walking upstairs or downstairs.

### 5. What are the input features?

The six core inputs are:

```text
acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
```

The final feature vector has 561 features.

### 6. Why 561 features?

The original UCI HAR dataset uses 561 engineered features. This project mimics that structure by combining core sensor values, magnitude features, and synthetic engineered features.

### 7. Which ML algorithm is used?

Random Forest Classifier.

### 8. Why Random Forest?

Random Forest works well for tabular feature data, handles nonlinear relationships, is robust to noise, and provides good performance without requiring heavy tuning.

### 9. What is StandardScaler?

StandardScaler standardizes each feature by removing mean and scaling to unit variance. It helps keep all features on comparable scales.

### 10. What is model.pkl?

It is the saved model file. It stores the trained Random Forest model, scaler, and accuracy.

### 11. What is the accuracy?

The displayed test accuracy is around 88 percent. This is based on synthetic UCI HAR-like test data.

### 12. Is the model trained on real sensor data?

This version uses synthetic UCI HAR-like data. The structure is inspired by the real UCI HAR dataset. In a production version, the synthetic generator should be replaced with real collected or downloaded UCI HAR data.

### 13. Why synthetic data?

Synthetic data makes the project easy to run offline, easy to demonstrate, and avoids dataset download/setup issues. It is good for demonstrating the pipeline, but real data would improve practical reliability.

### 14. How does live phone mode work?

The phone opens a sensor collector page. JavaScript reads phone motion sensors and sends the processed values to a Python bridge on the Mac. Streamlit reads the latest JSON file and passes the values to the ML model.

### 15. What is DeviceMotionEvent?

`DeviceMotionEvent` is a browser API that gives access to accelerometer and gyroscope-like motion data on supported mobile browsers.

### 16. Why is a bridge server needed?

The Streamlit app runs on the Mac. The phone sensor data must be sent from the phone to the Mac. The bridge server receives this data and stores the latest readings for Streamlit to consume.

### 17. What is fall detection based on?

Fall detection is based on:

- High impact
- High jerk
- Stillness after impact
- Time since impact

### 18. Why not use the ML model for fall detection?

The model was trained only on six activity classes and not on fall data. So fall detection is implemented separately using rule-based motion analysis.

### 19. What is jerk?

Jerk is the rate of change of acceleration. A sudden impact creates high jerk.

### 20. What happens after motion stabilizes?

The abnormal status decays over time. If the phone becomes stable and the spike is no longer recent, the fall detector returns to normal.

### 21. What are limitations?

Limitations:

- Model uses synthetic data.
- Phone browser sensor permissions can vary.
- Fall detection is rule-based, not trained on real fall data.
- Phone placement affects readings.
- It is not a medical-grade fall detector.

### 22. How can this project be improved?

Future improvements:

- Train on real UCI HAR dataset.
- Collect custom real phone data.
- Add LSTM, GRU, or 1D CNN for time-series classification.
- Add real fall dataset.
- Add user calibration.
- Add session history and activity timeline.
- Deploy with HTTPS for smoother phone sensor access.

## Limitations

This project is educational and demonstrative. It should not be used as a real medical or emergency fall detection system without real-world validation.

Important limitations:

- Synthetic training data is not a complete substitute for real-world data.
- Phone sensor values differ across devices.
- Sensor access depends on browser permissions.
- User phone placement changes readings.
- Rule-based fall detection may produce false positives or false negatives.

## Future Scope

Possible future upgrades:

1. Use the real UCI HAR dataset.
2. Collect live labeled data from the phone.
3. Train a deep learning time-series model.
4. Add CSV upload and activity timeline.
5. Add activity duration analytics.
6. Add calorie estimation.
7. Add user-specific calibration.
8. Add alert system for fall detection.
9. Deploy over HTTPS.
10. Build a mobile app version.

## Troubleshooting During Demo

### Phone cannot open the collector page

If the phone cannot open:

```text
http://<mac-ip>:8765
```

check these points:

1. Mac and phone must be on the same network or hotspot.
2. The bridge server must be running:

```bash
.venv/bin/python phone_sensor_bridge.py
```

3. macOS firewall may block inbound Python connections. Turn it off temporarily for demo or allow Python in firewall settings.
4. Check the Mac IP using:

```bash
ifconfig
```

5. Make sure the URL uses the Mac Wi-Fi IP, not `127.0.0.1`. The phone cannot use the Mac's localhost address.

### Phone page opens but shows no readings

This usually happens because Android Chrome blocks motion sensors on insecure HTTP origins.

Fix:

1. Open this in Chrome on the phone:

```text
chrome://flags/#unsafely-treat-insecure-origin-as-secure
```

2. Add the collector origin:

```text
http://<mac-ip>:8765
```

Example:

```text
http://10.96.212.95:8765
```

3. Enable the flag.
4. Relaunch Chrome.
5. Reopen the collector page.
6. Tap `Start Streaming`.

For production, the better solution is to host the collector over HTTPS.

### Streamlit is not updating

Check:

1. Streamlit server is running on port `8501`.
2. Phone bridge is running on port `8765`.
3. `phone_motion_latest.json` is updating.
4. Live Phone Sensor mode is selected in the sidebar.
5. Refresh the Streamlit browser tab.

### Abnormal status stays after movement stops

The app includes decay logic for old impact spikes.

If motion stabilizes, the card should return to `Normal`. It may remain `Abnormal` briefly if:

- The impact was very recent
- Jerk is still high
- The phone is still moving
- The stillness score is low

This behavior is expected because the detector waits to confirm stabilization.

## Conclusion

This project demonstrates a complete Human Activity Recognition pipeline:

```text
Sensor data -> Feature engineering -> Scaling -> ML prediction -> Visualization -> Fall detection
```

It combines machine learning, sensor processing, real-time web communication, and Streamlit visualization. The upgraded live phone sensor mode makes the project more practical than a basic slider demo, and the fall detection layer adds a useful real-world safety application.
