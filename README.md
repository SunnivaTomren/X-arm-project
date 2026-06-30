# X-Arm EMG Project

Control a UFACTORY xArm robotic arm using surface EMG (electromyography) signals classified by a machine learning model. EMG signals are recorded from muscles in the forearm, processed into time-domain features, and fed to a neural network that predicts the intended gesture in real time.

> **Note:** EMG signals are personal — a model trained on one person's signals will not necessarily work for another person without retraining.

---

## Hardware

| Component | Description |
|-----------|-------------|
| **Olimex EMG board** | Records raw surface EMG at 250 Hz over serial (57600 baud) |
| **UFACTORY xArm** | 6-axis robotic arm controlled over TCP/IP |

---

## Project Structure

```
X-arm-project/
│
├── data/                      # Raw EMG recordings (one folder per gesture)
│   ├── closing_fist/
│   ├── Open_fist/
│   ├── hiya_down/
│   ├── hiya_up/
│   └── lifting_cup/
│
├── scripts/                   # Pipeline scripts (run in order: 1 → 2 → 3 → 4)
│   ├── process_emg.py         # 1. Extract features from CSV files → features.xlsx
│   ├── train_emg.py           # 2. Train a PyTorch neural network
│   ├── predict_emg.py         # 3. Test the model on a single recording
│   └── EMG_to_xArm.py         # 4. Live EMG streaming + xArm control
│
├── output/                    # Generated files (do not edit manually)
│   ├── features.xlsx          # Feature matrix from process_emg.py
│   ├── emg_model.pt           # Saved PyTorch model weights
│   ├── resultat_trening.png   # Training loss + confusion matrix plot
│   └── resultat_prediksjon.png
│
├── .gitignore
└── README.md
```

---

## CSV Recording Format

Each recording file contains a short header followed by time-series data:

```
Action_Label,closing_fist
Fs_Hz,250
Total_Samples,1250

Sample_Index,Raw_EMG,Envelope
0,489.0,527.8
1,516.0,527.9
...
```

- **Raw_EMG** — raw 10-bit ADC value (baseline ≈ 512 at rest)
- **Envelope** — smoothed moving-average of the rectified signal

---

## Features Extracted Per Window

Each 200 ms window (50 samples at 250 Hz, 50 % overlap) is converted into 13 time-domain features:

| Feature | Description |
|---------|-------------|
| MAV | Mean Absolute Value |
| RMS | Root Mean Square |
| WL | Waveform Length |
| VAR | Variance |
| IEMG | Integrated EMG |
| ZC | Zero Crossings |
| SSC | Slope Sign Changes |
| WAMP | Willison Amplitude |
| PEAK | Peak absolute amplitude |
| ENV_MEAN | Mean of the envelope |
| ENV_STD | Standard deviation of the envelope |
| ENV_MAX | Maximum envelope value |
| ENV_RANGE | Envelope max − min |

---

## Setup

**Requirements:** Python 3.10+, a virtual environment with the packages below.

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / Mac

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install pandas scikit-learn matplotlib seaborn openpyxl
```

For live xArm control, also install:
```bash
pip install xarm-python-sdk pyserial
```

---

## Workflow

### Step 1 — Extract features from raw recordings

Reads all CSV files in `data/closing_fist/` and `data/Open_fist/`, splits them into windows, computes the 13 features per window, and saves everything to `output/features.xlsx`.

```bash
python scripts/process_emg.py
```

To include additional gestures, edit the `MAPPER` dictionary in `scripts/process_emg.py`:
```python
MAPPER = {
    os.path.join(DATA_DIR, "Open_fist"):    "opening",
    os.path.join(DATA_DIR, "closing_fist"): "closing",
    os.path.join(DATA_DIR, "hiya_down"):    "hiya_down",
    os.path.join(DATA_DIR, "hiya_up"):      "hiya_up",
    os.path.join(DATA_DIR, "lifting_cup"):  "lifting_cup",
}
```

---

### Step 2 — Train the neural network

Reads `output/features.xlsx`, trains a 4-layer MLP in PyTorch (80/20 train/test split), and saves the model.

```bash
python scripts/train_emg.py
```

**Output:**
- Terminal: accuracy per class, confusion matrix report
- `output/emg_model.pt` — saved model weights + scaler parameters
- `output/resultat_trening.png` — training loss curve and confusion matrix heatmap

**Model architecture:**

```
Input (13) → Linear(128) → BatchNorm → ReLU → Dropout(0.3)
           → Linear(64)  → BatchNorm → ReLU → Dropout(0.3)
           → Linear(32)  → ReLU
           → Linear(n_classes)
```

---

### Step 3 — Test the model on a single recording

Picks a random CSV file from a gesture folder, runs the model window by window, and plots the predictions over the raw signal.

```bash
python scripts/predict_emg.py
```

Change the source folder at the top of `scripts/predict_emg.py`:
```python
MAPPE = os.path.join(DATA_DIR, "closing_fist")   # or Open_fist, hiya_down, hiya_up, lifting_cup
```

**Output:**
- Terminal: predicted class and confidence (%) for each time window
- `output/resultat_prediksjon.png` — raw signal with color-coded prediction regions

---

### Step 4 — Live control of the xArm (optional)

Streams live EMG from the Olimex board, classifies each window using a majority-vote buffer, and sends commands to the xArm.

```bash
# Dry run (prints actions, does not move the arm)
python scripts/EMG_to_xArm.py run --model output/emg_model.joblib --port COM4

# Live motion enabled
python scripts/EMG_to_xArm.py run --model output/emg_model.joblib --port COM4 --enable-motion
```

Gesture-to-action mapping is defined in `GESTURE_TO_ACTION` inside `EMG_to_xArm.py`:
```python
GESTURE_TO_ACTION = {
    "closing": {"gripper": 0},     # close gripper
    "opening": {"gripper": 850},   # open gripper
    "rest":    None,               # do nothing
}
```

---

## Authors

University of Agder (UiA) — student research project using EMG signals to control a UFACTORY xArm.
