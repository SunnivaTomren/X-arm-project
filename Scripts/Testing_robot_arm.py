import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xarm.wrapper import XArmAPI

# ── Settings ──────────────────────────────────────────────────────────────────

ROBOT_IP    = "192.168.1.225"
MODEL_PATH  = "Models/emg_model_deep.pt"
TRAIN_DATA  = r"C:\Users\celin\Documents\Mekatronikk\Prosjekt_UiA\Robot_Arm\X-arm-project\data\processed\features.xlsx"
SAMPLE_FILE = r"C:\Users\celin\Documents\Mekatronikk\Prosjekt_UiA\Robot_Arm\X-arm-project\data\raw\closing_fist_celine\closing_fist_C_20260714_125316.csv"

WINDOW_SIZE = 250   # samples per window (1 second at 250 Hz)
STRIDE      = 125   # 50% overlap
ZC_THRESH   = 5     # zero-crossing threshold
WAMP_THRESH = 10    # Willison amplitude threshold

FEATURES = ["MAV", "RMS", "WL", "VAR", "IEMG", "ZC", "SSC", "WAMP", "PEAK",
            "ENV_MEAN", "ENV_STD", "ENV_RANGE"]

GRIPPER_OPEN   = 850
GRIPPER_CLOSED = 0

# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(emg, env):
    """Extract the 13 features from one window of raw EMG + envelope."""
    emg = np.array(emg, dtype=np.float32)
    env = np.array(env, dtype=np.float32)
    centered = emg - emg.mean()

    MAV      = np.mean(np.abs(emg))
    RMS      = np.sqrt(np.mean(emg ** 2))
    WL       = np.sum(np.abs(np.diff(emg)))
    VAR      = np.var(emg)
    IEMG     = np.sum(np.abs(emg))
    ZC       = np.sum((centered[:-1] * centered[1:] < 0) & (np.abs(np.diff(centered)) > ZC_THRESH))
    SSC      = np.sum(np.diff(np.sign(np.diff(emg))) != 0)
    WAMP     = np.sum(np.abs(np.diff(emg)) > WAMP_THRESH)
    PEAK     = np.max(np.abs(emg))
    ENV_MEAN = np.mean(env)
    ENV_STD  = np.std(env)
    ENV_MAX  = np.max(env)
    ENV_RANGE= np.max(env) - np.min(env)

    return [MAV, RMS, WL, VAR, IEMG, ZC, SSC, WAMP, PEAK,
            ENV_MEAN, ENV_STD, ENV_RANGE]

def extract_all_windows(csv_path):
    """Load a raw EMG CSV and extract features for every window."""
    df = pd.read_csv(csv_path, skiprows=4)
    emg = df["Raw_EMG"].values
    env = df["Envelope"].values

    rows = []
    for start in range(0, len(emg) - WINDOW_SIZE + 1, STRIDE):
        feats = extract_features(emg[start:start+WINDOW_SIZE],
                                 env[start:start+WINDOW_SIZE])
        rows.append(feats)

    return pd.DataFrame(rows, columns=FEATURES)

# ── Load model + scaler ───────────────────────────────────────────────────────

df_train = pd.read_excel(TRAIN_DATA)
le = LabelEncoder()
le.fit(df_train["label"])

scaler = StandardScaler()
scaler.fit(df_train[FEATURES].values)

n_classes = len(le.classes_)
model = nn.Sequential(
    nn.Linear(13, 64), nn.ReLU(),
    nn.Linear(64, 32), nn.ReLU(),
    nn.Linear(32, n_classes)
)
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model.eval()

print(f"Classes: {list(le.classes_)}")

# ── Connect to robot ──────────────────────────────────────────────────────────

arm = XArmAPI(ROBOT_IP)
arm.motion_enable(enable=True)
arm.set_mode(0)
arm.set_state(0)
arm.set_gripper_mode(0)
arm.set_gripper_enable(True)
print(f"Connected to xArm at {ROBOT_IP}")

print("Opening gripper...")
arm.set_gripper_position(GRIPPER_OPEN, wait=True)
# ── Predict & control ─────────────────────────────────────────────────────────

def predict(feature_row):
    x = scaler.transform([feature_row])
    x = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        pred = model(x).argmax(1).item()
    return le.classes_[pred]

def move_gripper(gesture):
    if gesture == "opening":
        print("→ Opening gripper")
        arm.set_gripper_position(GRIPPER_OPEN, wait=True)
    elif gesture == "closing":
        print("→ Closing gripper")
        arm.set_gripper_position(GRIPPER_CLOSED, wait=True)
    else:
        print("→ Rest — no movement")

# ── Main ──────────────────────────────────────────────────────────────────────

print(f"\nExtracting features from {SAMPLE_FILE}...")
sample_df = extract_all_windows(SAMPLE_FILE)
print(f"Got {len(sample_df)} windows — running predictions...\n")

for i, row in sample_df.iterrows():
    gesture = predict(row.values)
    print(f"Window {i+1}: {gesture}")
    move_gripper(gesture)

arm.disconnect()
print("\nDone.")