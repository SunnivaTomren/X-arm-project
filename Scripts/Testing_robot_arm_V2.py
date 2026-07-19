import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xarm.wrapper import XArmAPI

# ── Settings ──────────────────────────────────────────────────────────────────

ROBOT_IP    = "192.168.1.225"
MODEL_PATH  = "Models/emg_model_deep.pt"
TRAIN_DATA  = r"C:\Users\celin\Documents\Mekatronikk\Prosjekt_UiA\Robot_Arm\X-arm-project\data\processed\features.xlsx"
SAMPLE_FILE = r"C:\Users\celin\Documents\Mekatronikk\Prosjekt_UiA\Robot_Arm\X-arm-project\data\raw\closing_fist_celine\closing_fist_C_20260714_125316.csv"

# MUST match process_emg.py exactly -- these define how a raw signal gets
# turned into one feature row, and if they don't match what features.xlsx
# was built with, every feature ends up on a different scale than what the
# model learned from. This was previously 250/125/5 (a leftover from a
# different script's convention) while process_emg.py actually uses
# WIN=50, STEP=25, and a hardcoded zero-crossing threshold of 1 -- a 5x
# window-length mismatch that alone was enough to break every prediction.
WINDOW_SIZE = 50    # samples per window (200 ms at 250 Hz) -- = process_emg.py's WIN
STRIDE      = 25    # 50% overlap                            -- = process_emg.py's STEP
ZC_THRESH   = 1     # = process_emg.py's hardcoded zero-crossing threshold
WAMP_THRESH = 10    # Willison amplitude threshold
BASELINE    = 512   # ADC midpoint -- features.xlsx centers on this

# Must match Train_deep.py's FEATURES list exactly: same names, same order,
# same count (12) -- this is what the saved model's input layer expects.
FEATURES = ["MAV", "RMS", "WL", "VAR", "IEMG", "ZC", "SSC", "WAMP", "PEAK",
            "ENV_MEAN", "ENV_STD", "ENV_RANGE"]

GRIPPER_OPEN   = 850
GRIPPER_CLOSED = 0

# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(emg, env):
    """Extract features from one window. MAV/RMS/WL/VAR/IEMG/PEAK are
    computed from the ENVELOPE (smoother, less sample-to-sample noise
    than raw EMG) to match the updated process_emg.py. ZC and WAMP are
    kept on the RAW signal on purpose: tested empirically on real data,
    and ZC/WAMP computed on the envelope collapse to a constant 0 for
    every window (the envelope is too smooth to cross zero or jump past
    the threshold sample-to-sample) -- that would silently throw away 2
    of the 12 features. SSC stays on the envelope; it still varies fine
    there.
    """
    emg = np.array(emg, dtype=np.float32)
    xc  = emg - BASELINE      # raw signal, used only for ZC/WAMP (see above)
    dx  = np.diff(xc)

    env = np.array(env, dtype=np.float32)
    ec  = env - BASELINE
    de  = np.diff(ec)

    MAV      = np.mean(np.abs(ec))
    RMS      = np.sqrt(np.mean(ec ** 2))
    WL       = np.sum(np.abs(de))
    VAR      = np.var(ec)
    IEMG     = np.sum(np.abs(ec))
    ZC       = np.sum((xc[:-1] * xc[1:] < 0) & (np.abs(dx) > ZC_THRESH))
    SSC      = np.sum(np.diff(np.sign(de)) != 0)
    WAMP     = np.sum(np.abs(dx) > WAMP_THRESH)
    PEAK     = np.max(np.abs(ec))
    ENV_MEAN = np.mean(env)
    ENV_STD  = np.std(env)
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
#
# Everything below is rebuilt to exactly mirror Train_deep.py: same label
# encoding, the same train/test split (so `scaler` is fit on the SAME 80%
# of rows Train_deep.py fit it on, not the full dataset like before), and
# the same 10-layer network shape. Any mismatch here means load_state_dict()
# either errors on a shape mismatch, or -- worse -- silently loads and just
# predicts garbage because the weights don't mean what this architecture
# thinks they mean.

df_train = pd.read_excel(TRAIN_DATA)

le    = LabelEncoder()
y_all = le.fit_transform(df_train["label"])          # same call as Train_deep.py, same label order
X_all = df_train[FEATURES].values

# Train_deep.py fits StandardScaler on X_train only (an 80% split with
# random_state=42) -- NOT on the full dataset. Reproduce that exact split
# here so `scaler` has the same mean/scale the saved weights were trained
# against (fitting on 100% of the data, as before, gives slightly different
# statistics and quietly degrades predictions).
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42)
scaler = StandardScaler()
scaler.fit(X_train)

n_classes  = len(le.classes_)
n_features = len(FEATURES)
DROP = 0.3

# 10-layer architecture -- identical to the nn.Sequential in Train_deep.py
# (same layer sizes, BatchNorm, Dropout, and order). This was previously a
# 3-layer network with the wrong input size, which is why loading the
# saved state_dict would have failed or mismatched silently.
model = nn.Sequential(
    nn.Linear(n_features, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(DROP),
    nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(DROP),
    nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(DROP),
    nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(DROP),
    nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(DROP),
    nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(DROP),
    nn.Linear(128, 64),  nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(DROP),
    nn.Linear(64, 64),   nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(DROP),
    nn.Linear(64, 32),   nn.BatchNorm1d(32),  nn.ReLU(), nn.Dropout(DROP),
    nn.Linear(32, n_classes),
)
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model.eval()   # disables Dropout, freezes BatchNorm to its trained running stats

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

def predict(feature_row, verbose=False):
    x = scaler.transform([feature_row])
    x_t = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        logits = model(x_t)
        probs  = torch.softmax(logits, dim=1).numpy()[0]
        pred   = int(logits.argmax(1).item())
    if verbose:
        print(f"    raw features:    {np.round(np.asarray(feature_row, dtype=float), 2)}")
        print(f"    scaled features: {np.round(x[0], 2)}")
        print("    probabilities:   " +
              ", ".join(f"{c}={p:.3f}" for c, p in zip(le.classes_, probs)))
    return le.classes_[pred]

def move_gripper(gesture):
    # Match by substring, not exact string, in case your label set ever
    # includes a variant name (e.g. "closing_fist"). Currently your classes
    # are exactly ["closing", "opening", "rest"], so this also just works
    # with plain equality -- substring matching is only here to be robust
    # to future label changes.
    g = gesture.lower()
    if "open" in g:
        print(f"→ Opening gripper  (label: {gesture})")
        arm.set_gripper_position(GRIPPER_OPEN, wait=True)
    elif "clos" in g:
        print(f"→ Closing gripper  (label: {gesture})")
        arm.set_gripper_position(GRIPPER_CLOSED, wait=True)
    else:
        print(f"→ Rest — no movement  (label: {gesture})")

# ── Main ──────────────────────────────────────────────────────────────────────

print(f"\nExtracting features from {SAMPLE_FILE}...")
sample_df = extract_all_windows(SAMPLE_FILE)
print(f"Got {len(sample_df)} windows — running predictions...\n")

for i, row in sample_df.iterrows():
    print(f"Window {i+1}:")
    gesture = predict(row.values, verbose=True)
    print(f"  -> predicted: {gesture}")
    move_gripper(gesture)

arm.disconnect()
print("\nDone.")