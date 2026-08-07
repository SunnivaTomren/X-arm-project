import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xarm.wrapper import XArmAPI

# ── Settings ──────────────────────────────────────────────────────────────────

ROBOT_IP = "192.168.1.225"

# Paths are resolved relative to this file, not the working directory, so the
# script runs the same from any cwd and on any machine that has the repo.
_here       = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(_here, "..", "Models", "emg_model_deep.pt")
TRAIN_DATA  = os.path.join(_here, "..", "data", "processed", "features.xlsx")

# MUST match process_emg.py exactly -- these define how a raw signal gets
# turned into one feature row, and if they don't match what features.xlsx
# was built with, every feature ends up on a different scale than what the
# model learned from. This was previously 250/125/5 (a leftover from a
# different script's convention) while process_emg.py actually uses
# WIN=50, STEP=25, and a hardcoded zero-crossing threshold of 1 -- a 5x
# window-length mismatch that alone was enough to break every prediction.
FS          = 250   # sampling rate (Hz) -- needed for MNF/MDF; = process_emg.py's FS
WINDOW_SIZE = 50    # samples per window (200 ms at 250 Hz) -- = process_emg.py's WIN
STRIDE      = 25    # 50% overlap                            -- = process_emg.py's STEP
ZC_THRESH   = 1     # = process_emg.py's hardcoded zero-crossing threshold
WAMP_THRESH = 10    # = process_emg.py's WAMP_THRESHOLD
BASELINE    = 512   # ADC midpoint -- features.xlsx centers on this

# Must match Train_deep.py's FEATURES list exactly: same names, same order,
# same count (7) -- this is what the saved model's input layer expects.
# The old 12/13-feature amplitude set was cut down in process_emg.py because
# MAV/IEMG/ENV_MEAN and PEAK/ENV_MAX were near-duplicates (r≈1.00); MNF/MDF
# were added in their place as amplitude-independent frequency features.
FEATURES = ["ENV_MEAN", "ENV_STD", "WAMP", "ZC", "SSC", "MNF", "MDF"]

GRIPPER_OPEN   = 850
GRIPPER_CLOSED = 0

# ── Feature extraction ────────────────────────────────────────────────────────de for how we gathered samples looks like this

def _mnf_mdf(xc):
    """Mean and median frequency of the centred RAW signal's power spectrum.
    Both are ratios within the spectrum, so they are amplitude-independent --
    that is exactly why they separate opening from closing where the pure
    amplitude features could not. Copied verbatim from process_emg.py.
    """
    n = len(xc)
    freqs = np.fft.rfftfreq(n, d=1.0 / FS)
    P = np.abs(np.fft.rfft(xc)) ** 2
    P[0] = 0.0                              # drop the DC component
    tot = P.sum()
    if tot <= 0:
        return 0.0, 0.0
    mnf = float(np.sum(freqs * P) / tot)
    mdf = float(freqs[np.searchsorted(np.cumsum(P), tot / 2)])
    return mnf, mdf

def extract_features(emg, env):
    """Extract one feature row for one window -- must stay numerically
    identical to process_emg.py's features(), since that is what built the
    features.xlsx the model was trained and scaled on.

    Which signal each feature comes from is deliberate: WAMP/ZC/MNF/MDF use
    the RAW signal, because on the envelope they collapse (it is far too
    smooth to cross zero, exceed the Willison threshold, or carry usable
    spectral content). ENV_MEAN/ENV_STD/SSC use the ENVELOPE.
    """
    raw = np.asarray(emg, dtype=float)
    xc  = raw - BASELINE          # raw signal centred on 0
    dx  = np.diff(xc)

    env = np.asarray(env, dtype=float)
    ec  = env - BASELINE          # envelope centred on the ADC baseline
    de  = np.diff(ec)

    ENV_MEAN = np.mean(env)
    ENV_STD  = np.std(env)
    WAMP     = np.sum(np.abs(dx) > WAMP_THRESH)
    ZC       = np.sum((xc[:-1] * xc[1:] < 0) & (np.abs(dx) > ZC_THRESH))
    SSC      = np.sum(np.diff(np.sign(de)) != 0)
    MNF, MDF = _mnf_mdf(xc)

    return [ENV_MEAN, ENV_STD, WAMP, ZC, SSC, MNF, MDF]

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

# Train_deep.py fits StandardScaler on X_train only -- NOT on the full
# dataset -- so reproduce its exact split here, or `scaler` ends up with
# different mean/scale than the saved weights were trained against.
#
# The split is GroupShuffleSplit on opptak_id, not train_test_split: windows
# overlap by 50%, so neighbouring rows from the same recording are near
# duplicates, and grouping by recording keeps them out of both halves at
# once. Using the old plain train_test_split here would pick a different 80%
# and shift every feature's mean/scale.
groups = df_train["opptak_id"].values
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, _ = next(gss.split(X_all, y_all, groups))
scaler = StandardScaler()
scaler.fit(X_all[train_idx])

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
state = torch.load(MODEL_PATH, weights_only=True)

# Fail loudly if the checkpoint was trained on a different feature count than
# FEATURES above. This is the exact bug this file just had (12 features here
# vs 7 in Train_deep.py), and without the check a stale .pt file gives a wall
# of PyTorch shape errors instead of pointing at the real cause.
saved_n_features = state["0.weight"].shape[1]
if saved_n_features != n_features:
    raise SystemExit(
        f"{MODEL_PATH} expects {saved_n_features} input features, but FEATURES "
        f"lists {n_features} ({FEATURES}).\nRe-run Train_deep.py to regenerate "
        f"the model, or align FEATURES with the checkpoint.")

model.load_state_dict(state)
model.eval()   # disables Dropout, freezes BatchNorm to its trained running stats

print(f"Classes: {list(le.classes_)}")
print(f"Features ({n_features}): {FEATURES}")

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
        print(f"-> Opening gripper  (label: {gesture})")
        arm.set_gripper_position(GRIPPER_OPEN, wait=True)
    elif "clos" in g:
        print(f"-> Closing gripper  (label: {gesture})")
        arm.set_gripper_position(GRIPPER_CLOSED, wait=True)
    else:
        print(f"-> Rest -- no movement  (label: {gesture})")

# This file is a module only -- it loads the model and connects to the robot
# on import, and is driven live by live_robot_control.py.