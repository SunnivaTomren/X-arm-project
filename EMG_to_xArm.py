"""
emg_to_xarm_ml.py
=================================================================
Connect EMG recordings (stored in Excel files) to a machine-learning
classifier, and use that classifier to drive a UFACTORY xArm.

The script has two modes:

    python emg_to_xarm_ml.py train  --excel data/*.xlsx
    python emg_to_xarm_ml.py run    --model emg_model.joblib --port COM4

------------------------------------------------------------------
HOW TRAINING DATA IS ORGANISED
------------------------------------------------------------------
The Excel file(s) don't exist yet -- this script is written so you can drop
them in later with no code changes. You said there will be 4 datasets; the
intended setup is **one Excel file per gesture**, e.g.:

    closing_fist.xlsx
    opening_hand.xlsx
    rest.xlsx
    wave.xlsx

Train on all of them in one call:

    python emg_to_xarm_ml.py train --excel closing_fist.xlsx opening_hand.xlsx rest.xlsx wave.xlsx
    python emg_to_xarm_ml.py train --excel "data/*.xlsx"     (a glob also works, quote it)

Each Excel's gesture label is taken from its filename (e.g. "closing_fist.xlsx"
-> label "closing_fist"). If a single Excel already mixes several recordings
internally (one row per sample, many recordings stacked, with a Source_File
column identifying each recording -- like a combined export), each
recording's label is parsed from Source_File instead. Both styles can be
mixed freely in the same --excel list.

------------------------------------------------------------------
EXPECTED EXCEL FORMAT  (one row per sample)
------------------------------------------------------------------
    Source_File          Sample_Index   Raw_EMG   Envelope   Active/non Active
    closing_fist_*.csv   0              489.0     527.8      (optional)
    ...

Only Raw_EMG is required. Sample_Index, Envelope, Source_File, and the
Active/non Active column are all optional / auto-filled if missing.
=================================================================
"""

import os
import re
import sys
import argparse
import numpy as np
import pandas as pd
import joblib

# ------------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------------
SAMPLING_RATE_HZ = 250          # Olimex acquisition rate (samples per second)
BASELINE         = 512          # mid-point of the 10-bit ADC (raw signal sits here at rest)

WINDOW_SAMPLES   = 250          # 1.0 s analysis window  -> one feature vector
WINDOW_OVERLAP   = 0.5          # 50 % overlap between consecutive windows

# How to derive the class label for each recording:
#   "filename"  -> gesture name parsed from Source_File   (default, multi-gesture)
#   "column"    -> use the "Active/non Active" column     (active vs rest within a file)
LABEL_MODE       = "filename"

WAMP_THRESHOLD   = 10           # Willison-amplitude threshold (ADC units)


# ------------------------------------------------------------------
# 2. LOADING THE EXCEL FILE(S)
# ------------------------------------------------------------------
def load_excel(path):
    """Read ONE Excel file into a clean DataFrame with numeric columns.

    If the file has no usable Source_File column (i.e. it's a single
    recording / single-gesture file, which is the expected case for your
    4 upcoming datasets), Source_File is filled in from the filename so
    every row still has a recording id to group by.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Excel file not found: {path}")

    df = pd.read_excel(path, sheet_name=0)

    # Normalise column names (strip spaces / unify the label column)
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        lc = c.lower()
        if lc.startswith("source"):        rename[c] = "Source_File"
        elif lc.startswith("sample"):      rename[c] = "Sample_Index"
        elif lc.startswith("raw"):         rename[c] = "Raw_EMG"
        elif lc.startswith("envelope"):    rename[c] = "Envelope"
        elif "active" in lc:               rename[c] = "Active"
    df = df.rename(columns=rename)

    if "Raw_EMG" not in df.columns:
        raise ValueError(
            f"'{path}': couldn't find a Raw_EMG column. "
            f"Found columns: {list(df.columns)}"
        )

    # Values may come in as strings ('489.0') -> force numeric
    for col in ("Sample_Index", "Raw_EMG", "Envelope"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Raw_EMG"]).reset_index(drop=True)

    if "Sample_Index" not in df.columns:
        df["Sample_Index"] = np.arange(len(df))

    if "Source_File" not in df.columns or df["Source_File"].isna().all():
        # Single-recording file -> use the Excel's own filename as the
        # recording id. This is the expected case for your 4 datasets,
        # one Excel per gesture (e.g. closing_fist.xlsx).
        df["Source_File"] = os.path.basename(path)

    if "Envelope" not in df.columns or df["Envelope"].isna().all():
        # If no envelope was stored, compute one (moving mean-absolute-value),
        # per recording so windows don't bleed across file boundaries.
        df["Envelope"] = (
            df.groupby("Source_File")["Raw_EMG"]
              .transform(lambda s: _compute_envelope(s.to_numpy()))
        )

    # GESTURE LABEL.
    # Case A (expected for your 4 datasets): one Excel = one gesture, named
    # after the file itself, e.g. closing_fist.xlsx -> "closing_fist".
    # Case B: a combined export where Source_File values look like
    # "<gesture>_<timestamp>.csv" (multiple gestures stacked in one Excel) ->
    # derive the label per-row from Source_File instead.
    looks_combined = df["Source_File"].astype(str).str.contains(
        r"_\d{8}_\d{6}\.csv$", regex=True, na=False
    ).any()
    if looks_combined:
        df["Gesture_Label"] = df["Source_File"].apply(label_from_filename)
    else:
        df["Gesture_Label"] = label_from_filename(os.path.basename(path))

    print(f"  Loaded '{os.path.basename(path)}': {len(df):,} samples, "
          f"{df['Source_File'].nunique()} recording(s)  ->  "
          f"gesture label(s): {sorted(df['Gesture_Label'].unique())}")
    return df


def load_excels(paths):
    """
    Load and concatenate multiple Excel files (one per dataset/gesture, or
    combined files -- any mix is fine). `paths` may contain glob patterns.
    """
    import glob

    resolved = []
    for p in paths:
        matches = sorted(glob.glob(p))
        resolved.extend(matches if matches else [p])   # keep literal path if no glob match

    resolved = list(dict.fromkeys(resolved))  # dedupe, keep order
    if not resolved:
        raise FileNotFoundError(f"No Excel files matched: {paths}")

    print(f"Loading {len(resolved)} Excel file(s):")
    frames = [load_excel(p) for p in resolved]
    df = pd.concat(frames, ignore_index=True)
    print(f"Total: {len(df):,} samples from {df['Source_File'].nunique()} recording(s).\n")
    return df


def _compute_envelope(raw, window=30):
    """Mean-absolute-value envelope around the baseline (same idea as the acq script)."""
    rect = np.abs(raw - BASELINE)
    env = pd.Series(rect).rolling(window, min_periods=1).mean().to_numpy() + BASELINE
    return env


def label_from_filename(name):
    """
    Derive a gesture label from a recording/file name. Handles both:
      'closing_fist.xlsx'                 -> 'closing_fist'   (one Excel per gesture)
      'closing_fist_20260531_104945.csv'  -> 'closing_fist'   (combined export row)
    """
    base = re.sub(r"_\d{8}_\d{6}(\.csv|\.xlsx)?$", "", str(name))  # drop _DATE_TIME[.ext]
    base = re.sub(r"\.(csv|xlsx)$", "", base)                       # drop a bare extension
    return base.strip() or "unknown"


# ------------------------------------------------------------------
# 3. FEATURE EXTRACTION  (time-domain EMG features per window)
# ------------------------------------------------------------------
def extract_features(raw_window, env_window):
    """
    Compute a fixed-length feature vector from one window of EMG.
    These are the classic time-domain features used for sEMG gesture
    recognition (Hudgins set + a few envelope statistics).
    """
    x   = np.asarray(raw_window, dtype=float)
    xc  = x - BASELINE                       # centre the raw signal at 0
    env = np.asarray(env_window, dtype=float)
    dx  = np.diff(xc)

    mav  = np.mean(np.abs(xc))                                   # Mean Absolute Value
    rms  = np.sqrt(np.mean(xc ** 2))                             # Root Mean Square
    wl   = np.sum(np.abs(dx))                                    # Waveform Length
    var  = np.var(xc)                                            # Variance
    iemg = np.sum(np.abs(xc))                                    # Integrated EMG
    zc   = np.sum((xc[:-1] * xc[1:] < 0) & (np.abs(dx) > 1))     # Zero Crossings
    ssc  = np.sum((np.diff(np.sign(dx)) != 0))                   # Slope Sign Changes
    wamp = np.sum(np.abs(dx) > WAMP_THRESHOLD)                   # Willison Amplitude
    peak = np.max(np.abs(xc))                                    # Peak amplitude

    env_mean = np.mean(env)
    env_std  = np.std(env)
    env_max  = np.max(env)
    env_rng  = np.max(env) - np.min(env)

    return [mav, rms, wl, var, iemg, zc, ssc, wamp, peak,
            env_mean, env_std, env_max, env_rng]


FEATURE_NAMES = ["MAV", "RMS", "WL", "VAR", "IEMG", "ZC", "SSC", "WAMP", "PEAK",
                 "ENV_MEAN", "ENV_STD", "ENV_MAX", "ENV_RANGE"]


def build_dataset(df):
    """
    Slide a window across every recording and turn each window into one
    feature row. Returns X (features), y (labels), groups (source file id).
    """
    step = max(1, int(WINDOW_SAMPLES * (1 - WINDOW_OVERLAP)))
    X, y, groups = [], [], []

    for source, g in df.groupby("Source_File", sort=False):
        g = g.sort_values("Sample_Index")
        raw = g["Raw_EMG"].to_numpy()
        env = g["Envelope"].to_numpy()

        if LABEL_MODE == "column" and "Active" in g.columns and g["Active"].notna().any():
            labels_series = g["Active"].astype(str).to_numpy()
        else:
            labels_series = None
        # Per-Excel-file gesture label, set once in load_excel(). Falls back to
        # parsing the recording id itself only if Gesture_Label is missing
        # (shouldn't happen via load_excel, but keeps this function robust
        # if called directly on a hand-built DataFrame).
        file_label = g["Gesture_Label"].iloc[0] if "Gesture_Label" in g.columns \
            else label_from_filename(source)

        for start in range(0, len(raw) - WINDOW_SAMPLES + 1, step):
            end = start + WINDOW_SAMPLES
            X.append(extract_features(raw[start:end], env[start:end]))
            if labels_series is not None:
                # majority label inside the window
                vals, counts = np.unique(labels_series[start:end], return_counts=True)
                y.append(vals[np.argmax(counts)])
            else:
                y.append(file_label)
            groups.append(source)

    X = np.array(X)
    y = np.array(y)
    groups = np.array(groups)
    print(f"Built {len(X)} windows  |  {X.shape[1]} features  |  "
          f"{len(np.unique(y))} class(es): {sorted(np.unique(y))}")
    return X, y, groups


# ------------------------------------------------------------------
# 4. TRAIN + EVALUATE
# ------------------------------------------------------------------
def train(excel_paths, model_out):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    df = load_excels(excel_paths)
    X, y, groups = build_dataset(df)

    classes = np.unique(y)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=None,
            class_weight="balanced", random_state=42, n_jobs=-1)),
    ])

    if len(classes) < 2:
        # Only one gesture present -> we can't *classify*, but we can still fit
        # the model so the pipeline is ready for when more gestures are added.
        print("\n  WARNING: only ONE gesture class is present in this Excel file.")
        print("  A classifier needs at least two gestures (e.g. add 'rest' and")
        print("  'opening_hand' recordings). Fitting a single-class model anyway")
        print("  so the pipeline is validated; accuracy cannot be evaluated.\n")
        pipe.fit(X, y)
    else:
        # GroupKFold so windows from the SAME recording never leak across folds
        n_splits = min(5, len(np.unique(groups)), min(np.bincount(
            pd.factorize(y)[0])))
        n_splits = max(2, n_splits)
        cv = GroupKFold(n_splits=n_splits)
        y_pred = cross_val_predict(pipe, X, y, groups=groups, cv=cv, n_jobs=-1)

        print(f"\n=== {n_splits}-fold GroupKFold cross-validation ===")
        print(f"Accuracy: {accuracy_score(y, y_pred):.3f}\n")
        print(classification_report(y, y_pred))
        print("Confusion matrix (rows=true, cols=pred):")
        print("labels:", list(classes))
        print(confusion_matrix(y, y_pred, labels=classes))

        # Refit on ALL data for the final saved model
        pipe.fit(X, y)

        # Feature importances (handy for your report)
        imp = pipe.named_steps["clf"].feature_importances_
        order = np.argsort(imp)[::-1]
        print("\nFeature importances:")
        for i in order:
            print(f"  {FEATURE_NAMES[i]:>10}: {imp[i]:.3f}")

    bundle = {
        "pipeline": pipe,
        "feature_names": FEATURE_NAMES,
        "classes": list(pipe.named_steps["clf"].classes_),
        "window_samples": WINDOW_SAMPLES,
        "baseline": BASELINE,
        "sampling_rate": SAMPLING_RATE_HZ,
    }
    joblib.dump(bundle, model_out)
    print(f"\nModel saved to: {os.path.abspath(model_out)}")
    return bundle


# ------------------------------------------------------------------
# 5. REAL-TIME INFERENCE  +  xARM CONTROL
# ------------------------------------------------------------------
# Map each predicted gesture to an xArm action. Edit freely.
# 'gripper' positions for the UFACTORY gripper: 0 = closed, 850 = open.
GESTURE_TO_ACTION = {
    "closing_fist":  {"gripper": 0},      # close gripper
    "opening_hand":  {"gripper": 850},    # open gripper
    "rest":          None,                 # do nothing
}


class XArmController:
    """Thin, SAFETY-gated wrapper around the official xArm-Python-SDK."""

    def __init__(self, ip, enable_motion=False):
        self.enable_motion = enable_motion          # must be True to actually move
        self.arm = None
        if not enable_motion:
            print("xArm controller in DRY-RUN mode (no motion). "
                  "Pass --enable-motion to move the real arm.")
            return
        try:
            from xarm.wrapper import XArmAPI
        except ImportError:
            raise ImportError("Install the SDK first:  pip install xarm-python-sdk")
        self.arm = XArmAPI(ip)
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(0)
        self.arm.set_gripper_mode(0)
        self.arm.set_gripper_enable(True)
        self.arm.set_gripper_position(850, wait=True)   # start open
        print(f"xArm connected at {ip} and ready.")

    def do(self, gesture):
        action = GESTURE_TO_ACTION.get(gesture)
        if action is None:
            return
        if "gripper" in action:
            pos = action["gripper"]
            print(f"  -> gesture '{gesture}': set gripper to {pos}")
            if self.enable_motion and self.arm is not None:
                self.arm.set_gripper_position(pos, wait=False)

    def close(self):
        if self.enable_motion and self.arm is not None:
            self.arm.disconnect()


def run_live(model_path, serial_port, baud, arm_ip, enable_motion,
             vote_window=5):
    """
    Read live EMG from the Olimex board, classify a rolling window, and
    command the xArm. Uses majority voting to debounce predictions.
    """
    import serial, struct, time
    from collections import deque

    bundle = joblib.load(model_path)
    pipe   = bundle["pipeline"]
    win_n  = bundle["window_samples"]

    arm = XArmController(arm_ip, enable_motion=enable_motion)

    ser = serial.Serial(serial_port, baud, timeout=0.01)
    buf = bytearray()
    PACKET_SIZE, CH = 17, 0

    raw_window = deque(maxlen=win_n)
    votes      = deque(maxlen=vote_window)
    last_action = None

    print("Streaming... press Ctrl-C to stop.")
    try:
        while True:
            if ser.in_waiting:
                buf += ser.read(ser.in_waiting)
            while len(buf) >= PACKET_SIZE:
                if buf[0] == 0xA5 and buf[1] == 0x5A:
                    val = struct.unpack(">6H", buf[4:16])[CH]
                    raw_window.append(float(val))
                    buf = buf[PACKET_SIZE:]
                else:
                    buf.pop(0)

            if len(raw_window) == win_n:
                raw = np.array(raw_window)
                env = _compute_envelope(raw)
                feats = np.array(extract_features(raw, env)).reshape(1, -1)
                pred = pipe.predict(feats)[0]
                votes.append(pred)

                # act only when the vote is stable and the action changed
                if len(votes) == votes.maxlen:
                    vals, counts = np.unique(votes, return_counts=True)
                    winner = vals[np.argmax(counts)]
                    if winner != last_action:
                        arm.do(winner)
                        last_action = winner
            time.sleep(0.005)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        ser.close()
        arm.close()


# ------------------------------------------------------------------
# 6. COMMAND-LINE INTERFACE
# ------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="EMG -> ML -> xArm pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train", help="train a model from one or more Excel files")
    pt.add_argument("--excel", required=True, nargs="+", #This is where u add your excel file names
                     help="one or more Excel files (or glob patterns), "
                          "e.g. --excel closing_fist.xlsx opening_hand.xlsx rest.xlsx wave.xlsx")
    pt.add_argument("--model", default="emg_model.joblib")

    pr = sub.add_parser("run", help="live EMG -> xArm")
    pr.add_argument("--model", default="emg_model.joblib")
    pr.add_argument("--port",  default="COM4", help="serial port of the Olimex board")
    pr.add_argument("--baud",  type=int, default=57600)
    pr.add_argument("--arm-ip", default="192.168.1.215", help="xArm controller IP")
    pr.add_argument("--enable-motion", action="store_true",
                    help="actually move the arm (default is a safe dry run)")

    args = p.parse_args()
    if args.cmd == "train":
        train(args.excel, args.model)
    elif args.cmd == "run":
        run_live(args.model, args.port, args.baud, args.arm_ip, args.enable_motion)


if __name__ == "__main__":
    main()