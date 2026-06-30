"""
EMG Prediksjon  -- test modellen på én enkelt CSV-fil
======================================================
Velger en tilfeldig fil fra en mappe, kjører modellen vindu for vindu,
og viser prediksjonen som en tidslinje + plot.
"""

import os
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE, "..", "data")
OUTPUT_DIR = os.path.join(BASE, "..", "output")

# -----------------------------------------------------------------------
# INNSTILLINGER -- same verdier som i process_emg.py
# -----------------------------------------------------------------------
MAPPE         = os.path.join(DATA_DIR, "Open_fist")   # <-- bytt til closing_fist, hiya_down osv.
WIN           = 50
STEP          = 25
BASELINE      = 512
WAMP_THRESHOLD = 10
MODELL_FIL    = os.path.join(OUTPUT_DIR, "emg_model.pt")

# -----------------------------------------------------------------------
# 1. LAST INN MODELLEN OG SCALER
# -----------------------------------------------------------------------
checkpoint = torch.load(MODELL_FIL, map_location="cpu")
KLASSER    = checkpoint["klasser"]
scaler_mean = np.array(checkpoint["scaler_mean"], dtype=np.float32)
scaler_std  = np.array(checkpoint["scaler_std"],  dtype=np.float32)

class EMGClassifier(nn.Module):
    def __init__(self, n_features, n_klasser):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),         nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32),          nn.ReLU(),
            nn.Linear(32, n_klasser),
        )
    def forward(self, x):
        return self.net(x)

model = EMGClassifier(13, len(KLASSER))
model.load_state_dict(checkpoint["model_state"])
model.eval()
print(f"Modell lastet. Klasser: {KLASSER}")

# -----------------------------------------------------------------------
# 2. VELG EN TILFELDIG CSV-FIL
# -----------------------------------------------------------------------
filer = glob.glob(os.path.join(MAPPE, "*.csv"))
if not filer:
    raise FileNotFoundError(f"Ingen CSV-filer funnet i {MAPPE}")

valgt_fil = random.choice(filer)
print(f"\nValgt fil: {os.path.basename(valgt_fil)}")

# Les filen (samme format som process_emg.py)
with open(valgt_fil) as f:
    linjer = f.readlines()

action_label = None
data_start   = None
for i, l in enumerate(linjer):
    deler = l.strip().split(",")
    if deler[0] == "Action_Label":
        action_label = deler[1]
    elif deler[0] == "Sample_Index":
        data_start = i + 1
        break

raw, env = [], []
for l in linjer[data_start:]:
    d = l.strip().split(",")
    if len(d) == 3:
        raw.append(float(d[1]))
        env.append(float(d[2]))

raw = np.array(raw)
env = np.array(env)
print(f"Ekte label: '{action_label}' | {len(raw)} samples @ 250 Hz = {len(raw)/250:.1f} sek")

# -----------------------------------------------------------------------
# 3. FEATURE-UTTREKK (identisk med train_emg.py og process_emg.py)
# -----------------------------------------------------------------------
def features(vindu_raw, vindu_env):
    x   = vindu_raw.astype(float)
    xc  = x - BASELINE
    dx  = np.diff(xc)
    env = vindu_env.astype(float)

    mav  = np.mean(np.abs(xc))
    rms  = np.sqrt(np.mean(xc ** 2))
    wl   = np.sum(np.abs(dx))
    var  = np.var(xc)
    iemg = np.sum(np.abs(xc))
    zc   = np.sum((xc[:-1] * xc[1:] < 0) & (np.abs(dx) > 1))
    ssc  = np.sum((np.diff(np.sign(dx)) != 0))
    wamp = np.sum(np.abs(dx) > WAMP_THRESHOLD)
    peak = np.max(np.abs(xc))

    return [mav, rms, wl, var, iemg, zc, ssc, wamp, peak,
            np.mean(env), np.std(env), np.max(env), np.max(env) - np.min(env)]

# -----------------------------------------------------------------------
# 4. KJØR MODELLEN VINDU FOR VINDU
# -----------------------------------------------------------------------
vinduer   = []   # (start_sample, slutt_sample)
pred_klasse  = []
pred_sikkerhet = []

for start in range(0, len(raw) - WIN, STEP):
    slutt = start + WIN
    feat  = np.array(features(raw[start:slutt], env[start:slutt]), dtype=np.float32)
    feat  = (feat - scaler_mean) / scaler_std          # normaliser

    with torch.no_grad():
        logits = model(torch.tensor(feat).unsqueeze(0))
        probs  = torch.softmax(logits, dim=1).squeeze().numpy()
        pred   = int(probs.argmax())

    vinduer.append((start, slutt))
    pred_klasse.append(KLASSER[pred])
    pred_sikkerhet.append(probs[pred] * 100)

# -----------------------------------------------------------------------
# 5. SKRIV TIDSLINJE TIL TERMINALEN
# -----------------------------------------------------------------------
FS = 250
print("\n--- PREDIKSJON VINDU FOR VINDU ---")
print(f"{'Tid (s)':<12} {'Predikert':<15} {'Sikkerhet'}")
print("-" * 40)
for (start, slutt), klasse, sikkerhet in zip(vinduer, pred_klasse, pred_sikkerhet):
    tid_start = start / FS
    tid_slutt = slutt / FS
    print(f"{tid_start:.2f}-{tid_slutt:.2f}s  {klasse:<15} {sikkerhet:.0f}%")

# -----------------------------------------------------------------------
# 6. PLOT: RAW SIGNAL + FARGEKODEDE PREDIKSJONER
# -----------------------------------------------------------------------
FARGER = {
    "rest":    "lightgray",
    "opening": "steelblue",
    "closing": "tomato",
    "hiya_down": "gold",
    "hiya_up":   "mediumseagreen",
    "lifting_cup": "mediumpurple",
}

fig, ax = plt.subplots(figsize=(14, 5))
tid_akse = np.arange(len(raw)) / FS
ax.plot(tid_akse, raw, color="black", linewidth=0.8, alpha=0.8, label="Raw EMG")

for (start, slutt), klasse in zip(vinduer, pred_klasse):
    farge = FARGER.get(klasse, "orange")
    ax.axvspan(start / FS, slutt / FS, alpha=0.25, color=farge)

# Legende
lapper = [mpatches.Patch(color=FARGER.get(k, "orange"), label=k, alpha=0.5)
          for k in sorted(set(pred_klasse))]
ax.legend(handles=lapper, loc="upper right")

ax.set_xlabel("Tid (sekunder)")
ax.set_ylabel("Raw EMG")
ax.set_title(f"Prediksjon | Fil: {os.path.basename(valgt_fil)} | Ekte label: '{action_label}'")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "resultat_prediksjon.png"), dpi=120)
plt.show()
print(f"\nPlot lagret som {os.path.join(OUTPUT_DIR, 'resultat_prediksjon.png')}")
