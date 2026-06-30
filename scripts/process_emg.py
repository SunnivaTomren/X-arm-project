"""
EMG-opprydding og feature-uttrekk
==================================
Leser alle CSV-opptak fra to mapper (bevegelse + rest), finner hvor musklene
faktisk er aktive, deler signalet i vinduer, regner ut features, og skriver
ALT til EN samlet fil: features.csv

Kjor:  python process_emg.py
Juster STIENE under til dine egne mapper.
"""

import os
import glob
import numpy as np
import pandas as pd

BASE       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE, "..", "data")
OUTPUT_DIR = os.path.join(BASE, "..", "output")

# ----------------------------------------------------------------------
# 1. INNSTILLINGER  -- juster disse
# ----------------------------------------------------------------------
MAPPER = {
    # mappe -> hvilken etikett de AKTIVE vinduene faar.
    # De flate vinduene i hver mappe blir automatisk 'rest'.
    os.path.join(DATA_DIR, "Open_fist"):    "opening",
    os.path.join(DATA_DIR, "closing_fist"): "closing",
}

WIN            = 50    # vindusstorrelse i samples (50 @ 250Hz = 200 ms)
STEP           = 25    # steg mellom vinduer (25 = 50% overlapp)
TERSKEL_STD    = 5     # hvor mange MAD over hvile for aa regnes som "aktiv"
UT_FIL         = os.path.join(OUTPUT_DIR, "features.xlsx")
BASELINE       = 512   # midtpunkt for 10-bit ADC (samme som EMG_to_xArm.py)
WAMP_THRESHOLD = 10    # Willison-amplitude terskel (samme som EMG_to_xArm.py)

# ----------------------------------------------------------------------
# 2. LES EN CSV-FIL (hopper over header-blokken)
# ----------------------------------------------------------------------
def les_opptak(sti):
    label = None
    fs = None
    data_start = None
    with open(sti) as f:
        linjer = f.readlines()
    for i, l in enumerate(linjer):
        deler = l.strip().split(",")
        if deler[0] == "Action_Label":
            label = deler[1]
        elif deler[0] == "Fs_Hz":
            fs = int(deler[1])
        elif deler[0] == "Sample_Index":
            data_start = i + 1
            break
    raw, env = [], []
    for l in linjer[data_start:]:
        d = l.strip().split(",")
        if len(d) == 3:
            raw.append(float(d[1]))
            env.append(float(d[2]))
    return label, fs, np.array(raw), np.array(env)

# ----------------------------------------------------------------------
# 3. FINN HVOR MUSKELEN ER AKTIV (via envelope + terskel)
# ----------------------------------------------------------------------
def finn_aktiv(env):
    baseline = np.median(env)
    mad = np.median(np.abs(env - baseline)) + 1e-9
    terskel = baseline + TERSKEL_STD * mad
    return env > terskel          # boolean array

# ----------------------------------------------------------------------
# 4. FEATURES PER VINDU  -- identiske med EMG_to_xArm.py slik at modellen
#    kan brukes direkte i live-inferens mot xArmen.
# ----------------------------------------------------------------------
def features(vindu_raw, vindu_env):
    x   = vindu_raw.astype(float)
    xc  = x - BASELINE
    env = vindu_env.astype(float)
    dx  = np.diff(xc)

    mav  = np.mean(np.abs(xc))
    rms  = np.sqrt(np.mean(xc ** 2))
    wl   = np.sum(np.abs(dx))
    var  = np.var(xc)
    iemg = np.sum(np.abs(xc))
    zc   = np.sum((xc[:-1] * xc[1:] < 0) & (np.abs(dx) > 1))
    ssc  = np.sum((np.diff(np.sign(dx)) != 0))
    wamp = np.sum(np.abs(dx) > WAMP_THRESHOLD)
    peak = np.max(np.abs(xc))

    env_mean = np.mean(env)
    env_std  = np.std(env)
    env_max  = np.max(env)
    env_rng  = np.max(env) - np.min(env)

    return [mav, rms, wl, var, iemg, zc, ssc, wamp, peak,
            env_mean, env_std, env_max, env_rng]

FEATURE_NAVN = ["MAV", "RMS", "WL", "VAR", "IEMG", "ZC", "SSC", "WAMP", "PEAK",
                "ENV_MEAN", "ENV_STD", "ENV_MAX", "ENV_RANGE"]

# ----------------------------------------------------------------------
# 5. HOVEDLOKKE
# ----------------------------------------------------------------------
rader = []
for mappe, aktiv_label in MAPPER.items():
    filer = glob.glob(os.path.join(mappe, "*.csv"))
    print(f"{mappe}: {len(filer)} filer")
    for sti in filer:
        file_label, fs, raw, env = les_opptak(sti)
        aktiv = finn_aktiv(env)
        opptak_id = os.path.basename(sti)

        # gli gjennom signalet i vinduer
        for start in range(0, len(raw) - WIN, STEP):
            slutt = start + WIN
            vindu_raw = raw[start:slutt]
            # er dette vinduet hovedsaklig aktivt eller hvile?
            andel_aktiv = aktiv[start:slutt].mean()

            if andel_aktiv > 0.5:
                klasse = aktiv_label       # opening / closing
            else:
                klasse = "rest"            # flate biter foer/etter bevegelsen

            rad = [opptak_id, klasse] + features(vindu_raw, env[start:slutt])
            rader.append(rad)

# ----------------------------------------------------------------------
# 6. LAGRE SAMLET FIL
# ----------------------------------------------------------------------
df = pd.DataFrame(rader, columns=["opptak_id", "label"] + FEATURE_NAVN)
df.to_excel(UT_FIL, index=False)

print("\nFerdig! Skrev", len(df), "vinduer til", UT_FIL)
print("\nFordeling per klasse:")
print(df["label"].value_counts())
