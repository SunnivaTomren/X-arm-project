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

# ----------------------------------------------------------------------
# 1. INNSTILLINGER  -- juster disse
# ----------------------------------------------------------------------
MAPPER = {
    # mappe -> hvilken etikett de AKTIVE vinduene faar.
    # De flate vinduene i hver mappe blir automatisk 'rest'.
    "EMG-data/opening": "opening",
    "EMG-data/closing": "closing",
}

WIN          = 50    # vindusstorrelse i samples (50 @ 250Hz = 200 ms)
STEP         = 25    # steg mellom vinduer (25 = 50% overlapp)
TERSKEL_STD  = 5     # hvor mange MAD over hvile for aa regnes som "aktiv"
UT_FIL       = "features.xlsx"

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
# 4. FEATURES PER VINDU (standard time-domain EMG-features)
# ----------------------------------------------------------------------
def features(vindu):
    v = vindu.astype(float)
    senter = v - np.mean(v)
    rms = np.sqrt(np.mean(senter**2))
    mav = np.mean(np.abs(senter))
    wl  = np.sum(np.abs(np.diff(v)))            # waveform length
    zc  = np.sum(np.diff(np.sign(senter)) != 0) # zero crossings
    var = np.var(senter)
    return [rms, mav, wl, zc, var]

FEATURE_NAVN = ["rms", "mav", "wl", "zc", "var"]

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

            rad = [opptak_id, klasse] + features(vindu_raw)
            rader.append(rad)

# ----------------------------------------------------------------------
# 6. LAGRE SAMLET FIL
# ----------------------------------------------------------------------
df = pd.DataFrame(rader, columns=["opptak_id", "label"] + FEATURE_NAVN)
df.to_excel(UT_FIL, index=False)

print("\nFerdig! Skrev", len(df), "vinduer til", UT_FIL)
print("\nFordeling per klasse:")
print(df["label"].value_counts())
