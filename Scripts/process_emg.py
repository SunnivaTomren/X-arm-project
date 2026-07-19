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
DATA_DIR   = r"C:\Users\celin\Documents\Mekatronikk\Prosjekt_UiA\Robot_Arm\X-arm-project\data\raw"
OUTPUT_DIR = r"C:\Users\celin\Documents\Mekatronikk\Prosjekt_UiA\Robot_Arm\X-arm-project\data\processed"

# ----------------------------------------------------------------------
# 1. INNSTILLINGER  -- juster disse
# ----------------------------------------------------------------------
MAPPER = {
    # mappe -> hvilken etikett de AKTIVE vinduene faar.
    # De flate vinduene i hver mappe blir automatisk 'rest'.
    os.path.join(DATA_DIR, "open_fist_celine"):    "opening",
    os.path.join(DATA_DIR, "closing_fist_celine"): "closing",
}

WIN              = 50    # vindusstorrelse i samples (50 @ 250Hz = 200 ms)
STEP             = 25    # steg mellom vinduer (25 = 50% overlapp)
UT_FIL           = os.path.join(OUTPUT_DIR, "features.xlsx")
BASELINE         = 512   # midtpunkt for 10-bit ADC (samme som EMG_to_xArm.py)
WAMP_THRESHOLD   = 10    # Willison-amplitude terskel (samme som EMG_to_xArm.py)

# Faste envelope-terskler (fra analyse av closing_fist_celine / open_fist_celine):
#   REST_ENV_LOW     = 99.9-persentil av malt hvile-envelope (527.6, rundet opp)
#   GESTURE_ENV_HIGH = laveste observerte bevegelses-topp (545.7, rundet ned)
# Brukes med hysterese i stedet for adaptiv median+MAD-terskel.
REST_ENV_LOW     = 528
GESTURE_ENV_HIGH = 546

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
    """Merker hvert sample som aktivt/hvile med hysterese mellom REST_ENV_LOW og
    GESTURE_ENV_HIGH (faste terskler), i stedet for adaptiv median+MAD-terskel."""
    aktiv = np.zeros(len(env), dtype=bool)
    state = False
    for i, v in enumerate(env):
        if v >= GESTURE_ENV_HIGH:
            state = True
        elif v < REST_ENV_LOW:
            state = False
        aktiv[i] = state
    return aktiv

# ----------------------------------------------------------------------
# 4. FEATURES PER VINDU  -- identiske med EMG_to_xArm.py slik at modellen
#    kan brukes direkte i live-inferens mot xArmen.
#
#    ENDRING: MAV/RMS/WL/VAR/IEMG/PEAK regnes naa ut fra ENVELOPE-signalet
#    i stedet for det raa signalet. Envelopen er allerede et glidende
#    gjennomsnitt av det rektifiserte raasignalet (se _compute_envelope
#    andre steder i prosjektet), saa den er langt mindre stoyfoelsom
#    sample-til-sample -- disse features bor derfor bli mer stabile
#    mellom opptak enn naar de regnes paa det raa signalet direkte.
#
#    ZC og WAMP regnes fortsatt paa RAASIGNALET, med vilje: testet empirisk
#    paa ekte data, og ZC/WAMP paa envelopen kollapser til konstant 0 for
#    ALLE vinduer (envelopen er saa glatt at den naermest aldri krysser
#    null eller hopper mer enn terskelen fra sample til sample) -- det
#    ville fjernet 2 av 12 features helt. SSC forblir paa envelopen; den
#    varierer fortsatt fint der.
# ----------------------------------------------------------------------
def features(vindu_raw, vindu_env):
    raw = vindu_raw.astype(float)
    xc  = raw - BASELINE      # raasignal, brukes kun til ZC/WAMP (se over)
    dx  = np.diff(xc)

    env = vindu_env.astype(float)
    ec  = env - BASELINE      # senter envelopen paa ADC-baseline
    de  = np.diff(ec)

    mav  = np.mean(np.abs(ec))
    rms  = np.sqrt(np.mean(ec ** 2))
    wl   = np.sum(np.abs(de))
    var  = np.var(ec)
    iemg = np.sum(np.abs(ec))
    zc   = np.sum((xc[:-1] * xc[1:] < 0) & (np.abs(dx) > 1))
    ssc  = np.sum((np.diff(np.sign(de)) != 0))
    wamp = np.sum(np.abs(dx) > WAMP_THRESHOLD)
    peak = np.max(np.abs(ec))

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