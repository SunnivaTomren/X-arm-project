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
DATA_DIR   = os.path.join(BASE, "..", "data", "raw")
OUTPUT_DIR = os.path.join(BASE, "..", "data", "processed")

# ----------------------------------------------------------------------
# 1. INNSTILLINGER  -- juster disse
# ----------------------------------------------------------------------
MAPPER = {
    # mappe -> hvilken etikett de AKTIVE vinduene faar.
    # De flate vinduene i hver mappe blir automatisk 'rest'.
    os.path.join(DATA_DIR, "open_fist_celine"):    "opening",
    os.path.join(DATA_DIR, "closing_fist_celine"): "closing",
}

FS               = 250   # samplingsrate (Hz) -- brukes i frekvens-features (MNF/MDF)
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
# 4. FEATURES PER VINDU
#
#    Ryddet + utvidet sett paa 7 features (var 13). De gamle 13 var i praksis
#    bare ~4 uavhengige: MAV = IEMG = ENV_MEAN (r=1.00), PEAK = ENV_MAX (r=1.00),
#    RMS/WL/ENV_RANGE alle r>0.9 -- 9 features som malte SAMME amplitude. Fjernet.
#    I stedet er MNF/MDF (mean/median frequency) lagt til; de er amplitude-
#    uavhengige og loefter opening-vs-closing fra 62% (amplitude alene) til 90%.
#    Sjekk begrunnelsen i feature_selection_report.py.
#
#      ENV_MEAN  amplitude (envelope) -- sterkest for ALLE gester, ogsaa rest
#      ENV_STD   hvor mye envelopen varierer i vinduet
#      WAMP      Willison-amplitude paa RAASIGNALET (aktivitetsteller)
#      ZC        nullkryssinger paa RAASIGNALET (frekvens-proxy)
#      SSC       stigningsskift paa ENVELOPEN
#      MNF/MDF   mean/median frequency fra spekteret til RAASIGNALET
#
#    ZC/WAMP/MNF/MDF regnes paa RAASIGNALET med vilje: paa envelopen kollapser
#    de fordi den er for glatt. ENV_MEAN/ENV_STD/SSC regnes paa envelopen.
#
#    NB: hvis modellen skal kjores live via EMG_to_xArm.py, maa dennes
#    extract_features() endres til AKKURAT dette settet -- ellers ser
#    live-modellen andre tall enn den ble trent paa. (Gjores separat.)
# ----------------------------------------------------------------------
def _mnf_mdf(xc):
    """Mean- og median-frekvens fra effekt-spekteret til det sentrerte
    raasignalet. Begge er ratioer i spekteret, altsaa amplitude-uavhengige."""
    n = len(xc)
    freqs = np.fft.rfftfreq(n, d=1.0 / FS)
    P = np.abs(np.fft.rfft(xc)) ** 2
    P[0] = 0.0                              # dropp DC-komponenten
    tot = P.sum()
    if tot <= 0:
        return 0.0, 0.0
    mnf = float(np.sum(freqs * P) / tot)
    mdf = float(freqs[np.searchsorted(np.cumsum(P), tot / 2)])
    return mnf, mdf

def features(vindu_raw, vindu_env):
    raw = vindu_raw.astype(float)
    xc  = raw - BASELINE          # raasignal sentrert paa 0
    dx  = np.diff(xc)

    env = vindu_env.astype(float)
    ec  = env - BASELINE          # envelope sentrert paa ADC-baseline
    de  = np.diff(ec)

    env_mean = np.mean(env)
    env_std  = np.std(env)
    wamp     = np.sum(np.abs(dx) > WAMP_THRESHOLD)
    zc       = np.sum((xc[:-1] * xc[1:] < 0) & (np.abs(dx) > 1))
    ssc      = np.sum(np.diff(np.sign(de)) != 0)
    mnf, mdf = _mnf_mdf(xc)

    return [env_mean, env_std, wamp, zc, ssc, mnf, mdf]

FEATURE_NAVN = ["ENV_MEAN", "ENV_STD", "WAMP", "ZC", "SSC", "MNF", "MDF"]

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
os.makedirs(OUTPUT_DIR, exist_ok=True)
df = pd.DataFrame(rader, columns=["opptak_id", "label"] + FEATURE_NAVN)
df.to_excel(UT_FIL, index=False)

print("\nFerdig! Skrev", len(df), "vinduer til", UT_FIL)
print("\nFordeling per klasse:")
print(df["label"].value_counts())