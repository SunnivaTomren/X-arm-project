"""
Feature selection, is used to justify why we chose the 7 features we did. 
============================================================
This script explains why we chose the 7 features we did, and why we didn't just use the old 13.
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import f1_score, accuracy_score

# ----------------------------------------------------------------------
# 0. OPPSETT  (samme som process_emg.py)
# ----------------------------------------------------------------------
BASE       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE, "..", "data", "raw")
OUTPUT_DIR = os.path.join(BASE, "..", "data", "processed", "Stats", "Feature_selection")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAPPER = {
    os.path.join(DATA_DIR, "open_fist_celine"):    "opening",
    os.path.join(DATA_DIR, "closing_fist_celine"): "closing",
}
FS, WIN, STEP        = 250, 50, 25
BASELINE, WAMP_THR   = 512, 10
REST_ENV_LOW, GEST_HI = 528, 546

# Alle kandidat-features (de gamle 13 + de nye MNF/MDF)
KANDIDATER = ["MAV", "RMS", "WL", "VAR", "IEMG", "ZC", "SSC", "WAMP", "PEAK",
              "ENV_MEAN", "ENV_STD", "ENV_MAX", "ENV_RANGE", "MNF", "MDF"]
GAMLE_13   = ["MAV", "RMS", "WL", "VAR", "IEMG", "ZC", "SSC", "WAMP", "PEAK",
              "ENV_MEAN", "ENV_STD", "ENV_MAX", "ENV_RANGE"]
VALGTE_7   = ["ENV_MEAN", "ENV_STD", "WAMP", "ZC", "SSC", "MNF", "MDF"]

# ----------------------------------------------------------------------
# 1. LES RAADATA + REGN ALLE KANDIDAT-FEATURES
# ----------------------------------------------------------------------
def les_opptak(sti):
    linjer = open(sti).readlines()
    data_start = None
    for i, l in enumerate(linjer):
        if l.strip().split(",")[0] == "Sample_Index":
            data_start = i + 1
            break
    raw, env = [], []
    for l in linjer[data_start:]:
        d = l.strip().split(",")
        if len(d) == 3:
            raw.append(float(d[1])); env.append(float(d[2]))
    return np.array(raw), np.array(env)

def finn_aktiv(env):
    aktiv = np.zeros(len(env), dtype=bool); state = False
    for i, v in enumerate(env):
        if v >= GEST_HI:        state = True
        elif v < REST_ENV_LOW:  state = False
        aktiv[i] = state
    return aktiv

def mnf_mdf(xc):
    n = len(xc)
    freqs = np.fft.rfftfreq(n, d=1.0 / FS)
    P = np.abs(np.fft.rfft(xc)) ** 2; P[0] = 0.0
    tot = P.sum()
    if tot <= 0:
        return 0.0, 0.0
    return float((freqs * P).sum() / tot), float(freqs[np.searchsorted(np.cumsum(P), tot / 2)])

def alle_features(raw, env):
    xc = raw.astype(float) - BASELINE; dx = np.diff(xc)
    ec = env.astype(float) - BASELINE; de = np.diff(ec)
    mav = np.mean(np.abs(ec)); rms = np.sqrt(np.mean(ec ** 2)); wl = np.sum(np.abs(de))
    var = np.var(ec); iemg = np.sum(np.abs(ec))
    zc = np.sum((xc[:-1] * xc[1:] < 0) & (np.abs(dx) > 1))
    ssc = np.sum(np.diff(np.sign(de)) != 0); wamp = np.sum(np.abs(dx) > WAMP_THR)
    peak = np.max(np.abs(ec))
    em = np.mean(env); es = np.std(env); emax = np.max(env); erng = np.max(env) - np.min(env)
    mnf, mdf = mnf_mdf(xc)
    return [mav, rms, wl, var, iemg, zc, ssc, wamp, peak, em, es, emax, erng, mnf, mdf]

print("Leser raadata og regner ut alle 15 kandidat-features ...")
rader = []
for mappe, lab in MAPPER.items():
    for sti in glob.glob(os.path.join(mappe, "*.csv")):
        raw, env = les_opptak(sti); aktiv = finn_aktiv(env)
        oid = os.path.basename(sti)
        for s in range(0, len(raw) - WIN, STEP):
            e = s + WIN
            klasse = lab if aktiv[s:e].mean() > 0.5 else "rest"
            rader.append([oid, klasse] + alle_features(raw[s:e], env[s:e]))

df = pd.DataFrame(rader, columns=["opptak_id", "label"] + KANDIDATER)
print(f"  {len(df)} vinduer  |  fordeling: {dict(df['label'].value_counts())}\n")

# felles hjelpefunksjon: gruppert CV med RandomForest -> macro-F1
def cv_macro_f1(data, cols):
    y = data["label"].values; g = data["opptak_id"].values
    n = min(5, data["opptak_id"].nunique())
    clf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                 random_state=42, n_jobs=-1)
    pred = cross_val_predict(clf, data[cols].values, y, groups=g,
                             cv=GroupKFold(n), n_jobs=-1)
    return f1_score(y, pred, average="macro"), accuracy_score(y, pred)

# ----------------------------------------------------------------------
# FIGUR 1: redundans i det gamle 13-settet
#   Mange par med |r|>0.9 = de maalte det samme. Derfor kunne 9 slaas sammen.
# ----------------------------------------------------------------------
plt.figure(figsize=(9, 8))
korr = df[GAMLE_13].corr()
sns.heatmap(korr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            square=True, cbar_kws={"shrink": 0.8})
plt.title("Redundans i de gamle 13 featurene\n(|r| naer 1 = sier det samme -> duplikat)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "1_redundans_gamle13.png"), dpi=120)
plt.close()

# hvor mange par er sterkt korrelerte?
c = df[GAMLE_13].corr().abs()
par_over_09 = sum(c.iloc[i, j] > 0.9 for i in range(len(GAMLE_13))
                  for j in range(i + 1, len(GAMLE_13)))
print(f"FIGUR 1: {par_over_09} feature-par i det gamle settet har |r|>0.9 (redundante).")

# ----------------------------------------------------------------------
# FIGUR 2: settsammenligning (3-klasse, inkl. rest) -- macro-F1
#   Poeng: 7 valgte >= 13 gamle, og 15 (alle) gir ~ingenting ekstra.
# ----------------------------------------------------------------------
SETT = {
    "Amplitude alene\n(ENV_MEAN)": ["ENV_MEAN"],
    "13 gamle":                    GAMLE_13,
    "7 valgte":                    VALGTE_7,
    "Alle 15":                     KANDIDATER,
}
navn, f1er, accer = [], [], []
for n, cols in SETT.items():
    f1, acc = cv_macro_f1(df, cols)
    navn.append(n); f1er.append(f1); accer.append(acc)
    print(f"FIGUR 2: {n.splitlines()[0]:14s} ({len(cols):2d} feat)  macro-F1={f1:.3f}  acc={acc:.3f}")

plt.figure(figsize=(9, 5.5))
farger = ["#9e9e9e", "#7e57c2", "#2e7d32", "#90a4ae"]
bars = plt.bar(navn, f1er, color=farger)
for b, v in zip(bars, f1er):
    plt.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}",
             ha="center", fontsize=11, fontweight="bold")
plt.ylabel("macro-F1 (gruppert CV)")
plt.ylim(0, 1.0)
plt.title("Hvorfor 7 features?\n7 valgte slaar 13 gamle, og alle 15 gir ~ingenting ekstra")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "2_settsammenligning.png"), dpi=120)
plt.close()

# ----------------------------------------------------------------------
# FIGUR 3: det VANSKELIGE skillet -- kun opening vs closing
#   Poeng: amplitude alene ~62%, frekvens alene ~64%, men SAMMEN ~90%.
#   Viser at det IKKE bare er "closing er kraftigere".
# ----------------------------------------------------------------------
oc = df[df["label"].isin(["opening", "closing"])].copy()
SETT_OC = {
    "Amplitude alene\n(ENV_MEAN)":  ["ENV_MEAN"],
    "Frekvens alene\n(MNF+MDF)":    ["MNF", "MDF"],
    "Alle 7 sammen":                VALGTE_7,
}
navn_oc, acc_oc = [], []
for n, cols in SETT_OC.items():
    _, acc = cv_macro_f1(oc, cols)
    navn_oc.append(n); acc_oc.append(acc)
    print(f"FIGUR 3: {n.splitlines()[0]:16s}  opening-vs-closing acc={acc:.3f}")

plt.figure(figsize=(8, 5.5))
bars = plt.bar(navn_oc, acc_oc, color=["#1f77b4", "#d62728", "#2e7d32"])
for b, v in zip(bars, acc_oc):
    plt.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v*100:.0f}%",
             ha="center", fontsize=12, fontweight="bold")
plt.axhline(0.5, ls="--", color="gray", lw=1)
plt.text(2.4, 0.52, "myntkast (50%)", color="gray", fontsize=9)
plt.ylabel("noyaktighet opening vs closing (gruppert CV)")
plt.ylim(0, 1.0)
plt.title("Hva skiller opening fra closing?\nHverken amplitude eller frekvens alene holder -- kombinasjonen gjor det")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "3_open_vs_close.png"), dpi=120)
plt.close()

# ----------------------------------------------------------------------
# LAGRE TALLENE
# ----------------------------------------------------------------------
with pd.ExcelWriter(os.path.join(OUTPUT_DIR, "feature_selection.xlsx")) as w:
    pd.DataFrame({"sett": [n.replace("\n", " ") for n in navn],
                  "antall_features": [len(c) for c in SETT.values()],
                  "macro_F1": f1er, "accuracy": accer}).to_excel(w, "settsammenligning", index=False)
    pd.DataFrame({"sett": [n.replace("\n", " ") for n in navn_oc],
                  "open_vs_close_acc": acc_oc}).to_excel(w, "open_vs_close", index=False)

print("\nFerdig! Lagret 3 figurer + feature_selection.xlsx i:")
print(" ", OUTPUT_DIR)
