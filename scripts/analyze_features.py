"""
Feature-analyse og klassesammenligning
======================================
Leser de ferdig-uttrekte featurene fra output/features.xlsx og sammenligner
hvordan hver av de 13 featurene oppforer seg for de tre klassene
(opening / closing / rest).

Formaal: forstaa HVILKE features som skiller bevegelsene best fra hverandre.
En feature der klassene ligger i tre adskilte "hauger" er lett for modellen aa
laere av; en feature der klassene overlapper helt gir lite signal.

Kjor:  python scripts/analyze_features.py

Lager i output/:
  - feature_boxplots.png      boksplott per feature, en boks per klasse
  - feature_histograms.png    fordelingskurver (KDE) per feature
  - feature_ranking.png       F-score: hvor godt hver feature skiller klassene
  - feature_correlation.png   korrelasjon mellom features (redundans)
  - feature_stats.xlsx        snitt/std/median per klasse per feature
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

BASE       = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE, "..", "output")
INN_FIL    = os.path.join(OUTPUT_DIR, "features.xlsx")

FEATURE_NAVN = ["MAV", "RMS", "WL", "VAR", "IEMG", "ZC", "SSC", "WAMP", "PEAK",
                "ENV_MEAN", "ENV_STD", "ENV_MAX", "ENV_RANGE"]

# Rekkefolge + farger paa klassene i alle plott
KLASSER = ["rest", "opening", "closing"]
FARGER  = {"rest": "#9e9e9e", "opening": "#1f77b4", "closing": "#d62728"}

# ----------------------------------------------------------------------
# 1. LES DATA
# ----------------------------------------------------------------------
df = pd.read_excel(INN_FIL)
# behold bare klassene vi bryr oss om, i onsket rekkefolge
df = df[df["label"].isin(KLASSER)].copy()
df["label"] = pd.Categorical(df["label"], categories=KLASSER, ordered=True)

class_counts = df["label"].value_counts()[KLASSER]
print("=" * 70)
print("KLASSEBALANSE (antall vinduer per label)")
print("=" * 70)
totalt = int(class_counts.sum())
for k in KLASSER:
    n = int(class_counts[k])
    print(f"  {k:<10} {n:6d}  ({100 * n / totalt:5.1f} %)")
print(f"  {'TOTALT':<10} {totalt:6d}")
print("  NB: rest dominerer -> husk class_weight / balansering i train.py\n")

# ----------------------------------------------------------------------
# 2. STATISTIKK PER KLASSE PER FEATURE  (snitt +/- std, median)
# ----------------------------------------------------------------------
stats = df.groupby("label", observed=True)[FEATURE_NAVN].agg(["mean", "std", "median"])
stats = stats.round(2)
stats.to_excel(os.path.join(OUTPUT_DIR, "feature_stats.xlsx"))

print("=" * 70)
print("SNITT +/- STD PER KLASSE  (hoy forskjell mellom klasser = lett aa l1re)")
print("=" * 70)
for f in FEATURE_NAVN:
    linje = f"{f:<10}"
    for k in KLASSER:
        m = df.loc[df["label"] == k, f].mean()
        s = df.loc[df["label"] == k, f].std()
        linje += f"  {k}: {m:8.1f}+/-{s:6.1f}"
    print(linje)
print()

# ----------------------------------------------------------------------
# 3. F-SCORE:  hvor godt skiller HVER feature klassene? (ANOVA)
#    Hoy F = stor forskjell mellom klasser vs. spredning inni hver klasse.
#    Dette er i praksis "hvor mye modellen kan laere av denne featuren".
# ----------------------------------------------------------------------
X = df[FEATURE_NAVN].values
y = df["label"].values
f_scores, p_values = f_classif(X, y)

rang = (pd.DataFrame({"feature": FEATURE_NAVN, "F_score": f_scores})
        .sort_values("F_score", ascending=False)
        .reset_index(drop=True))

print("=" * 70)
print("FEATURES RANGERT ETTER SKILLEEVNE (F-score, hoyest = best)")
print("=" * 70)
print(rang.to_string(index=False))
print("  NB: denne rangeringen domineres av det LETTE rest-skillet.\n")

# ----------------------------------------------------------------------
# 3b. PARVIS: kun opening vs closing (rest ekskludert)
#     Dette er det VANSKELIGE skillet. Her ser vi hvilke features som
#     faktisk skiller de to bevegelsene fra hverandre, uten at rest
#     "drukner" rangeringen.
# ----------------------------------------------------------------------
par = df[df["label"].isin(["opening", "closing"])]
f_par, p_par = f_classif(par[FEATURE_NAVN].values, par["label"].values)

rang_par = (pd.DataFrame({"feature": FEATURE_NAVN, "F_score": f_par,
                          "p_verdi": p_par})
            .sort_values("F_score", ascending=False)
            .reset_index(drop=True))
rang_par["F_score"] = rang_par["F_score"].round(1)

print("=" * 70)
print("KUN OPENING vs CLOSING  (det vanskelige skillet, rest ekskludert)")
print("=" * 70)
print(rang_par.to_string(index=False))
print("  Hoy F + lav p_verdi = feature som faktisk skiller de to bevegelsene.\n")

# ----------------------------------------------------------------------
# 4. PLOTT: boksplott per feature (en boks per klasse)
# ----------------------------------------------------------------------
def rutenett(tittel):
    fig, akser = plt.subplots(3, 5, figsize=(20, 11))
    fig.suptitle(tittel, fontsize=16)
    akser = akser.flatten()
    return fig, akser

fig, akser = rutenett("Boksplott per feature - opening vs closing vs rest")
for i, f in enumerate(FEATURE_NAVN):
    sns.boxplot(data=df, x="label", y=f, order=KLASSER, ax=akser[i],
                hue="label", palette=FARGER, legend=False, showfliers=False)
    akser[i].set_title(f)
    akser[i].set_xlabel("")
    akser[i].set_ylabel("")
for j in range(len(FEATURE_NAVN), len(akser)):
    akser[j].axis("off")
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(os.path.join(OUTPUT_DIR, "feature_boxplots.png"), dpi=120)
plt.close()

# ----------------------------------------------------------------------
# 5. PLOTT: fordelingskurver (KDE) per feature
# ----------------------------------------------------------------------
fig, akser = rutenett("Fordeling per feature - overlapp = vanskelig for modellen")
for i, f in enumerate(FEATURE_NAVN):
    for k in KLASSER:
        data = df.loc[df["label"] == k, f]
        sns.kdeplot(data, ax=akser[i], color=FARGER[k], fill=True,
                    alpha=0.35, linewidth=1.5, label=k, warn_singular=False)
    akser[i].set_title(f)
    akser[i].set_xlabel("")
    akser[i].set_ylabel("")
akser[0].legend(fontsize=9)
for j in range(len(FEATURE_NAVN), len(akser)):
    akser[j].axis("off")
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(os.path.join(OUTPUT_DIR, "feature_histograms.png"), dpi=120)
plt.close()

# ----------------------------------------------------------------------
# 6. PLOTT: F-score rangering (soylediagram)
# ----------------------------------------------------------------------
plt.figure(figsize=(10, 6))
sns.barplot(data=rang, y="feature", x="F_score", hue="feature",
            palette="viridis", legend=False)
plt.title("Hvor godt skiller hver feature klassene? (F-score, hoyere = bedre)")
plt.xlabel("F-score")
plt.ylabel("")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_ranking.png"), dpi=120)
plt.close()

# ----------------------------------------------------------------------
# 7. PLOTT: korrelasjon mellom features (redundans)
# ----------------------------------------------------------------------
plt.figure(figsize=(9, 8))
korr = df[FEATURE_NAVN].corr()
sns.heatmap(korr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            square=True, cbar_kws={"shrink": 0.8})
plt.title("Korrelasjon mellom features (n 1 +/-1 = sier det samme)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_correlation.png"), dpi=120)
plt.close()

# ----------------------------------------------------------------------
# 8. PLOTT: scatter (ZC vs ENV_STD) + PCA av alle 13 features, side ved side
#    Her ser vi HVERT vindu som et punkt, ikke bare oppsummert i en boks.
#    Overlapper skyene? Da er klassene vanskelige aa skille.
# ----------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# -- 8a. Scatter: ZC vs ENV_STD (to av de mest komplementaere featurene) --
for k in KLASSER:
    sub = df[df["label"] == k]
    ax1.scatter(sub["ZC"], sub["ENV_STD"], s=8, alpha=0.4,
                color=FARGER[k], label=k)
ax1.set_xlabel("ZC")
ax1.set_ylabel("ENV_STD")
ax1.set_title("Scatter: ZC vs ENV_STD")
ax1.legend()

# -- 8b. PCA: standardiser alle 13 features, projiser til 2D --
X_std = StandardScaler().fit_transform(df[FEATURE_NAVN].values)
pca = PCA(n_components=2)
komponenter = pca.fit_transform(X_std)
var = pca.explained_variance_ratio_ * 100   # forklart varians i prosent

labels_arr = df["label"].values
for k in KLASSER:
    mask = labels_arr == k
    ax2.scatter(komponenter[mask, 0], komponenter[mask, 1], s=8, alpha=0.4,
                color=FARGER[k], label=k)
ax2.set_xlabel(f"PC1 ({var[0]:.1f} % varians)")
ax2.set_ylabel(f"PC2 ({var[1]:.1f} % varians)")
ax2.set_title("PCA: alle features projisert til 2D")
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "scatter_og_pca.png"), dpi=120)
plt.close()

# ----------------------------------------------------------------------
# 9. FOKUS: KUN opening vs closing (rest ekskludert)
#    Her plotter vi bare de to bevegelsene, uten at rest presser skalaen,
#    slik at vi ser hvor godt hver feature faktisk adskiller dem.
# ----------------------------------------------------------------------
KLASSER_PAR = ["opening", "closing"]

# -- 9a. Boksplott per feature (kun opening vs closing) --
fig, akser = rutenett("KUN opening vs closing - boksplott per feature")
for i, f in enumerate(FEATURE_NAVN):
    sns.boxplot(data=par, x="label", y=f, order=KLASSER_PAR, ax=akser[i],
                hue="label", palette=FARGER, legend=False, showfliers=False)
    akser[i].set_title(f)
    akser[i].set_xlabel("")
    akser[i].set_ylabel("")
for j in range(len(FEATURE_NAVN), len(akser)):
    akser[j].axis("off")
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(os.path.join(OUTPUT_DIR, "opening_vs_closing_boxplots.png"), dpi=120)
plt.close()

# -- 9b. Fordelingskurver (KDE) per feature (kun opening vs closing) --
fig, akser = rutenett("KUN opening vs closing - fordeling per feature")
for i, f in enumerate(FEATURE_NAVN):
    for k in KLASSER_PAR:
        data = par.loc[par["label"] == k, f]
        sns.kdeplot(data, ax=akser[i], color=FARGER[k], fill=True,
                    alpha=0.35, linewidth=1.5, label=k, warn_singular=False)
    akser[i].set_title(f)
    akser[i].set_xlabel("")
    akser[i].set_ylabel("")
akser[0].legend(fontsize=9)
for j in range(len(FEATURE_NAVN), len(akser)):
    akser[j].axis("off")
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(os.path.join(OUTPUT_DIR, "opening_vs_closing_histograms.png"), dpi=120)
plt.close()

# -- 9c. Fokusfigur: parvis F-score + scatter av de to beste featurene --
fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Fokus: hva adskiller opening fra closing?", fontsize=14)

# venstre: parvis F-score rangering (hvor godt hver feature skiller de to)
sns.barplot(data=rang_par, y="feature", x="F_score", hue="feature",
            palette="magma", legend=False, ax=axA)
axA.set_title("F-score (kun opening vs closing, hoyere = bedre)")
axA.set_xlabel("F-score")
axA.set_ylabel("")

# hoyre: scatter av de to hoyest rangerte featurene mot hverandre
beste = rang_par["feature"].tolist()[:2]      # f.eks. SSC og ZC
fx, fy = beste[0], beste[1]
for k in KLASSER_PAR:
    sub = par[par["label"] == k]
    axB.scatter(sub[fx], sub[fy], s=10, alpha=0.4, color=FARGER[k], label=k)
axB.set_xlabel(fx)
axB.set_ylabel(fy)
axB.set_title(f"Scatter: {fx} vs {fy} (de to beste)")
axB.legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(OUTPUT_DIR, "opening_vs_closing_fokus.png"), dpi=120)
plt.close()

print("Ferdig! Lagret i output/:")
print("  feature_boxplots.png, feature_histograms.png,")
print("  feature_ranking.png, feature_correlation.png,")
print("  scatter_og_pca.png, feature_stats.xlsx")
print("  opening_vs_closing_boxplots.png, opening_vs_closing_histograms.png,")
print("  opening_vs_closing_fokus.png")
