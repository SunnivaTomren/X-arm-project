# X-arm-project — EMG-styrt robotarm

Styr en **UFactory xArm** med muskelsignaler (EMG). Et Olimex-kort leser EMG fra
underarmen, en maskinlæringsmodell klassifiserer bevegelsen (`opening` / `closing`
/ `rest`), og klassen oversettes til en handling på armen (åpne/lukke griper).

> EMG-signaler er individuelle. En modell trent på én person virker ikke
> nødvendigvis like godt på en annen uten ny kalibrering/opptak.

---

## 1. Rask oppstart

```bash
# Fra prosjektroten (X-arm-project/). Bruk det medfølgende virtuelle miljøet:
.venv/Scripts/python.exe Scripts/process_emg.py            # rådata  -> features.xlsx
.venv/Scripts/python.exe Scripts/process_features.py       # analyseplott av de 7 featurene
.venv/Scripts/python.exe Scripts/feature_selection_report.py  # "hvorfor 7 features"-figurer
.venv/Scripts/python.exe Scripts/Train_deep.py             # tren modell -> Models/emg_model_deep.pt
```

**Alle Python-kommandoer skal kjøres via `.venv`** (`.venv/Scripts/python.exe`),
ikke system-Python. Nøkkelpakker: `torch` (CPU), `scikit-learn`, `pandas`,
`numpy`, `matplotlib`, `seaborn`, `openpyxl`, `joblib`. Live-styring krever i
tillegg `xarm-python-sdk` og `pyserial`.

---

## 2. Mappestruktur

```
X-arm-project/
├── data/
│   ├── raw/                     rå EMG-opptak (CSV), én undermappe per gest
│   │   ├── open_fist_celine/    150 opptak  <- BRUKES (label: opening)
│   │   ├── closing_fist_celine/ 150 opptak  <- BRUKES (label: closing)
│   │   ├── open_fist/ closing_fist/ hiya_up/ hiya_down/ lifting_cup/
│   │   │                        finnes, men brukes IKKE av gjeldende pipeline
│   └── processed/
│       ├── features.xlsx        utdata fra process_emg.py (én rad per vindu)
│       └── Stats/               plott og statistikk
│           ├── Open_close_stats/     fra process_features.py
│           └── Feature_selection/    fra feature_selection_report.py
├── Models/
│   ├── emg_model_deep.pt        <- HOVEDMODELL (fra Train_deep.py)
│   ├── emg_model.pt             gammelt eksperiment (Train.py)
│   └── emg_model_test.pt        gammelt eksperiment (Train_test.py)
└── Scripts/                     all kode (se seksjon 4)
```

---

## 3. Rådataformat (CSV)

Hver fil i `data/raw/<gest>/` er ett opptak. En header-blokk etterfulgt av
samples:

```
Action_Label,closing_fist_C
Fs_Hz,250
Total_Samples,1250

Sample_Index,Raw_EMG,Envelope
0,515.0,518.2
1,517.0,518.17
...
```

- **Raw_EMG** — rått 10-bit ADC-signal, hviler rundt `BASELINE = 512`.
- **Envelope** — glidende gjennomsnitt av det rektifiserte råsignalet (glattere).
- Samplingsrate `Fs_Hz = 250`.

---

## 4. Pipeline og scripts

Rekkefølgen dataene flyter i:

**rå CSV → `process_emg.py` → `features.xlsx` → `Train_deep.py` → `emg_model_deep.pt` → live-script → xArm**

| Script | Rolle |
|---|---|
| **process_emg.py** | Leser rådata, finner aktive vinduer (envelope-hysterese), regner **7 features** per vindu (50 samples, 50 % overlapp), skriver `features.xlsx`. |
| **process_features.py** | Analyseplott av de 7 featurene (F-score, boksplott, korrelasjon, PCA, opening-vs-closing). Ren rapportering, endrer ikke data. |
| **feature_selection_report.py** | «Forsvars»-figurer: hvorfor 7 features (regner alle 15 kandidater fra rådata og sammenligner med gruppert CV). |
| **Train_deep.py** | **Hovedtreningsscript.** Trener et dypt nett → `emg_model_deep.pt`. Håndterer klasseubalanse (WeightedRandomSampler + class weights) og datalekkasje (GroupShuffleSplit). |
| Train.py / Train_test.py | Eldre, enklere eksperimenter. Ikke i aktiv bruk. |
| **Testing_robot_arm_V2.py** | Live-inferens med `emg_model_deep.pt` → xArm (PyTorch-modell). |
| EMG_to_xArm.py | Alternativ, selvstendig pipeline (RandomForest → `.joblib`). Egen feature-funksjon. Deler ikke modellfil med de andre. |

---

## 5. Feature-settet (7 features)

```
ENV_MEAN, ENV_STD, WAMP, ZC, SSC, MNF, MDF
```

Ble redusert fra 13 til 7 (begrunnet empirisk, se `Feature_selection/`):

- **De gamle 13 var mest duplikater** — 29 feature-par hadde korrelasjon
  |r|>0.9 (f.eks. `MAV = IEMG = ENV_MEAN`, `PEAK = ENV_MAX`). Fjernet.
- **MNF/MDF (mean/median frequency) ble lagt til.** De regnes fra
  effekt-spekteret til råsignalet og er **amplitude-uavhengige**. Uten dem
  hviler opening-vs-closing nesten bare på «hvem som spenner hardest».

Målt med gruppert kryssvalidering (ærlige tall):

| Sett | macro-F1 (3 klasser) | opening vs closing |
|---|---|---|
| Amplitude alene | 0.69 | 62 % |
| Frekvens alene (MNF+MDF) | – | 64 % |
| 13 gamle | 0.87 | – |
| **7 valgte** | **0.89** | **90 %** |
| Alle 15 | 0.90 | – |

Poenget: 7 features slår 13, og flere features gir nesten ingenting — vi er ved
**datagrensa**, ikke feature-grensa.

---

## 6. ⚠️ Kritiske invarianter — LES FØR DU ENDRER

Feature-definisjonene er duplisert på tvers av flere filer. **Endrer du feature-
settet ett sted, må du oppdatere ALLE stedene under likt**, ellers ser modellen
andre tall enn den ble trent på (og alt bryter stille):

| Hvis du endrer... | ...må du også oppdatere |
|---|---|
| `features()` / `FEATURE_NAVN` i **process_emg.py** | `FEATURES` i **Train_deep.py**, `FEATURE_NAVN` i **process_features.py**, og live-scriptenes `extract_features` + `FEATURES` (**Testing_robot_arm_V2.py**, **EMG_to_xArm.py**) |
| `WIN` / `STEP` / `BASELINE` / `FS` / `WAMP_THRESHOLD` | samme konstanter i alle scriptene over |

Andre invarianter:

- **Datalekkasje:** vinduer overlapper 50 %, så nabovinduer er nesten like.
  Split ALLTID på `opptak_id` (GroupShuffleSplit / GroupKFold), aldri vanlig
  `train_test_split`. Ellers blir testtallene kunstig høye.
- **Klasseubalanse:** ~86 % av vinduene er `rest`. En modell kan få ~86 %
  «accuracy» ved å alltid gjette rest. Bruk class weights + sampler, og se på
  **per-klasse recall** for opening/closing, ikke bare total accuracy.

---

## 7. Kjente problemer / TODO

- **Live-scriptene er ute av synk med den nye modellen.** `Testing_robot_arm_V2.py`
  (og `EMG_to_xArm.py`) bruker fortsatt det gamle 12/13-feature-settet. Etter at
  `Train_deep.py` nå lager en modell med **7 inputs**, må disse oppdateres til
  akkurat `ENV_MEAN, ENV_STD, WAMP, ZC, SSC, MNF, MDF` (samme rekkefølge) og
  `WIN=50` før live-styring virker. Se seksjon 6.
- **Hardkodede stier:** `Testing_robot_arm_V2.py` har fortsatt absolutte
  Celine-stier (`TRAIN_DATA`, `SAMPLE_FILE`). Gjør dem relative (som i
  `process_emg.py`) før kjøring på annen maskin.
- **Modellarkitekturen er trolig for stor:** 10 lag for 7 features slår ikke en
  enkel RandomForest. Vurder et mindre nett.
- **Kun én EMG-kanal og én person.** Største mulige løft er en **andre elektrode**
  (bøye- vs strekkemuskel) og opptak fra flere personer med per-person-
  normalisering — mer enn noen ny feature.

---

## 8. For en AI-agent som skal gjøre endringer

- Kjør alt via `.venv/Scripts/python.exe`.
- Bruk **midlertidige/utforskende script i en scratch-mappe** utenfor prosjektet
  for analyse; skriv bare til prosjektet når endringen er bestemt.
- Endringer i features → regenerer `features.xlsx` (`process_emg.py`) FØR du
  trener, ellers trener du på gamle kolonner.
- Rapporter alltid **per-klasse** resultater (gruppert CV), ikke bare accuracy.
- Respekter synk-tabellen i seksjon 6.
