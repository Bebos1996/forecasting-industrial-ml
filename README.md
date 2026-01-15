# 📈 Industrial Sales Forecasting with LightGBM

Questo progetto implementa una **pipeline completa di forecasting su serie temporali** in un contesto industriale, utilizzando Python e LightGBM.

L’obiettivo è predire le **vendite giornaliere** di prodotti in diversi store, confrontando baseline semplici con un modello di machine learning ottimizzato, seguendo buone pratiche per dati temporali.

---

## 🧠 Obiettivi del progetto

- Costruire un workflow ML end-to-end per serie temporali
- Implementare **baseline di riferimento** per un confronto corretto
- Applicare **feature engineering specifico per time series**
- Allenare e ottimizzare un modello **LightGBM**
- Valutare le performance con **split temporale realistico**

---

## 📂 Struttura del progetto

forecasting-industrial-ml/
│
├── src/
│ ├── ingestion/
│ │ └── load_data.py
│ ├── features/
│ │ └── build_features.py
│ └── models/
│   ├── baseline.py
│   └── lightgbm_model.py
│
├── data/
│ ├── raw/ # dati originali (non versionati)
│ ├── processed/ # dati puliti (non versionati)
│ ├── features/ # dataset con feature engineering
│ └── models/ # output dei modelli
│
├── .gitignore
├── requirements.txt
└── README.md


> ⚠️ I file CSV non sono inclusi nella repository per evitare data leakage e problemi di licenza.

---

## 🔧 Tecnologie utilizzate

- Python 3
- Pandas, NumPy
- Scikit-learn
- LightGBM
- Matplotlib
- WSL + virtual environment

---

## 🏗️ Pipeline ML

### 1️⃣ Data ingestion
Caricamento e preparazione dei dati grezzi.
python src/ingestion/load_data.py

2️⃣ Feature engineering
Creazione di feature specifiche per serie temporali:

-Lag features: lag_1, lag_7, lag_14, lag_28
-Rolling statistics: rolling_mean_7, rolling_mean_14
-Time features (giorno della settimana, mese, trimestre)
-Encoding ciclico (sin/cos per weekday e month)
-Indicatori evento

python src/features/build_features.py

3️⃣ Baseline models
Per un confronto corretto vengono implementate:

Naive forecast → vendite del giorno precedente
Rolling mean (7 giorni)

python src/models/baseline.py

Risultati baseline (test set):
Naive RMSE ≈ 2.74
Rolling-7 RMSE ≈ 2.10

4️⃣ Modello LightGBM
Split temporale (80% train / 20% test)
Nessuno shuffle dei dati
Addestramento su feature ingegnerizzate

5️⃣ Hyperparameter tuning
Tuning manuale con TimeSeriesSplit cross-validation
Parametri ottimizzati:

max_depth
num_leaves
learning_rate

Il tuning migliora la stabilità del modello e riduce l’overfitting.

6️⃣ Valutazione finale
Risultati finali (test set):

Modello	RMSE
Naive	~2.74
Rolling Mean (7)	~2.10
LightGBM	~2.00

Il modello LightGBM supera le baseline e si avvicina al limite predittivo del dataset.

🚀 Come eseguire il progetto
# crea virtual environment
python3 -m venv .venv
source .venv/bin/activate

# installa dipendenze
pip install -r requirements.txt

# esegui pipeline
python src/ingestion/load_data.py
python src/features/build_features.py
python src/models/baseline.py
python src/models/lightgbm_model.py

📌 Considerazioni finali
Il modello ML supera le baseline

Il principale limite è la qualità informativa del dataset, non il modello

Miglioramenti futuri richiederebbero:
feature esogene aggiuntive
maggiore storicità
informazioni di contesto (promozioni, stock, calendario avanzato)