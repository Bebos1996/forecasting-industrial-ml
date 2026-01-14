import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import mean_squared_error

"""
Load data
"""
features_path = Path("data/features/features.csv")
df = pd.read_csv(features_path, parse_dates=["date"]) #la colonna date è trasformata in datetime


"""
Ordine temporale
Per serie temporali non si può fare shuffle casuale perché mi servono dati passati per predire il futuro
"""
df = df.sort_values("date")

"""
Seleziono le features
"""
Target = "sales"
Features = [
    "lag_1", "lag_7", "lag_14", "lag_28",
    "rolling_mean_7", "rolling_mean_14",
    "dow_sin", "dow_cos", "month_sin", "month_cos"
]

df_model = df.dropna(subset=Features + [Target])
X = df_model[Features]
y = df_model[Target]

"""
Split train / test (time-based)
"""
split_date = df_model["date"].quantile(0.8) #splitto su percentile temporale (80% train, 20% test)
X_train = X[df_model["date"] <= split_date]
X_test = X[df_model["date"] > split_date]

y_train = y[df_model["date"] <= split_date]
y_test = y[df_model["date"] > split_date]

model = lgb.LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

#Alleno il modello
model.fit(X_train, y_train)

"""
Evaluation: faccio predire al modello allenato la y del test set e poi confronto i risultati con i valori reali
"""
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
print(f"LightGBM RMSE: {rmse:.2f}")

"""
Salva le predizioni
"""
output = df_model.loc[X_test.index, ["date", Target]].copy()
output["y_pred_lgbm"] = y_pred
output_path = Path("data/models")
output_path.mkdir(parents=True, exist_ok=True)
output.to_csv(output_path / "lightgbm_predictions.csv", index=False)
print("Predictions saved")