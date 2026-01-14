import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

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


"""
Tuning iperparametri del modello
"""
param_grid = {
    "max_depth": [5, 6, 7],
    "num_leaves": [20, 30, 40],
    "learning_rate": [0.03, 0.05, 0.07]
}
tscv = TimeSeriesSplit(n_splits=3)
best_rmse = float("inf")
best_params = None
rmses = []

print("Starting hyperparameter tuning...")
for max_depth, num_leaves, learning_rate in itertools.product(
    param_grid["max_depth"], param_grid["num_leaves"], param_grid["learning_rate"]
):
    rmse = []
    for train_idx, val_idx in tscv.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model = lgb.LGBMRegressor(
            n_estimators=300,
            max_depth=max_depth,
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            random_state=42
        )
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        rmses.append(mean_squared_error(y_val, y_pred) ** 0.5)
    
    mean_rmse = np.mean(rmses)
    print(f"Params: max_depth={max_depth}, num_leaves={num_leaves}, lr={learning_rate} -> RMSE CV: {mean_rmse:.3f}")
    if mean_rmse < best_rmse:
        best_rmse = mean_rmse
        best_params = {"max_depth": max_depth, "num_leaves": num_leaves, "learning_rate": learning_rate}

print("Best params:", best_params)
print("Best CV RMSE:", best_rmse)

final_model = lgb.LGBMRegressor(
    n_estimators=300,
    learning_rate=best_params["learning_rate"],
    max_depth=best_params["max_depth"],
    num_leaves=best_params["num_leaves"],
    random_state=42
)

#Alleno il modello
final_model.fit(X_train, y_train)

"""
Evaluation: faccio predire al modello allenato la y del test set e poi confronto i risultati con i valori reali
"""
y_pred = final_model.predict(X_test)
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


"""
Features importance plot
"""
importances = final_model.feature_importances_
plt.figure(figsize=(10,6))
plt.barh(Features, importances)
plt.xlabel("Importance")
plt.title("LightGBM Feature Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")  # salva il file
print("Feature importance plot saved as feature_importance.png")