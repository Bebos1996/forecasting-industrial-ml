import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error

def naive_forecast(df: pd.DataFrame) -> pd.DataFrame:
    """
    Previsione naive: y_pred = sales giorno precedente
    """

    df = df.sort_values(["item_id", "store_id", "date"])
    df["y_pred_naive"] = df.groupby(["item_id", "store_id"])["sales"].shift(1)
    return df

def rolling_forecast(df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """
    Previsione rolling mean: media degli ultimi 'window' giorni
    """
    df = df.sort_values(["item_id", "store_id", "date"])
    df[f"y_pred_roll_{window}"] = df.groupby(["item_id", "store_id"])["sales"].transform(
        lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
    )
    return df

def evaluate(df: pd.DataFrame, y_col="sales", y_pred_col="y_pred_naive"):
    """
    Prendo solo le righe in cui la predizione non è Nan
    Calcolo RMSE tra y_col e y_pred_col
    """
    mask = ~df[y_pred_col].isna()
    mse = mean_squared_error(df.loc[mask, y_col], df.loc[mask, y_pred_col])
    rmse = mse ** 0.5
    print(f"RMSE ({y_pred_col}): {rmse:.2f}")
    return rmse

if __name__ == "__main__":
    # Paths
    features_path = Path("data/features/features.csv")

    if not features_path.exists():
        raise FileNotFoundError(f"{features_path} non trovato. Esegui prima build_features.py")

    # Carica features
    df = pd.read_csv(features_path, parse_dates=["date"])

    # Baseline naive
    df = naive_forecast(df)
    evaluate(df, y_pred_col="y_pred_naive")

    # Baseline rolling
    df = rolling_forecast(df, window=7)
    evaluate(df, y_pred_col="y_pred_roll_7")

    # Salva previsioni
    output_path = Path("data/models/")
    output_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path / "baseline_predictions.csv", index=False)
    print(f"✅ Baseline predictions saved to {output_path / 'baseline_predictions.csv'}")