import pandas as pd
from pathlib import Path

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggiungo colonne temporali dal campo 'date'
    """
    df["day_of_week"] = df["date"].dt.weekday
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["day"] = df["date"].dt.day
    df["year"] = df["date"].dt.year
    df["is_event"] = df["event_name"].notna().astype(int)
    return df

def add_lag_features(df: pd.DataFrame, lags=[1, 7]) -> pd.DataFrame:
    """
    Aggiunge feature lag, cioè vendite dei giorni passati.
    """
    df = df.sort_values(["item_id", "store_id", "date"])
    for lag in lags:
        df[f"lag_{lag}"] = df.groupby(["item_id", "store_id"])["sales"].shift(lag)
    return df

def add_rolling_features(df: pd.DataFrame, windows=[7, 14]) -> pd.DataFrame:
    """
    Aggiunge feature rolling mean per vendite passate.
    """
    df = df.sort_values(["item_id", "store_id", "date"])
    for window in windows:
        df[f"rolling_mean_{window}"] = df.groupby(["item_id", "store_id"])["sales"].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )
    return df

def save_features(df: pd.DataFrame, output_path: str):
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "features.csv", index=False)
    print(f"✅ Features saved to {output_dir / 'features.csv'}")

if __name__ == "__main__":
    # Paths
    raw_features_path = Path("data/processed/clean_sales.csv")
    features_output_path = Path("data/features/")

    # Load data
    df = pd.read_csv(raw_features_path, parse_dates=["date"])

    # Feature engineering
    df = add_time_features(df)
    df = add_lag_features(df, lags=[1,7])
    df = add_rolling_features(df, windows=[7,14])

    # Save
    save_features(df, features_output_path)