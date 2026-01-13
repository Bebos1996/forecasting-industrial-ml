import os
import yaml
import pandas as pd
from pathlib import Path

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
    

def load_raw_data(config: dict):
    raw_path = Path(config["data"]["raw_path"])

    sales_path = raw_path / "sales_train_validation.csv"
    calendar_path = raw_path / "calendar.csv"
    prices_path = raw_path / "sell_prices.csv"

    sales_df = pd.read_csv(sales_path, nrows=1000)
    calendar_df = pd.read_csv(calendar_path)
    prices_df = pd.read_csv(prices_path)

    return sales_df, calendar_df, prices_df


def transform_sales_data(sales_df: pd.DataFrame, calendar_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
    sales_long = sales_df.melt(
        id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
        var_name="d",
        value_name="sales"
    )

    calendar_df["date"] = pd.to_datetime(calendar_df["date"])
    #Merge with calendar
    df = sales_long.merge(calendar_df, on="d", how="left")
    #Merge with prices
    df = df.merge(prices_df, on=["store_id", "item_id", "wm_yr_wk"], how="left")

    #Select & Clean columns
    df = df[
        [
            "date",
            "item_id",
            "store_id",
            "sales",
            "sell_price",
            "event_name_1",
            "weekday"
        ]
    ]

    df = df.rename(
        columns={
            "sell_price": "price",
            "event_name_1": "event_name"
        }
    )

    df["sales"] = df["sales"].astype(int)

    return df

def save_processed_data(df: pd.DataFrame, output_path: str):
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "clean_sales.csv"
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    config = load_config("config/config.yaml")
    sales_df, calendar_df, prices_df = load_raw_data(config)
    clean_df = transform_sales_data(sales_df, calendar_df, prices_df)
    save_processed_data(clean_df, config["data"]["processed_path"])

print("Data ingestion completed successfully")