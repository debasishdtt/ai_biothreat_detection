import pandas as pd
from pathlib import Path


def process_mobility():
    input_path = Path("data/raw/COVID19/mobility/2020_US_Region_Mobility_Report.csv")
    output_dir = Path("data/processed")
    output_dir.mkdir(exist_ok=True)

    df = pd.read_csv(input_path, usecols=['sub_region_1', 'date', 'retail_and_recreation_percent_change_from_baseline'])

    # Clean and standardize
    df = (
        df.rename(columns={'sub_region_1': 'Province_State'})
        .dropna(subset=['Province_State'])
        .assign(Province_State=lambda x: x['Province_State'].str.title().str.strip())
    )

    # Parse dates with ISO week
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df['week'] = df['date'].dt.strftime('%G-W%V')  # ISO week format

    # Aggregate
    df = (
        df.groupby(['Province_State', 'week'])
        .agg({'retail_and_recreation_percent_change_from_baseline': 'mean'})
        .reset_index()
    )

    df.to_csv(output_dir / "mobility_processed.csv", index=False)
    print(f"Processed {len(df)} mobility records")


if __name__ == "__main__":
    process_mobility()