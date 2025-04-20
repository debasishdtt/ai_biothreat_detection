import pandas as pd
from pathlib import Path


def process_epi_data():
    input_path = Path("data/raw/COVID19/epidemiological/time_series_covid19_confirmed_US.csv")
    output_dir = Path("data/processed")
    output_dir.mkdir(exist_ok=True)

    # Load and process data
    df = pd.read_csv(input_path)
    date_cols = [col for col in df.columns
                 if pd.to_datetime(col, format='%m/%d/%y', errors='coerce') is not pd.NaT]

    # Melt and clean data
    df = df[['Province_State'] + date_cols].melt(id_vars=['Province_State'], var_name='date', value_name='cases')
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%y', errors='coerce')
    df = df.dropna(subset=['date'])

    # Generate ISO weeks (Monday-based)
    df['week'] = df['date'].dt.strftime('%G-W%V')  # Critical fix: %G=ISO year, %V=ISO week

    # Aggregate
    df = df.groupby(['Province_State', 'week'], observed=False)['cases'].sum().reset_index()

    df.to_csv(output_dir / "epi_processed.csv", index=False)
    print(f"Processed {len(df)} epidemiological records")


if __name__ == "__main__":
    process_epi_data()