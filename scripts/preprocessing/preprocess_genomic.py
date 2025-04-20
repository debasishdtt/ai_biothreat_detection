import pandas as pd
from pathlib import Path


def process_genomic():
    input_path = Path("data/raw/COVID19/genomic/sequences.csv")
    output_path = Path("data/processed/genomic_processed.csv")

    df = pd.read_csv(input_path, usecols=['Pangolin', 'Geo_Location', 'Collection_Date'])

    # Clean state names
    df['state'] = (
        df['Geo_Location']
        .str.replace(r'^USA:\s*', '', regex=True)
        .str.title()
        .str.strip()
        .str.replace(r',\s*([A-Z]{2})$', lambda m: f", {m.group(1).upper()}", regex=True)
    )

    # Parse dates with ISO week
    df['Collection_Date'] = pd.to_datetime(df['Collection_Date'], errors='coerce')
    df = df.dropna(subset=['Collection_Date'])
    df['week'] = df['Collection_Date'].dt.strftime('%G-W%V')  # ISO week format

    # Process variants
    df['Pangolin'] = (
        df['Pangolin']
        .str.replace(r'[^A-Z0-9.]', '', regex=True)
        .str.split('.', n=3).str[:3].str.join('.')
        .fillna('Unknown')
    )

    # Create dummy variables
    variants = pd.get_dummies(df['Pangolin'], prefix='Variant')
    variant_counts = (
        pd.concat([df[['state', 'week']], variants], axis=1)
        .groupby(['state', 'week'])
        .sum()
        .reset_index()
        .drop_duplicates()
    )

    variant_counts.to_csv(output_path, index=False)
    print(f"Processed {len(variant_counts)} genomic records")


if __name__ == "__main__":
    process_genomic()