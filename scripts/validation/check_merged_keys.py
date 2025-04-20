import pandas as pd
from pathlib import Path


def check_merge_keys():
    files = {
        'epi': Path("data/processed/epi_processed.csv"),
        'genomic': Path("data/processed/genomic_processed.csv"),
        'mobility': Path("data/processed/mobility_processed.csv")
    }

    for name, path in files.items():
        df = pd.read_csv(path)
        duplicates = df.duplicated(['Province_State', 'week']).sum()
        print(f"{name} duplicates: {duplicates}")


if __name__ == "__main__":
    check_merge_keys()