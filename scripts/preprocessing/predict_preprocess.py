import subprocess
from pathlib import Path


def preprocess_new_data():
    """Process new data using modified preprocessing scripts"""
    new_data_dir = Path("data/raw/new")

    commands = [
        f"python scripts/preprocessing/preprocess_epidemiological.py --input {new_data_dir}/epidemiological/new_epi_data.csv",
        f"python scripts/preprocessing/preprocess_genomic.py --input {new_data_dir}/genomic/new_genomic_data.csv",
        f"python scripts/preprocessing/preprocess_mobility.py --input {new_data_dir}/mobility/new_mobility_data.csv",
        "python scripts/preprocessing/merge_datasets.py --predict"
    ]

    for cmd in commands:
        subprocess.run(cmd, shell=True, check=True)


if __name__ == "__main__":
    preprocess_new_data()