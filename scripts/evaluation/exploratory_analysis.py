import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def analyze_data():
    try:
        df = pd.read_csv(Path("data/processed/merged_data_with_labels.csv"))

        # Variant analysis
        variant_cols = [c for c in df.columns if c.startswith('Variant_')]
        if variant_cols:
            plt.figure(figsize=(12, 6))
            df[variant_cols].sum().sort_values(ascending=False).head(20).plot(
                kind='bar', title="Top 20 COVID Variants"
            )
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(Path("results/variants.png"))
            plt.close()

        # Case-mobility correlation
        plt.figure(figsize=(10, 6))
        plt.scatter(df['retail_and_recreation_percent_change_from_baseline'],
                    df['cases'], alpha=0.3)
        plt.title("Mobility vs Confirmed Cases")
        plt.xlabel("Retail Mobility Change (%)")
        plt.ylabel("Confirmed Cases")
        plt.savefig(Path("results/mobility_cases.png"))
        plt.close()

        # Class distribution
        if 'threat' in df.columns:
            plt.figure(figsize=(8, 6))
            df['threat'].value_counts().plot(
                kind='pie', autopct='%1.1f%%',
                labels=['Non-Threat', 'Threat']
            )
            plt.title("Threat Class Distribution")
            plt.savefig(Path("results/class_distribution.png"))
            plt.close()
        else:
            print("Warning: Threat labels not found")

    except Exception as e:
        print(f"Analysis failed: {str(e)}")
        raise


if __name__ == "__main__":
    analyze_data()