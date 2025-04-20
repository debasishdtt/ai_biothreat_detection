import pandas as pd
import numpy as np
from pathlib import Path

def merge_data():
    try:
        print("🔍 Checking processed files...")
        processed_dir = Path("data/processed")
        required_files = {
            'epi': processed_dir / "epi_processed.csv",
            'genomic': processed_dir / "genomic_processed.csv",
            'mobility': processed_dir / "mobility_processed.csv"
        }

        # Validate files
        for name, path in required_files.items():
            if not path.exists():
                raise FileNotFoundError(f"Missing {name} file: {path}")

        print("✅ All files found\n🗂 Merging datasets...")

        # Load data
        epi = pd.read_csv(required_files['epi'], dtype={'week': 'string'})
        genomic = pd.read_csv(required_files['genomic'], dtype={'week': 'string'})
        mobility = pd.read_csv(required_files['mobility'], dtype={'week': 'string'})

        # Standardize state names
        genomic['Province_State'] = genomic['state'].str.title().str.strip()
        mobility['Province_State'] = mobility['Province_State'].str.title().str.strip()

        # Validate week formats
        week_pattern = r'^\d{4}-W(0[1-9]|[1-4][0-9]|5[0-3])$'  # Valid ISO weeks
        for name, df in [('Epidemiological', epi),
                        ('Genomic', genomic),
                        ('Mobility', mobility)]:
            invalid = df[~df['week'].str.match(week_pattern, na=False)]
            if not invalid.empty:
                print(f"❌ Invalid weeks in {name}:")
                print(invalid['week'].unique())
                raise ValueError("Fix week formatting in preprocessing")

        # Merge datasets
        merged = pd.merge(epi, genomic, on=['Province_State', 'week'], how='outer', validate='one_to_one')
        merged = pd.merge(merged, mobility, on=['Province_State', 'week'], how='outer', validate='one_to_one')

        # Final processing
        numeric_cols = merged.select_dtypes(include=np.number).columns
        merged[numeric_cols] = merged[numeric_cols].fillna(0)
        merged.to_csv(processed_dir / "merged_data.csv", index=False)

        print(f"\n✅ Merge successful! Final shape: {merged.shape}")
        print("Week coverage:", merged['week'].nunique(), "weeks")
        return True

    except Exception as e:
        print(f"\n❌ Merge failed: {str(e)}")
        return False

if __name__ == "__main__":
    merge_data()