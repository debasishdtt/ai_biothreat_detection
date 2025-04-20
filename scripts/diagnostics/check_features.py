import tensorflow as tf
import pandas as pd
from pathlib import Path


def check_features():
    try:
        # Check data first
        data_path = Path("data/processed/merged_data_with_labels.csv")
        if not data_path.exists():
            raise FileNotFoundError("Run preprocessing scripts first!")

        df = pd.read_csv(data_path)

        # Check model exists
        model_path = Path("models/threat_model.keras")
        if not model_path.exists():
            raise FileNotFoundError("Train model first! Run train_model.py")

        # Feature analysis
        variant_cols = [col for col in df.columns if col.startswith('Variant_')]
        print(f"Variant Features: {len(variant_cols)} (Should be ~395)")
        print(f"Total Features: {len(df.columns)}")

        # Model verification
        model = tf.keras.models.load_model(model_path)
        print(f"\nModel expects: {model.input_shape[1]} features")

    except Exception as e:
        print(f"Diagnostic failed: {str(e)}")
        print("Next Steps:")
        print("1. Run preprocessing scripts")
        print("2. Train model")
        print("3. Re-run this check")


if __name__ == "__main__":
    check_features()