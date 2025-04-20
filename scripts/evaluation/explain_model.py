import tensorflow as tf
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from pathlib import Path


def explain_model():
    try:
        model = tf.keras.models.load_model(Path("models/threat_model.keras"))
        df = pd.read_csv(Path("data/processed/merged_data_with_labels.csv"))

        # REPLICATE TRAINING FEATURE SELECTION PROCESS
        variant_cols = [c for c in df.columns if c.startswith('Variant_')]
        top_variants = df[variant_cols].sum().nlargest(20).index.tolist()  # Match training

        features = top_variants + [
            'rolling_growth',
            'retail_and_recreation_percent_change_from_baseline'
        ]

        # Clean data
        sample_data = df[features].replace([np.inf, -np.inf], np.nan).dropna()
        sample = sample_data.sample(50, random_state=42).astype('float32')

        # Verify feature count matches model expectations
        assert len(features) == model.input_shape[1], \
            f"Feature mismatch: Model expects {model.input_shape[1]} features, got {len(features)}"

        # SHAP explanation with correct features
        explainer = shap.KernelExplainer(
            model.predict,
            shap.sample(sample, 10)  # Background sample
        )
        shap_values = explainer.shap_values(sample)

        plt.figure(figsize=(15, 10))
        shap.summary_plot(shap_values, sample, feature_names=features, show=False)
        plt.title("Feature Impact Analysis (Top 20 Variants + Key Metrics)", fontsize=14)
        plt.tight_layout()
        plt.savefig(Path("results/shap_summary.png"), dpi=300)
        plt.close()

        print("SHAP analysis completed with correct features")

    except Exception as e:
        print(f"Explanation failed: {str(e)}")
        raise


if __name__ == "__main__":
    explain_model()