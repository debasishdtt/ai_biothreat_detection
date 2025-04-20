import tensorflow as tf
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    PrecisionRecallDisplay
)
import matplotlib.pyplot as plt


def evaluate():
    try:
        model = tf.keras.models.load_model(Path("models/threat_model.keras"))
        df = pd.read_csv(Path("data/processed/merged_data_with_labels.csv"))

        # Feature alignment with training
        variant_cols = [c for c in df.columns if c.startswith('Variant_')]
        top_variants = df[variant_cols].sum().nlargest(20).index.tolist()
        features = top_variants + [
            'rolling_growth',
            'retail_and_recreation_percent_change_from_baseline'
        ]

        X = df[features].astype('float32')
        y = df['threat'].astype('int8')

        # Get probabilities and find optimal threshold
        y_probs = model.predict(X).flatten()

        # Precision-Recall based threshold tuning
        precision, recall, thresholds = precision_recall_curve(y, y_probs)
        f2_scores = (5 * precision * recall) / (4 * precision + recall + 1e-8)
        optimal_idx = np.argmax(f2_scores)
        optimal_threshold = thresholds[optimal_idx]

        # Final predictions with optimal threshold
        y_pred = (y_probs >= optimal_threshold).astype(int)

        # Enhanced reporting
        print("\nOptimal Threshold:", round(optimal_threshold, 3))
        print("\nClassification Report:")
        print(classification_report(y, y_pred, zero_division=0, target_names=['Non-Threat', 'Threat']))

        # Confusion Matrix
        fig, ax = plt.subplots(1, 2, figsize=(15, 6))
        ConfusionMatrixDisplay.from_predictions(y, y_pred, ax=ax[0], colorbar=False)
        ax[0].set_title("Confusion Matrix")

        # Precision-Recall Curve
        PrecisionRecallDisplay.from_predictions(y, y_probs, ax=ax[1])
        ax[1].set_title("Precision-Recall Curve")
        ax[1].axvline(recall[optimal_idx], color='r', linestyle='--',
                      label=f'Optimal Threshold ({optimal_threshold:.2f})')
        ax[1].legend()

        plt.tight_layout()
        plt.savefig(Path("results/combined_metrics.png"))
        plt.close()

    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        raise


if __name__ == "__main__":
    evaluate()