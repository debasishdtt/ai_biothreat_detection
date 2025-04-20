import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import precision_recall_curve, roc_curve


def generate_report(y_true: np.ndarray, y_probs: np.ndarray):
    try:
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_probs)
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(recall, precision)
        plt.title("Precision-Recall Curve")

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        plt.subplot(1, 2, 2)
        plt.plot(fpr, tpr)
        plt.title("ROC Curve")

        plt.savefig(Path("results/combined_metrics.png"))
        plt.close()

    except Exception as e:
        print(f"Report generation failed: {str(e)}")
        raise


if __name__ == "__main__":
    # Get data from evaluate_model
    from evaluate_model import evaluate

    y_probs = evaluate(
        model_path=Path("models/threat_model.keras"),
        data_path=Path("data/processed/merged_data.csv")
    )

    # Load true labels
    df = pd.read_csv(Path("data/processed/merged_data.csv"))
    y_true = df['threat'].astype('int32')

    generate_report(y_true, y_probs)