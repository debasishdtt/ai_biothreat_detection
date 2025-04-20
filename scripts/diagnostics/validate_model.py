# scripts/diagnostics/validate_model.py
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import matthews_corrcoef, fbeta_score


def load_data():
    """Load and prepare training data"""
    df = pd.read_csv(Path("data/processed/merged_data_with_labels.csv"))

    # Replicate feature engineering
    variant_cols = [c for c in df.columns if c.startswith('Variant_')]
    top_variants = df[variant_cols].sum().nlargest(20).index.tolist()

    features = top_variants + [
        'rolling_growth',
        'retail_and_recreation_percent_change_from_baseline'
    ]

    X = df[features].astype('float32')
    y = df['threat'].astype('int8')
    return X, y


def cross_validate():
    # Load model and data
    model = tf.keras.models.load_model(Path("models/threat_model.keras"))
    X, y = load_data()

    # Time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=3)  # Reduced for speed
    metrics = []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Clone model to reset weights
        cloned_model = tf.keras.models.clone_model(model)
        cloned_model.set_weights(model.get_weights())

        cloned_model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['Recall']
        )

        # Train on fold
        cloned_model.fit(
            X_train, y_train,
            epochs=15,
            verbose=0,
            batch_size=32
        )

        # Evaluate
        y_pred = (cloned_model.predict(X_test) > 0.5).astype(int)
        metrics.append({
            'mcc': matthews_corrcoef(y_test, y_pred),
            'f2': fbeta_score(y_test, y_pred, beta=2)
        })

    print(f"Cross-validated MCC: {np.mean([m['mcc'] for m in metrics]):.2f}")
    print(f"F2-Scores: {[m['f2'] for m in metrics]}")


if __name__ == "__main__":
    cross_validate()