import tensorflow as tf
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def train():
    try:
        # Load merged data
        df = pd.read_csv(Path("data/processed/merged_data.csv"))

        # Feature engineering with index-safe operations
        df = df.sort_values(['Province_State', 'week']).reset_index(drop=True)

        # Calculate case growth with proper group alignment
        df['case_growth'] = (
            df.groupby('Province_State', group_keys=False)['cases']
            .apply(lambda x: x.pct_change().replace([np.inf, -np.inf], np.nan))
            .fillna(0)
        )

        # Exponential moving average with index preservation
        df['rolling_growth'] = (
            df.groupby('Province_State', group_keys=False)['case_growth']
            .transform(lambda x: x.ewm(span=3).mean().shift(1))
            .fillna(0)
        )

        # Threat labeling with index-safe operations
        variant_cols = [c for c in df.columns if c.startswith('Variant_')]
        top_variants = df[variant_cols].sum().nlargest(20).index.tolist()

        # Ensure boolean conditions are series with same index
        growth_condition = df['rolling_growth'] > 0.3
        variant_condition = df[top_variants].sum(axis=1) >= 1
        mobility_condition = df['retail_and_recreation_percent_change_from_baseline'] < -15

        df['threat'] = (
            (growth_condition & variant_condition & mobility_condition)
            .astype('int8')
        )

        # Save labeled data
        df.to_csv(Path("data/processed/merged_data_with_labels.csv"), index=False)

        # Prepare features with validated columns
        features = top_variants + [
            'rolling_growth',
            'retail_and_recreation_percent_change_from_baseline'
        ]
        X = df[features].astype('float32')
        y = df['threat']

        # Stratified split with index alignment
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )

        # SMOTE with proper resampling
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X_train, y_train)

        # Enhanced model architecture
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(X_res.shape[1],)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['Recall', 'Precision', tf.keras.metrics.AUC(name='pr_auc', curve='PR')]
        )

        # Early stopping with validation monitoring
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_pr_auc',
            patience=7,
            mode='max',
            restore_best_weights=True
        )

        history = model.fit(
            X_res, y_res,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=[early_stop],
            verbose=1
        )

        model.save(Path("models/threat_model.keras"))
        print(f"Model trained with {X_res.shape[1]} features")

    except Exception as e:
        print(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    train()