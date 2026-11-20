
"""
Credit Card Fraud Detection
Main Training Pipeline

This script runs the complete model training workflow.
"""

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.cleaning import clean_data
from src.config import (
    RANDOM_STATE,
    RAW_DATA_PATH,
    TARGET_COLUMN,
    THRESHOLD_PATH,
    TUNED_MODEL_PATH,
)
from src.data_loader import load_data
from src.evaluate import (
    choose_threshold,
    evaluate_model,
    get_prediction_probabilities,
    save_threshold,
)
from src.train import build_pipeline
from src.validation import validate_dataset


def main():

    print("=" * 60)
    print("CREDIT CARD FRAUD DETECTION")
    print("MAIN TRAINING PIPELINE")
    print("=" * 60)

    # --------------------------------------
    # 1. Load Dataset
    # --------------------------------------

    print("\n[1/8] Loading dataset...")

    df = load_data(RAW_DATA_PATH)

    print("Dataset loaded successfully.")
    print("Dataset shape:", df.shape)

    # --------------------------------------
    # 2. Validate Dataset
    # --------------------------------------

    print("\n[2/8] Validating dataset...")

    validate_dataset(df)

    print("Dataset validation completed.")

    # --------------------------------------
    # 3. Clean Dataset
    # --------------------------------------

    print("\n[3/8] Cleaning dataset...")

    df = clean_data(
        df,
        remove_duplicates=False
    )

    print("Cleaning completed.")
    print("Cleaned shape:", df.shape)

    # --------------------------------------
    # 4. Split Features and Target
    # --------------------------------------

    print("\n[4/8] Splitting features and target...")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # First split: 80% development, 20% test
    X_development, X_test, y_development, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    )

    # Second split: 75% training, 25% validation
    # Final proportions: 60% train, 20% validation, 20% test

    X_train, X_val, y_train, y_val = train_test_split(
        X_development,
        y_development,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_development,
    )

    print("Training samples:", len(X_train))
    print("Validation samples:", len(X_val))
    print("Testing samples:", len(X_test))

    # --------------------------------------
    # 5. Build and Train Pipeline
    # --------------------------------------

    print("\n[5/8] Building model pipeline...")

    model = build_pipeline(
        model_name="random_forest",
        sampling_method="smote",
    )

    print("Training model...")

    model.fit(X_train, y_train)

    print("Model training completed.")

    # --------------------------------------
    # 6. Select Threshold on Validation Set
    # --------------------------------------

    print("\n[6/8] Selecting decision threshold...")

    val_probabilities = get_prediction_probabilities(
        model,
        X_val
    )

    best_threshold = choose_threshold(
        y_val,
        val_probabilities,
        metric="f1"
    )

    print("Selected threshold:", best_threshold)

    # --------------------------------------
    # 7. Save Model and Threshold
    # --------------------------------------

    print("\n[7/8] Saving model and threshold...")

    TUNED_MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    THRESHOLD_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        TUNED_MODEL_PATH
    )

    save_threshold(
        best_threshold,
        THRESHOLD_PATH
    )

    print("Model saved:", TUNED_MODEL_PATH)
    print("Threshold saved:", THRESHOLD_PATH)

    # --------------------------------------
    # 8. Final Evaluation on Test Set
    # --------------------------------------

    print("\n[8/8] Evaluating final model...")

    test_metrics = evaluate_model(
        model,
        X_test,
        y_test,
        threshold=best_threshold
    )

    print("\nFinal Test Metrics")
    print("-" * 40)

    for metric_name, metric_value in test_metrics.items():

        print(f"{metric_name}: {metric_value}")

    print("\nTraining workflow completed successfully.")

    print("=" * 60)


if __name__ == "__main__":
    main()
