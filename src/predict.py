import json

import joblib
import pandas as pd

from src.config import (
    BEST_MODEL_PATH,
    FEATURE_COLUMNS,
    MODELS_DIR,
    THRESHOLD_PATH,
)


def load_trained_model(
    model_path=BEST_MODEL_PATH
):
    """
    Load saved end-to-end pipeline.
    """

    try:

        model = joblib.load(model_path)

    except FileNotFoundError:

        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    return model


def load_saved_threshold(
    threshold_path=THRESHOLD_PATH,
    default_threshold=0.50
):
    """
    Load saved classification threshold.
    """

    try:

        with open(
            threshold_path,
            "r",
            encoding="utf-8"
        ) as file:

            result = json.load(file)

        threshold = float(
            result["threshold"]
        )

    except FileNotFoundError:

        threshold = default_threshold

    if not 0 < threshold < 1:
        raise ValueError(
            "Threshold must be between 0 and 1."
        )

    return threshold



def validate_prediction_input(data):
    """
    Validate transaction data and ignore target column.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "Input must be a pandas DataFrame."
        )

    # Remove target column if it exists
    data = data.drop(
        columns=["Class"],
        errors="ignore"
    )

    # Check required features
    missing_columns = [
        column
        for column in FEATURE_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required features: {missing_columns}"
        )

    # Keep only required model features
    data = data[FEATURE_COLUMNS].copy()

    # Convert features to numeric
    for column in FEATURE_COLUMNS:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    # Check invalid values
    if data.isnull().any().any():
        raise ValueError(
            "Dataset contains missing or non-numeric values."
        )

    return data



def predict_transactions(
    data,
    model=None,
    threshold=None
):
    """
    Predict fraud probabilities and labels.

    Parameters
    ----------
    data : pandas.DataFrame
        New transactions with original feature columns.

    model : fitted pipeline, optional
        If omitted, the saved model is loaded.

    threshold : float, optional
        If omitted, the saved threshold is loaded.

    Returns
    -------
    pandas.DataFrame
        Fraud probability and predicted class.
    """

    data = validate_prediction_input(data)

    if model is None:
        model = load_trained_model()

    if threshold is None:
        threshold = load_saved_threshold()

    if not 0 < threshold < 1:
        raise ValueError(
            "Threshold must be between 0 and 1."
        )

    if not hasattr(model, "predict_proba"):
        raise AttributeError(
            "The model must support predict_proba."
        )

    probabilities = model.predict_proba(data)

    classes = list(model.classes_)

    positive_class_index = classes.index(1)

    fraud_probabilities = probabilities[
        :,
        positive_class_index
    ]

    predictions = (
        fraud_probabilities >= threshold
    ).astype(int)

    results = pd.DataFrame(
        {
            "fraud_probability": fraud_probabilities,
            "predicted_class": predictions,
        },
        index=data.index
    )

    results["prediction_label"] = (
        results["predicted_class"]
        .map({
            0: "Normal",
            1: "Fraud"
        })
    )

    return results


def predict_single_transaction(
    transaction,
    model=None,
    threshold=None
):
    """
    Predict one transaction represented as a dictionary.
    """

    data = pd.DataFrame(
        [transaction]
    )

    return predict_transactions(
        data,
        model=model,
        threshold=threshold
    )


if __name__ == "__main__":

    model = load_trained_model()

    sample = pd.DataFrame(
        [
            {
                column: 0.0
                for column in FEATURE_COLUMNS
            }
        ]
    )

    result = predict_transactions(
        sample,
        model=model
    )

    print(result)