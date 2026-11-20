import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import (
    DEFAULT_THRESHOLD,
    METRICS_DIR,
    MODELS_DIR,
    TARGET_COLUMN,
    THRESHOLD_PATH,
)


def get_prediction_probabilities(model, X):
    """
    Return probability of the positive class.
    """

    if not hasattr(model, "predict_proba"):
        raise AttributeError(
            "The supplied model does not support predict_proba."
        )

    probabilities = model.predict_proba(X)

    classes = list(model.classes_)

    positive_class_index = classes.index(1)

    return probabilities[:, positive_class_index]


def calculate_metrics(
    y_true,
    y_prob,
    threshold=DEFAULT_THRESHOLD
):
    """
    Calculate classification metrics.
    """

    y_pred = (
        np.asarray(y_prob) >= threshold
    ).astype(int)

    metrics = {
        "threshold": float(threshold),

        "accuracy": accuracy_score(
            y_true,
            y_pred
        ),

        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "f1_score": f1_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "roc_auc": roc_auc_score(
            y_true,
            y_prob
        ),

        "average_precision": average_precision_score(
            y_true,
            y_prob
        ),

        "mcc": matthews_corrcoef(
            y_true,
            y_pred
        ),

        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            y_pred
        ),
    }

    return metrics


def choose_threshold(
    y_true,
    y_prob,
    metric="f1"
):
    """
    Select a classification threshold using validation data.

    Supported metrics:
    - f1
    - recall

    Do not use the test set for threshold selection.
    """

    precision, recall, thresholds = precision_recall_curve(
        y_true,
        y_prob
    )

    if len(thresholds) == 0:
        raise ValueError(
            "Unable to calculate thresholds."
        )

    if metric == "f1":

        f1_values = (
            2 * precision[:-1] * recall[:-1]
        ) / (
            precision[:-1] + recall[:-1] + 1e-12
        )

        best_index = int(np.argmax(f1_values))

    elif metric == "recall":

        valid_indices = np.where(
            precision[:-1] >= 0.80
        )[0]

        if len(valid_indices) == 0:
            raise ValueError(
                "No threshold meets the minimum precision requirement."
            )

        best_index = valid_indices[
            np.argmax(recall[valid_indices])
        ]

    else:

        raise ValueError(
            "Supported metrics are: f1 and recall."
        )

    return float(thresholds[best_index])


def save_threshold(
    threshold,
    file_path=THRESHOLD_PATH
):
    """
    Save selected threshold.
    """

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(file_path, "w", encoding="utf-8") as file:

        json.dump(
            {"threshold": float(threshold)},
            file,
            indent=4
        )


def load_threshold(
    file_path=THRESHOLD_PATH
):
    """
    Load saved threshold.
    """

    with open(file_path, "r", encoding="utf-8") as file:

        result = json.load(file)

    return float(result["threshold"])


def plot_confusion_matrix(
    y_true,
    y_prob,
    threshold=DEFAULT_THRESHOLD,
    save=True
):
    """
    Plot confusion matrix.
    """

    y_pred = (
        np.asarray(y_prob) >= threshold
    ).astype(int)

    matrix = confusion_matrix(
        y_true,
        y_pred
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Normal", "Fraud"]
    )

    display.plot(
        cmap="Blues",
        values_format="d"
    )

    plt.title("Fraud Detection Confusion Matrix")

    if save:

        METRICS_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        plt.savefig(
            METRICS_DIR / "confusion_matrix.png",
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()


def evaluate_model(
    model,
    X_test,
    y_test,
    threshold=DEFAULT_THRESHOLD
):
    """
    Evaluate a trained model on test data.
    """

    y_prob = get_prediction_probabilities(
        model,
        X_test
    )

    metrics = calculate_metrics(
        y_test,
        y_prob,
        threshold
    )

    y_pred = (
        y_prob >= threshold
    ).astype(int)

    print("\nClassification Metrics")
    print("----------------------")

    for metric_name, metric_value in metrics.items():

        print(
            f"{metric_name}: {metric_value:.4f}"
        )

    print("\nClassification Report")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Normal", "Fraud"],
            zero_division=0
        )
    )

    plot_confusion_matrix(
        y_test,
        y_prob,
        threshold
    )

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    pd.DataFrame(
        [metrics]
    ).to_csv(
        METRICS_DIR / "test_metrics.csv",
        index=False
    )

    return metrics


if __name__ == "__main__":

    from sklearn.model_selection import train_test_split

    from src.config import (
        BEST_MODEL_PATH,
        RANDOM_STATE,
        RAW_DATA_PATH,
        TEST_SIZE,
    )
    from src.data_loader import load_data

    df = load_data(RAW_DATA_PATH)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    model = joblib.load(BEST_MODEL_PATH)

    evaluate_model(
        model,
        X_test,
        y_test,
        threshold=DEFAULT_THRESHOLD
    )