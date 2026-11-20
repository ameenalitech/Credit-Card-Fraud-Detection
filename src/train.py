import json

import joblib
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.config import (
    BASELINE_COMPARISON_PATH,
    BEST_MODEL_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    TARGET_COLUMN,
)
from src.data_loader import load_data
from src.feature_engineering import FeatureEngineer
from src.imbalance import create_sampler
from src.preprocessing import create_preprocessor


def get_model(model_name="random_forest"):
    """
    Return a classification model.
    """

    model_name = model_name.lower()

    if model_name == "logistic_regression":

        return LogisticRegression(
            max_iter=2000,
            class_weight=None,
            random_state=RANDOM_STATE
        )

    elif model_name == "random_forest":

        return RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            class_weight=None,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    elif model_name == "xgboost":

        try:
            from xgboost import XGBClassifier

        except ImportError:
            raise ImportError(
                "Install XGBoost using: pip install xgboost"
            )

        return XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    else:

        raise ValueError(
            "Model must be logistic_regression, "
            "random_forest, or xgboost."
        )


def build_pipeline(
    model_name="random_forest",
    sampling_method="smote"
):
    """
    Build complete training pipeline.
    """

    pipeline = Pipeline(
        steps=[
            (
                "feature_engineering",
                FeatureEngineer()
            ),
            (
                "preprocessing",
                create_preprocessor()
            ),
            (
                "sampler",
                create_sampler(
                    method=sampling_method,
                    random_state=RANDOM_STATE
                )
            ),
            (
                "model",
                get_model(model_name)
            ),
        ]
    )

    return pipeline


def train_model(
    X_train,
    y_train,
    model_name="random_forest",
    sampling_method="smote"
):
    """
    Train a complete machine learning pipeline.
    """

    pipeline = build_pipeline(
        model_name=model_name,
        sampling_method=sampling_method
    )

    pipeline.fit(
        X_train,
        y_train
    )

    return pipeline


def save_model(model, file_path=BEST_MODEL_PATH):
    """
    Save trained model.
    """

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        file_path
    )

    print(f"Model saved to: {file_path}")


def train_baseline_models(
    X_train,
    y_train,
    sampling_method="none"
):
    """
    Train baseline models and return a dictionary.
    """

    model_names = [
        "logistic_regression",
        "random_forest",
        "xgboost"
    ]

    trained_models = {}

    for model_name in model_names:

        try:

            print(f"Training {model_name}...")

            model = train_model(
                X_train,
                y_train,
                model_name=model_name,
                sampling_method=sampling_method
            )

            trained_models[model_name] = model

        except ImportError as error:

            print(error)

    return trained_models


if __name__ == "__main__":

    from sklearn.model_selection import train_test_split

    df = load_data()

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE
    )

    model = train_model(
        X_train,
        y_train,
        model_name="random_forest",
        sampling_method="smote"
    )

    save_model(model)

    print("Training completed.")