import matplotlib.pyplot as plt
import numpy as np

from src.config import FIGURES_DIR


def get_transformed_features(model, X):
    """
    Apply feature engineering and preprocessing
    using the already-fitted pipeline steps.
    """

    feature_engineer = model.named_steps[
        "feature_engineering"
    ]

    preprocessor = model.named_steps[
        "preprocessing"
    ]

    X_features = feature_engineer.transform(X)

    X_transformed = preprocessor.transform(
        X_features
    )

    feature_names = feature_engineer.get_feature_names_out()

    return X_transformed, feature_names


def explain_model(
    model,
    X,
    max_samples=500
):
    """
    Generate SHAP explanations for the final model.

    The model must be a fitted imblearn Pipeline
    created by src.train.build_pipeline.
    """

    try:
        import shap

    except ImportError:
        raise ImportError(
            "Install SHAP using: pip install shap"
        )

    if len(X) > max_samples:

        X_sample = X.sample(
            n=max_samples,
            random_state=42
        )

    else:

        X_sample = X.copy()

    X_transformed, feature_names = get_transformed_features(
        model,
        X_sample
    )

    classifier = model.named_steps["model"]

    explainer = shap.TreeExplainer(
        classifier
    )

    shap_values = explainer.shap_values(
        X_transformed
    )

    if isinstance(shap_values, list):
        shap_values = shap_values[-1]

    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = shap_values[:, :, -1]

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    shap.summary_plot(
        shap_values,
        X_transformed,
        feature_names=feature_names,
        show=False
    )

    plt.savefig(
        FIGURES_DIR / "shap_summary.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    return shap_values


def get_feature_importance(model):
    """
    Return tree-based feature importance.
    """

    classifier = model.named_steps["model"]

    if not hasattr(classifier, "feature_importances_"):
        raise ValueError(
            "This model does not expose feature_importances_."
        )

    feature_engineer = model.named_steps[
        "feature_engineering"
    ]

    feature_names = feature_engineer.get_feature_names_out()

    importance = classifier.feature_importances_

    if len(feature_names) != len(importance):
        raise ValueError(
            "Feature names and importance dimensions do not match."
        )

    importances = (
        __import__("pandas")
        .DataFrame({
            "feature": feature_names,
            "importance": importance
        })
        .sort_values(
            by="importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    return importances


if __name__ == "__main__":

    import joblib

    from src.config import (
        BEST_MODEL_PATH,
        TARGET_COLUMN,
    )
    from src.data_loader import load_data

    df = load_data()

    X = df.drop(columns=[TARGET_COLUMN])

    model = joblib.load(BEST_MODEL_PATH)

    importance = get_feature_importance(model)

    print(importance.head(20))

    explain_model(
        model,
        X
    )