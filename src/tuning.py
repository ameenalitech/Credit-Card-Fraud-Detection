import joblib
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from src.config import (
    MODELS_DIR,
    RANDOM_STATE,
    TUNED_MODEL_PATH,
)
from src.train import build_pipeline


def get_parameter_grid(model_name):
    """
    Return hyperparameter search space.
    """

    if model_name == "random_forest":

        return {
            "model__n_estimators": [100, 200, 300, 500],
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
        }

    elif model_name == "logistic_regression":

        return {
            "model__C": [0.01, 0.1, 1, 10, 100],
            "model__solver": ["lbfgs"],
            "model__class_weight": [None, "balanced"],
        }

    elif model_name == "xgboost":

        return {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [3, 5, 7],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__subsample": [0.7, 0.8, 1.0],
            "model__colsample_bytree": [0.7, 0.8, 1.0],
        }

    else:

        raise ValueError(
            "Unsupported model name."
        )


def tune_model(
    X_train,
    y_train,
    model_name="random_forest",
    sampling_method="smote",
    n_iter=20,
    cv_splits=5,
    scoring="average_precision"
):
    """
    Tune model hyperparameters using stratified CV.

    The test set must not be passed to this function.
    """

    pipeline = build_pipeline(
        model_name=model_name,
        sampling_method=sampling_method
    )

    parameter_grid = get_parameter_grid(
        model_name
    )

    cross_validation = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=parameter_grid,
        n_iter=n_iter,
        scoring=scoring,
        cv=cross_validation,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        refit=True
    )

    search.fit(
        X_train,
        y_train
    )

    return search


def save_tuned_model(search, file_path=TUNED_MODEL_PATH):
    """
    Save the best tuned pipeline.
    """

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        search.best_estimator_,
        file_path
    )

    print(f"Tuned model saved to: {file_path}")

    return search.best_estimator_


if __name__ == "__main__":

    from sklearn.model_selection import train_test_split

    from src.config import TARGET_COLUMN
    from src.data_loader import load_data

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

    search = tune_model(
        X_train,
        y_train,
        model_name="random_forest",
        sampling_method="smote",
        n_iter=10,
        cv_splits=5
    )

    print("Best parameters:")
    print(search.best_params_)

    print("Best CV Average Precision:")
    print(search.best_score_)

    save_tuned_model(search)