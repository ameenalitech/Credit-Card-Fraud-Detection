from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_preprocessor():
    """
    Create preprocessing pipeline for numeric features.

    Steps:
    1. Median imputation
    2. Standard scaling
    """

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                slice(None)
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    return preprocessor


if __name__ == "__main__":

    from src.data_loader import load_data
    from src.feature_engineering import FeatureEngineer

    df = load_data()

    X = df.drop(columns=["Class"])

    feature_engineer = FeatureEngineer()

    X_features = feature_engineer.fit_transform(X)

    preprocessor = create_preprocessor()

    X_processed = preprocessor.fit_transform(X_features)

    print("Processed shape:", X_processed.shape)