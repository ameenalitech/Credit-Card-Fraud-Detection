import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Create additional features for fraud detection.

    New features:
    - Amount_log1p
    - Time_log1p
    """

    def __init__(
        self,
        add_amount_log=True,
        add_time_log=True
    ):

        self.add_amount_log = add_amount_log
        self.add_time_log = add_time_log

        self.feature_names_in_ = None

    def fit(self, X, y=None):

        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "FeatureEngineer expects a pandas DataFrame."
            )

        self.feature_names_in_ = X.columns.tolist()

        required_columns = ["Amount", "Time"]

        missing_columns = [
            column
            for column in required_columns
            if column not in X.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing columns: {missing_columns}"
            )

        return self

    def transform(self, X):

        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "FeatureEngineer expects a pandas DataFrame."
            )

        if self.feature_names_in_ is None:
            raise RuntimeError(
                "FeatureEngineer must be fitted before transform."
            )

        if X.columns.tolist() != self.feature_names_in_:
            raise ValueError(
                "Input columns do not match the columns used during fit."
            )

        transformed = X.copy()

        if self.add_amount_log:

            if (transformed["Amount"] < 0).any():
                raise ValueError(
                    "Amount contains negative values."
                )

            transformed["Amount_log1p"] = np.log1p(
                transformed["Amount"]
            )

        if self.add_time_log:

            if (transformed["Time"] < 0).any():
                raise ValueError(
                    "Time contains negative values."
                )

            transformed["Time_log1p"] = np.log1p(
                transformed["Time"]
            )

        return transformed

    def get_feature_names_out(self, input_features=None):

        if input_features is None:
            input_features = self.feature_names_in_

        feature_names = list(input_features)

        if self.add_amount_log:
            feature_names.append("Amount_log1p")

        if self.add_time_log:
            feature_names.append("Time_log1p")

        return np.array(feature_names, dtype=object)


def create_features(data):
    """
    Convenience function to create features outside a pipeline.
    """

    transformer = FeatureEngineer()

    return transformer.fit_transform(data)


if __name__ == "__main__":

    from src.data_loader import load_data

    df = load_data()

    X = df.drop(columns=["Class"])

    X_transformed = create_features(X)

    print(X_transformed.head())
    print("Original shape:", X.shape)
    print("New shape:", X_transformed.shape)