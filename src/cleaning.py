import numpy as np
import pandas as pd

from src.config import (
    CLEANED_DATA_PATH,
    PROCESSED_DATA_DIR,
    RAW_DATA_PATH,
    TARGET_COLUMN,
)
from src.data_loader import load_data
from src.validation import validate_dataset


def replace_infinite_values(data):
    """
    Replace positive and negative infinity with NaN.
    """

    cleaned_data = data.copy()

    cleaned_data = cleaned_data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return cleaned_data


def remove_exact_duplicates(data):
    """
    Remove exact duplicate rows.

    Use this function only after confirming that
    duplicate rows are invalid records for the project.
    """

    cleaned_data = data.drop_duplicates().copy()

    return cleaned_data


def handle_missing_values(data, strategy="median"):
    """
    Handle missing values in feature columns.

    The target column is never imputed.
    """

    cleaned_data = data.copy()

    if cleaned_data[TARGET_COLUMN].isnull().any():
        raise ValueError(
            "Target contains missing values. "
            "Resolve target missingness before training."
        )

    feature_columns = [
        column
        for column in cleaned_data.columns
        if column != TARGET_COLUMN
    ]

    if strategy == "median":

        for column in feature_columns:

            median_value = cleaned_data[column].median()

            cleaned_data[column] = cleaned_data[column].fillna(
                median_value
            )

    elif strategy == "mean":

        for column in feature_columns:

            mean_value = cleaned_data[column].mean()

            cleaned_data[column] = cleaned_data[column].fillna(
                mean_value
            )

    elif strategy == "drop":

        cleaned_data = cleaned_data.dropna(
            subset=feature_columns
        )

    else:
        raise ValueError(
            "Supported strategies are: median, mean, drop."
        )

    return cleaned_data


def clean_data(
    data,
    remove_duplicates=False,
    missing_strategy="median"
):
    """
    Run the cleaning process.

    Parameters
    ----------
    data : pandas.DataFrame
        Raw dataset.

    remove_duplicates : bool
        Whether exact duplicates should be removed.

    missing_strategy : str
        Missing values strategy.

    Returns
    -------
    pandas.DataFrame
        Cleaned dataset.
    """

    cleaned_data = data.copy()

    cleaned_data = replace_infinite_values(cleaned_data)

    if remove_duplicates:
        cleaned_data = remove_exact_duplicates(cleaned_data)

    cleaned_data = handle_missing_values(
        cleaned_data,
        strategy=missing_strategy
    )

    return cleaned_data


def save_cleaned_data(data, file_path=CLEANED_DATA_PATH):
    """
    Save cleaned dataset.
    """

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    data.to_csv(
        file_path,
        index=False
    )

    print(f"Cleaned data saved to: {file_path}")


if __name__ == "__main__":

    df = load_data(RAW_DATA_PATH)

    validate_dataset(df)

    cleaned_df = clean_data(
        df,
        remove_duplicates=False,
        missing_strategy="median"
    )

    save_cleaned_data(cleaned_df)

    print("Cleaning completed.")
    print(cleaned_df.shape)