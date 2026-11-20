import numpy as np
import pandas as pd

from src.config import (
    EXPECTED_COLUMNS,
    TARGET_COLUMN,
)


def validate_columns(data):
    """
    Validate that all required columns exist.
    """

    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    return True


def validate_target(data):
    """
    Validate the target column.
    """

    if TARGET_COLUMN not in data.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' does not exist."
        )

    if data[TARGET_COLUMN].isnull().any():
        raise ValueError(
            "Target column contains missing values."
        )

    unique_values = set(data[TARGET_COLUMN].unique())

    if not unique_values.issubset({0, 1}):
        raise ValueError(
            f"Target must contain only 0 and 1. Found: {unique_values}"
        )

    if len(unique_values) < 2:
        raise ValueError(
            "Target must contain both classes: 0 and 1."
        )

    return True


def validate_numeric_columns(data):
    """
    Check whether expected dataset columns are numeric.
    """

    non_numeric_columns = data[
        EXPECTED_COLUMNS
    ].select_dtypes(exclude=[np.number]).columns.tolist()

    if non_numeric_columns:
        raise TypeError(
            f"Non-numeric columns found: {non_numeric_columns}"
        )

    return True


def validate_infinite_values(data):
    """
    Check whether numeric columns contain infinite values.
    """

    numeric_data = data.select_dtypes(include=[np.number])

    infinite_counts = np.isinf(numeric_data).sum()

    columns_with_infinity = infinite_counts[
        infinite_counts > 0
    ].to_dict()

    if columns_with_infinity:
        raise ValueError(
            "Infinite values detected: "
            f"{columns_with_infinity}"
        )

    return True


def validate_dataset(data):
    """
    Run all dataset validation checks.
    """

    validate_columns(data)
    validate_target(data)
    validate_numeric_columns(data)
    validate_infinite_values(data)

    return True


if __name__ == "__main__":

    from src.data_loader import load_data

    df = load_data()

    validate_dataset(df)

    print("Dataset validation passed.")