
"""
Tests for data loading and validation.
"""

import pandas as pd
import pytest
from src.config import (
    FEATURE_COLUMNS,
    RAW_DATA_PATH,
    TARGET_COLUMN,
)
from src.data_loader import load_data
from src.validation import validate_dataset


def test_raw_dataset_exists():

    assert RAW_DATA_PATH.exists(), (
        "Raw dataset file does not exist."
    )


def test_load_data_returns_dataframe():

    df = load_data(RAW_DATA_PATH)

    assert isinstance(df, pd.DataFrame)

    assert not df.empty


def test_dataset_contains_required_columns():

    df = load_data(RAW_DATA_PATH)

    required_columns = (
        set(FEATURE_COLUMNS) | {TARGET_COLUMN}
    )

    assert required_columns.issubset(
        set(df.columns)
    )


def test_dataset_validation():

    df = load_data(RAW_DATA_PATH)

    # Should complete without raising an exception
    validate_dataset(df)
