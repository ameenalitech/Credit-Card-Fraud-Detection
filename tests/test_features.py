
"""
Tests for feature engineering.
"""

import numpy as np
import pandas as pd
from src.feature_engineering import (
    FeatureEngineer,
    create_features,
)


def create_sample_data():

    return pd.DataFrame({

        "Time": [
            0,
            60,
            120,
            180,
            240
        ],

        "Amount": [
            10,
            50,
            100,
            250,
            500
        ],

    })


def test_create_features_returns_dataframe():

    df = create_sample_data()

    result = create_features(df)

    assert isinstance(result, pd.DataFrame)


def test_feature_engineering_preserves_rows():

    df = create_sample_data()

    result = create_features(df)

    assert len(result) == len(df)


def test_new_features_are_created():

    df = create_sample_data()

    result = create_features(df)

    expected_features = {
        "Amount_log1p",
        "Time_log1p"
    }

    assert expected_features.issubset(
        set(result.columns)
    )


def test_new_features_are_finite():

    df = create_sample_data()

    result = create_features(df)

    new_features = [
        "Amount_log1p",
        "Time_log1p"
    ]

    assert np.isfinite(
        result[new_features].to_numpy()
    ).all()


def test_feature_engineer_transform():

    df = create_sample_data()

    engineer = FeatureEngineer()

    result = engineer.fit_transform(df)

    assert "Amount_log1p" in result.columns

    assert "Time_log1p" in result.columns
