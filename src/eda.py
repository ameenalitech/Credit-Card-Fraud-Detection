import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.utils.multiclass import type_of_target

from src.config import (
    FIGURES_DIR,
    TARGET_COLUMN,
)


def get_target_distribution(data):
    """
    Return the count of each target class.
    """

    return data[TARGET_COLUMN].value_counts().sort_index()


def get_target_percentage(data):
    """
    Return the percentage of each target class.
    """

    return (
        data[TARGET_COLUMN]
        .value_counts(normalize=True)
        .sort_index()
        * 100
    )


def get_numeric_summary(data):
    """
    Generate descriptive statistics for numeric columns.
    """

    return data.describe().T


def get_missing_values(data):
    """
    Return missing values summary.
    """

    missing = data.isnull().sum()

    result = pd.DataFrame({
        "missing_count": missing,
        "missing_percentage": (
            missing / len(data)
        ) * 100
    })

    return result.sort_values(
        by="missing_percentage",
        ascending=False
    )


def plot_target_distribution(data, save=True):
    """
    Plot target class distribution.
    """

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=data,
        x=TARGET_COLUMN
    )

    plt.title("Fraud vs Non-Fraud Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of Transactions")

    if save:
        plt.savefig(
            FIGURES_DIR / "target_distribution.png",
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()


def plot_correlation_matrix(data, save=True):
    """
    Plot feature correlation matrix.
    """

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    correlation = data.corr(numeric_only=True)

    plt.figure(figsize=(18, 14))

    sns.heatmap(
        correlation,
        cmap="coolwarm",
        center=0
    )

    plt.title("Correlation Matrix")

    if save:
        plt.savefig(
            FIGURES_DIR / "correlation_matrix.png",
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()


def plot_amount_distribution(data, save=True):
    """
    Plot transaction amount distribution.
    """

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(figsize=(10, 5))

    sns.histplot(
        data=data,
        x="Amount",
        bins=50,
        kde=True
    )

    plt.title("Transaction Amount Distribution")

    if save:
        plt.savefig(
            FIGURES_DIR / "amount_distribution.png",
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()


def generate_eda_report(data):
    """
    Generate a basic EDA report.
    """

    report = {
        "shape": data.shape,
        "target_distribution": get_target_distribution(data),
        "target_percentage": get_target_percentage(data),
        "numeric_summary": get_numeric_summary(data),
        "missing_values": get_missing_values(data),
        "target_type": type_of_target(data[TARGET_COLUMN]),
    }

    return report


if __name__ == "__main__":

    from src.data_loader import load_data

    df = load_data()

    report = generate_eda_report(df)

    for name, result in report.items():

        print(f"\n{name}")
        print(result)

    plot_target_distribution(df)
    plot_amount_distribution(df)
    plot_correlation_matrix(df)