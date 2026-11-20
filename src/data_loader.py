import pandas as pd

from src.config import RAW_DATA_PATH


def load_data(file_path=RAW_DATA_PATH):
    """
    Load the credit card fraud dataset.

    Parameters
    ----------
    file_path : str or Path
        Dataset file location.

    Returns
    -------
    pandas.DataFrame
        Loaded dataset.
    """

    file_path = str(file_path)

    try:
        data = pd.read_csv(file_path)

    except FileNotFoundError:
        raise FileNotFoundError(
            f"Dataset was not found at: {file_path}"
        )

    except pd.errors.EmptyDataError:
        raise ValueError(
            "The dataset file is empty."
        )

    except pd.errors.ParserError as error:
        raise ValueError(
            f"Unable to parse the CSV file: {error}"
        )

    if data.empty:
        raise ValueError("The dataset contains no rows.")

    return data


def get_data_information(data):
    """
    Return basic information about the dataset.
    """

    information = {
        "rows": data.shape[0],
        "columns": data.shape[1],
        "column_names": data.columns.tolist(),
        "missing_values": data.isnull().sum().to_dict(),
        "duplicate_rows": int(data.duplicated().sum()),
        "memory_usage_mb": round(
            data.memory_usage(deep=True).sum() / (1024 ** 2),
            2
        ),
    }

    return information


if __name__ == "__main__":

    df = load_data()

    info = get_data_information(df)

    print("Dataset loaded successfully.")
    print(info)