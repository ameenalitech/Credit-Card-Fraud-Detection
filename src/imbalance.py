from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


def create_sampler(
    method="smote",
    random_state=42
):
    """
    Create a sampling strategy.

    Supported methods:
    - smote
    - random_over
    - random_under
    - none
    """

    method = method.lower()

    if method == "smote":

        return SMOTE(
            random_state=random_state,
            sampling_strategy="auto"
        )

    elif method == "random_over":

        return RandomOverSampler(
            random_state=random_state
        )

    elif method == "random_under":

        return RandomUnderSampler(
            random_state=random_state
        )

    elif method == "none":

        return "passthrough"

    else:

        raise ValueError(
            "Unsupported sampling method. "
            "Choose smote, random_over, random_under, or none."
        )


if __name__ == "__main__":

    from collections import Counter

    from sklearn.model_selection import train_test_split

    from src.data_loader import load_data

    df = load_data()

    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42
    )

    sampler = create_sampler("smote")

    X_resampled, y_resampled = sampler.fit_resample(
        X_train,
        y_train
    )

    print("Before SMOTE:", Counter(y_train))
    print("After SMOTE:", Counter(y_resampled))