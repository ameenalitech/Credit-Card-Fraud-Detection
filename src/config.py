from pathlib import Path


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJECT_ROOT / "models"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"


# =========================================================
# DATA FILES
# =========================================================

RAW_DATA_PATH = RAW_DATA_DIR / "creditcard.csv"

CLEANED_DATA_PATH = PROCESSED_DATA_DIR / "cleaned_data.csv"

TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.csv"


# =========================================================
# MODEL FILES
# =========================================================

BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"

TUNED_MODEL_PATH = MODELS_DIR / "best_model_tuned.joblib"

THRESHOLD_PATH = MODELS_DIR / "threshold.json"

MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

BASELINE_COMPARISON_PATH = MODELS_DIR / "baseline_comparison.csv"


# =========================================================
# TARGET CONFIGURATION
# =========================================================

TARGET_COLUMN = "Class"

RANDOM_STATE = 42

TEST_SIZE = 0.20

# =========================================================
# STREAMLIT CONFIGURATION
# =========================================================

APP_TITLE = "Credit Card Fraud Detection"

APP_ICON = "💳"

APP_LAYOUT = "wide"

MAX_UPLOAD_SIZE_MB = 50


# =========================================================
# DATASET CONFIGURATION
# =========================================================

EXPECTED_COLUMNS = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount", "Class"]
)

FEATURE_COLUMNS = [
    column
    for column in EXPECTED_COLUMNS
    if column != TARGET_COLUMN
]


# =========================================================
# MODEL CONFIGURATION
# =========================================================

DEFAULT_MODEL = "random_forest"

DEFAULT_THRESHOLD = 0.50


# =========================================================
# DIRECTORY CREATION
# =========================================================

def create_project_directories():
    """
    Create the directories required by the project.
    """

    directories = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        EXTERNAL_DATA_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        METRICS_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    create_project_directories()
    print("Project directories created successfully.")