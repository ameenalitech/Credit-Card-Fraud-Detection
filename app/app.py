
"""
Credit Card Fraud Detection
Machine Learning Streamlit Dashboard
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# ==========================================
# Project Configuration
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.config import (
    FEATURE_COLUMNS,
    THRESHOLD_PATH,
    TUNED_MODEL_PATH,
)
from src.predict import (
    load_saved_threshold,
    load_trained_model,
    predict_transactions,
)

# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
)


# ==========================================
# Custom Styling
# ==========================================

st.markdown("""
<style>
    .main {
        background-color: #f8fafc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    h1 {
        color: #172554;
    }

    h2, h3 {
        color: #1e293b;
    }

    [data-testid="stMetric"] {
        background-color: white;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# Header
# ==========================================

st.title("💳 Credit Card Fraud Detection")

st.markdown(
    """
    **Machine Learning Based Fraud Detection System**

    This application analyzes credit card transactions
    and identifies potentially fraudulent activities
    using a trained Machine Learning model.
    """
)

st.divider()


# ==========================================
# Model Loading
# ==========================================

@st.cache_resource
def load_model():

    if not TUNED_MODEL_PATH.exists():
        raise FileNotFoundError("Trained model file not found.")

    return load_trained_model(TUNED_MODEL_PATH)



@st.cache_data
def load_threshold():

    return load_saved_threshold(
        threshold_path=THRESHOLD_PATH,
        default_threshold=0.50
    )


try:

    model = load_model()
    threshold = load_threshold()

    st.success("Model loaded successfully.")

    col1, col2 = st.columns(2)

    col1.metric(
        "Decision Threshold",
        f"{threshold:.4f}"
    )

    col2.metric(
        "Required Features",
        len(FEATURE_COLUMNS)
    )

except Exception as error:

    st.error(f"Model loading failed: {error}")
    st.stop()


st.divider()


# ==========================================
# Project Information
# ==========================================

with st.expander("About This Project", expanded=False):

    st.markdown("""
    ### Project Overview

    The system uses Machine Learning to detect
    fraudulent credit card transactions.

    **How does it work?**

    1. Upload a transaction dataset in CSV format.
    2. The application validates the input features.
    3. The trained model analyzes each transaction.
    4. The model classifies transactions as Fraud or Normal.
    5. The application displays statistics and prediction results.

    **Important Concepts**

    - **Fraud:** A potentially fraudulent transaction.
    - **Normal:** A transaction classified as legitimate.
    - **Threshold:** The probability cutoff used to classify transactions.
    - **Machine Learning Model:** The trained algorithm used for prediction.

    Note: Predictions are model estimates and should not be
    treated as confirmed fraud without further investigation.
    """)


# ==========================================
# Dataset Upload
# ==========================================

st.subheader("📂 Upload Transaction Dataset")

uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=["csv"],
    help="The dataset must contain all required model features."
)


if uploaded_file is None:

    st.info("Upload a CSV file to start fraud detection.")

    st.stop()


# ==========================================
# Read and Validate Dataset
# ==========================================

try:

    input_data = pd.read_csv(uploaded_file)

    if input_data.empty:
        st.error("The uploaded dataset is empty.")
        st.stop()

    if not input_data.columns.is_unique:
        st.error("The dataset contains duplicate column names.")
        st.stop()

    if len(input_data) > 1_000_000:
        st.error("Dataset is too large. Please upload at most 1 million rows.")
        st.stop()

    missing_columns = (
        set(FEATURE_COLUMNS) - set(input_data.columns)
    )

    if missing_columns:

        st.error(
            "Missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

        st.stop()

    if not input_data.loc[:, FEATURE_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    ).notna().all().all():

        st.error(
            "Required features contain missing or non-numeric values. "
            "Please clean your dataset and try again."
        )

        st.stop()

except Exception as error:

    st.error(f"Unable to read dataset: {error}")
    st.stop()


# ==========================================
# Dataset Overview
# ==========================================

st.subheader("📊 Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Transactions",
    f"{len(input_data):,}"
)

col2.metric(
    "Total Features",
    len(FEATURE_COLUMNS)
)

col3.metric(
    "Missing Values",
    int(input_data.loc[:, FEATURE_COLUMNS].isna().sum().sum())
)

with st.expander("Preview Uploaded Data"):

    st.dataframe(
        input_data.head(10),
        use_container_width=True
    )


st.divider()


# ==========================================
# Fraud Detection
# ==========================================

st.subheader("🔍 Fraud Detection")

st.write(
    "Click the button below to analyze transactions "
    "using the trained Machine Learning model."
)

if st.button(
    "Run Fraud Detection",
    type="primary",
    use_container_width=True
):

    try:

        with st.spinner("Analyzing transactions..."):

            predictions = predict_transactions(
                input_data,
                model=model,
                threshold=threshold
            )

        # Validate prediction output

        required_output_columns = {
            "predicted_class"
        }

        if not required_output_columns.issubset(predictions.columns):
            raise ValueError(
                "Prediction output must contain predicted_class."
            )

        if len(predictions) != len(input_data):
            raise ValueError(
                "Prediction output row count does not match input data."
            )

        if predictions["predicted_class"].isna().any():
            raise ValueError(
                "Predictions contain missing classification values."
            )

        if not predictions["predicted_class"].isin([0, 1]).all():
            raise ValueError(
                "Predicted classes must be either 0 (Normal) or 1 (Fraud)."
            )

        # Save results in session state

        st.session_state["predictions"] = predictions.copy()

        st.success("Transaction analysis completed successfully!")

    except Exception as error:

        st.error(f"Prediction failed: {error}")


# ==========================================
# Display Prediction Results
# ==========================================

if "predictions" in st.session_state:

    predictions = st.session_state["predictions"]

    total_transactions = len(predictions)

    fraud_count = int(
        (predictions["predicted_class"] == 1).sum()
    )

    normal_count = int(
        (predictions["predicted_class"] == 0).sum()
    )

    fraud_rate = (
        fraud_count / total_transactions * 100
        if total_transactions > 0 else 0
    )

    st.divider()

    st.subheader("📈 Fraud Detection Dashboard")

    # Metrics

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Transactions",
        f"{total_transactions:,}"
    )

    col2.metric(
        "Fraudulent Transactions",
        f"{fraud_count:,}"
    )

    col3.metric(
        "Normal Transactions",
        f"{normal_count:,}"
    )

    col4.metric(
        "Fraud Detection Rate",
        f"{fraud_rate:.2f}%"
    )

    # Classification Chart

    st.subheader("Transaction Classification")

    chart_data = pd.DataFrame({
        "Transaction Type": ["Fraud", "Normal"],
        "Count": [fraud_count, normal_count]
    })

    fig = px.pie(
        chart_data,
        names="Transaction Type",
        values="Count",
        color="Transaction Type",
        color_discrete_map={
            "Fraud": "#ef4444",
            "Normal": "#22c55e"
        },
        hole=0.55,
    )

    fig.update_layout(
        showlegend=True,
        margin=dict(t=20, b=20, l=20, r=20)
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # Prediction Table

    st.subheader("📋 Prediction Results")

    if "predicted_probability" in predictions.columns:

        st.caption(
            "Predicted probability represents the model's estimated "
            "fraud probability, not a confirmed fraud."
        )

    fraud_only = st.checkbox(
        "Show fraudulent transactions only"
    )

    if fraud_only:

        displayed_results = predictions[
            predictions["predicted_class"] == 1
        ]

    else:

        displayed_results = predictions

    st.dataframe(
        displayed_results,
        use_container_width=True,
        hide_index=True
    )

    # Download Results

    csv_data = predictions.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        label="⬇️ Download Prediction Results",
        data=csv_data,
        file_name="fraud_predictions.csv",
        mime="text/csv",
        use_container_width=True
    )


# ==========================================
# Footer
# ==========================================

st.divider()

st.caption(
    "Credit Card Fraud Detection | "
    "Machine Learning Project"
)
