
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Supply Chain Risk Prediction",
    page_icon="📦",
    layout="wide"
)

BASE = Path(__file__).parent

@st.cache_resource
def load_models():
    classifier = joblib.load(BASE / "classification_model.joblib")
    regressor = joblib.load(BASE / "regression_model.joblib")

    with open(BASE / "model_metadata.json", "r") as file:
        metadata = json.load(file)

    return classifier, regressor, metadata


st.title("📦 Supply Chain Risk Prediction")
st.write(
    "Predict late-delivery risk and estimate shipping duration "
    "using Machine Learning."
)

try:
    classifier, regressor, metadata = load_models()
except Exception as error:
    st.error(f"Could not load model files: {error}")
    st.stop()

categorical = metadata["categorical_features"]
numerical = metadata["numeric_features"]
feature_columns = metadata["feature_columns"]

st.header("Enter Order Details")

inputs = {}

with st.form("prediction_form"):

    st.subheader("Categorical Information")

    col1, col2 = st.columns(2)

    for i, feature in enumerate(categorical):
        options = metadata["category_options"].get(feature, [])

        with (col1 if i % 2 == 0 else col2):
            if options:
                inputs[feature] = st.selectbox(
                    feature, options
                )
            else:
                inputs[feature] = st.text_input(feature)

    st.subheader("Numerical Information")

    col1, col2 = st.columns(2)

    for i, feature in enumerate(numerical):
        default = float(
            metadata["numeric_defaults"].get(feature, 0)
        )

        with (col1 if i % 2 == 0 else col2):
            inputs[feature] = st.number_input(
                feature,
                value=default
            )

    submitted = st.form_submit_button(
        "Predict Delivery Risk",
        type="primary"
    )


if submitted:

    input_df = pd.DataFrame(
        [inputs],
        columns=feature_columns
    )

    try:
        risk = int(classifier.predict(input_df)[0])
        shipping_days = float(regressor.predict(input_df)[0])

        st.divider()
        st.header("Prediction Results")

        result1, result2 = st.columns(2)

        with result1:
            if risk == 1:
                st.error("HIGH LATE-DELIVERY RISK")
            else:
                st.success("LOW LATE-DELIVERY RISK")

            if hasattr(classifier, "predict_proba"):
                probability = classifier.predict_proba(
                    input_df
                )[0][1]

                st.metric(
                    "Estimated Late Risk Probability",
                    f"{probability:.2%}"
                )

        with result2:
            st.metric(
                "Estimated Shipping Duration",
                f"{shipping_days:.2f} days"
            )

        st.info(
            "Predictions are estimates based on the trained "
            "dataset and are not guaranteed delivery outcomes."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")


with st.expander("About this Project"):
    st.write(
        "This project uses Machine Learning classification "
        "to predict late-delivery risk and regression to "
        "estimate actual shipping duration."
    )

    st.write(
        "Regression model:",
        metadata.get("best_regression_name", "Saved regression model")
    )

    st.write(
        "Classification model: Tuned Random Forest"
    )

st.divider()
st.caption(
    "Supply Chain Risk Prediction and Analysis | "
    "Machine Learning Academic Project"
)
