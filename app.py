"""Streamlit web application for automated model training and MLflow registration.

Evaluates the feasibility of integrating PyCaret AutoML with MLflow Tracking
and Model Registry for an automated training architecture.
"""

import pandas as pd
import streamlit as st
from ml_utils import (
    setup_mlflow,
    train_regression_models,
    register_best_model_from_experiment,
    get_available_metrics,
    DEFAULT_EXPERIMENT_NAME,
    DEFAULT_MODEL_NAME,
)

# Page configuration
st.set_page_config(
    page_title="AutoML to MLflow - PoC",
    layout="wide",
)

# Initialize MLflow tracking URI
tracking_uri = setup_mlflow()

# Header and introduction
st.title("AutoML to MLflow Pipeline")
st.markdown(
    """
    This **Proof of Concept (PoC)** demonstrates end-to-end integration between **PyCaret** (AutoML) 
    and **MLflow** (Tracking & Model Registry), validating architectural feasibility for automated 
    model selection and deployment pipelines.
    """
)

st.caption(f"Connected to MLflow Tracking Server at: `{tracking_uri}`")

# Initialize session state keys
if "dataset" not in st.session_state:
    st.session_state.dataset = None
if "target_feature" not in st.session_state:
    st.session_state.target_feature = None
if "training_completed" not in st.session_state:
    st.session_state.training_completed = False
if "pycaret_exp" not in st.session_state:
    st.session_state.pycaret_exp = None
if "best_model" not in st.session_state:
    st.session_state.best_model = None

# Section 1: Dataset Upload
st.subheader("1. Dataset Ingestion")
uploaded_file = st.file_uploader(
    label="Upload your CSV dataset",
    type=["csv"],
    help="Upload a clean CSV file. A sample dataset is available at 'data/boston_weather_data.csv'.",
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state.dataset = df

        st.markdown("**Dataset Preview:**")
        st.dataframe(df.head(10), use_container_width=True)
        st.text(f"Dimensions: {df.shape[0]} rows × {df.shape[1]} columns")

        # Section 2: Model Training Configuration
        st.subheader("2. Automated Training Configuration")
        col1, col2 = st.columns([2, 1])

        with col1:
            st.session_state.target_feature = st.selectbox(
                label="Select the target feature to predict:",
                options=df.columns,
                key="target_feature_select",
            )

        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            clicked_train = st.button(
                "Train & Compare Models",
                key="train_button",
                type="primary",
                use_container_width=True,
            )

        if clicked_train and st.session_state.target_feature:
            with st.spinner("Training models with PyCaret and logging runs to MLflow..."):
                pycaret_exp, best_model = train_regression_models(
                    data=st.session_state.dataset,
                    target_feature=st.session_state.target_feature,
                    experiment_name=DEFAULT_EXPERIMENT_NAME,
                )
                st.session_state.pycaret_exp = pycaret_exp
                st.session_state.best_model = best_model
                st.session_state.training_completed = True

            st.success("Training completed! All candidate models and metrics are logged in MLflow.")
            st.markdown("### Top Performing Model")
            st.code(str(best_model), language="python")

    except Exception as exc:
        st.error(f"Error processing file: {exc}")
else:
    st.info("Please upload a CSV file above to begin. You can find sample data in `data/boston_weather_data.csv`.")
    st.session_state.dataset = None
    st.session_state.target_feature = None
    st.session_state.training_completed = False
    st.session_state.pycaret_exp = None
    st.session_state.best_model = None

# Section 3: Model Registry
if st.session_state.training_completed and st.session_state.pycaret_exp:
    st.divider()
    st.subheader("3. MLflow Model Registry Promotion")
    st.markdown(
        "Promote the best model from the experiment runs into the **MLflow Model Registry** "
        "with an operational tag or alias (e.g., `production`, `staging`)."
    )

    available_metrics = get_available_metrics(
        experiment_name=DEFAULT_EXPERIMENT_NAME,
        pycaret_exp=st.session_state.pycaret_exp,
    )

    col_reg1, col_reg2, col_reg3 = st.columns([2, 1, 2])

    with col_reg1:
        default_index = available_metrics.index("R2") if "R2" in available_metrics else 0
        comparison_metric = st.selectbox(
            label="Evaluation metric for model selection:",
            options=available_metrics,
            index=default_index,
            key="comparison_metric_select",
        )

    with col_reg2:
        default_higher = comparison_metric in ["R2"]
        higher_is_better = st.checkbox(
            "Higher is better?",
            value=default_higher,
            key="higher_is_better_checkbox",
            help="Checked for scores like R2; unchecked for error metrics like RMSE, MAE, MSE.",
        )

    with col_reg3:
        tag_alias = st.text_input(
            label="Stage Tag or Alias:",
            value="production",
            key="model_tag_input",
            help="Alias or stage tag assigned to the registered version (e.g. 'production', 'staging', 'champion').",
        )

    col_btn, _ = st.columns([2, 2])
    with col_btn:
        clicked_register = st.button(
            "Register Model to MLflow",
            key="register_button",
            type="primary",
            use_container_width=True,
        )

    if clicked_register:
        if comparison_metric and tag_alias:
            with st.spinner("Registering model and setting alias in MLflow..."):
                success, message = register_best_model_from_experiment(
                    experiment_name=DEFAULT_EXPERIMENT_NAME,
                    metric_name=comparison_metric,
                    registered_model_name=DEFAULT_MODEL_NAME,
                    higher_is_better=higher_is_better,
                    tag_or_alias=tag_alias,
                )
            if success:
                st.success(message)
                st.info(f"Open your MLflow UI at [{tracking_uri}]({tracking_uri}) to inspect the model and runs.")
            else:
                st.error(message)
        else:
            st.warning("Please fill in all registry configuration fields.")
