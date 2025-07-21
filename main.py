import pandas as pd
import streamlit as st
import mlflow
from pycaret.regression import RegressionExperiment

mlflow.set_tracking_uri(uri='http://localhost:8222')

st.title("AutoTrain to MLFlow")
uploaded_file = st.file_uploader("Choose a dataset", type="csv")

if uploaded_file is not None:
    try:
        dataset_streamlit = pd.read_csv(uploaded_file)
        st.dataframe(dataset_streamlit.head())
        pycaret_experiment = RegressionExperiment()
        pycaret_experiment.setup(data=dataset_streamlit, target='tmax', log_experiment=True)

        best_regressors = pycaret_experiment.compare_models()
    except Exception as e:
        st.error(f"Ocorreu um erro ao ler o arquivo: {e}")
else:
    st.info("Por favor, faça o upload de um arquivo CSV para continuar.")
