import pandas as pd
import streamlit as st
import mlflow
from pycaret.regression import RegressionExperiment

st.title("AutoTreinamento no MLFlow")

mlflow.set_tracking_uri(uri='http://localhost:8222')

uploaded_file = st.file_uploader("Escolha seu dataset csv", type="csv")
target_feature = None
clicked = False

def register_best_model():
    print()


def auto_train():
    pycaret_experiment = RegressionExperiment()
    pycaret_experiment.setup(data=dataset_streamlit, target=target_feature, log_experiment=True)
    best_model = pycaret_experiment.compare_models()
    st.success("Treinamento concluído e modelos logados no MLflow!")
    st.write("Melhor modelo encontrado:")
    st.write(best_model)

if uploaded_file is not None:
    try:
        dataset_streamlit = pd.read_csv(uploaded_file)
        st.dataframe(dataset_streamlit.head()) 
        target_feature = st.selectbox(label="Selecione o atributo meta", options=dataset_streamlit.columns)
        clicked = st.button("Treinar")
        if clicked and target_feature is not None:
            auto_train()

    except Exception as e:
        st.error(f"Ocorreu um erro ao ler o arquivo: {e}")
else:
    st.info("Por favor, faça o upload de um arquivo CSV para continuar.")