import pandas as pd
import streamlit as st
import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from pycaret.regression import RegressionExperiment

mlflow.set_tracking_uri(uri='http://localhost:8222')

st.title("AutoTreinamento no MLFlow")

if 'dataset_streamlit' not in st.session_state:
    st.session_state.dataset_streamlit = None
if 'target_feature' not in st.session_state:
    st.session_state.target_feature = None
if 'training_completed' not in st.session_state:
    st.session_state.training_completed = False
if 'pycaret_exp' not in st.session_state:
    st.session_state.pycaret_exp = None
if 'best_model' not in st.session_state:
    st.session_state.best_model = None

def register_best_model_from_experiment(
    experiment_name: str,
    metric_name: str,
    registered_model_name: str,
    higher_is_better: bool = True,
    tag_or_alias: str = "production"
) -> None:
    client = MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        st.error(f"Erro: Experimento '{experiment_name}' não encontrado.")
        return

    experiment_id = experiment.experiment_id

    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=[f"metrics.{metric_name} DESC" if higher_is_better else f"metrics.{metric_name} ASC"],
        run_view_type=ViewType.ACTIVE_ONLY
    )

    if not runs:
        st.warning(f"Não há runs ativas encontradas para o experimento '{experiment_name}'.")
        return

    best_run = None
    best_metric_value = float('-inf') if higher_is_better else float('inf')

    for run in runs:
        if metric_name in run.data.metrics:
            current_metric_value = run.data.metrics[metric_name]
            if (higher_is_better and current_metric_value > best_metric_value) or \
               (not higher_is_better and current_metric_value < best_metric_value):
                best_metric_value = current_metric_value
                best_run = run

    if best_run is None:
        st.warning(f"Nenhuma run encontrada com a métrica '{metric_name}' no experimento '{experiment_name}'.")
        return

    model_uri = f"runs:/{best_run.info.run_id}/model"

    try:
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=registered_model_name
        )

        client = MlflowClient()
        if hasattr(client, 'set_registered_model_alias'):
            client.set_registered_model_alias(name=registered_model_name, alias=tag_or_alias, version=model_version.version)
            st.info(f"Alias '{tag_or_alias}' atribuído à versão {model_version.version}.")
        else:
            client.set_model_version_tag(name=registered_model_name, version=model_version.version, key="stage", value=tag_or_alias)
            st.info(f"Tag 'stage:{tag_or_alias}' atribuída à versão {model_version.version}.")

    except Exception as e:
        st.error(f"Erro ao registrar modelo ou definir tag/alias: {e}")


uploaded_file = st.file_uploader("Escolha seu dataset csv", type="csv")

if uploaded_file is not None:
    try:
        st.session_state.dataset_streamlit = pd.read_csv(uploaded_file)
        st.dataframe(st.session_state.dataset_streamlit.head())

        st.session_state.target_feature = st.selectbox(
            label="Selecione o atributo meta",
            options=st.session_state.dataset_streamlit.columns,
            key="target_feature_select"
        )

        clicked_train = st.button("Treinar Modelos", key="train_button")

        if clicked_train and st.session_state.target_feature:
            st.info("Iniciando treinamento... Isso pode levar alguns minutos.")
        
            pycaret_experiment = RegressionExperiment()
            with st.spinner("Configurando e comparando modelos..."):
                pycaret_experiment.setup(
                    data=st.session_state.dataset_streamlit,
                    target=st.session_state.target_feature,
                    log_experiment=True,
                    experiment_name='reg-autotrain',
                    use_gpu=True
                )
                st.session_state.pycaret_exp = pycaret_experiment
                st.session_state.best_model = pycaret_experiment.compare_models()
            
            st.success("Treinamento concluído e modelos logados no MLflow!")
            st.write("Melhor modelo encontrado:")
            st.write(st.session_state.best_model)
            st.session_state.training_completed = True

    except Exception as e:
        st.error(f"Ocorreu um erro ao ler o arquivo: {e}")
else:
    st.info("Por favor, faça o upload de um arquivo CSV para continuar.")
    st.session_state.training_completed = False
    st.session_state.dataset_streamlit = None
    st.session_state.target_feature = None
    st.session_state.pycaret_exp = None
    st.session_state.best_model = None

if st.session_state.training_completed and st.session_state.pycaret_exp:
    st.subheader("Registrar Melhor Modelo")
    
    comparison_metric = st.selectbox(
        label="Selecione a métrica de avaliação para registro:",
        options=st.session_state.pycaret_exp.get_metrics(),
        key="comparison_metric_select"
    )

    higher_is_better_checkbox = st.checkbox(
        "Métrica: Um valor mais alto é melhor?",
        key="higher_is_better_checkbox"
    )

    tag_input = st.text_input(
        label="Digite o alias/tag para registro do modelo (ex: 'production', 'staging')",
        value="production",
        key="model_tag_input"
    )

    clicked_register = st.button("Registrar Modelo no MLflow", key="register_button")

    if clicked_register:
        if comparison_metric and tag_input:
            with st.spinner("Registrando o melhor modelo..."):
                register_best_model_from_experiment(
                    experiment_name='reg-autotrain',
                    metric_name=comparison_metric,
                    registered_model_name='autotrainstreamlit',
                    higher_is_better=higher_is_better_checkbox,
                    tag_or_alias=tag_input
                )
        else:
            st.warning("Por favor, preencha todos os campos para registrar o modelo.")