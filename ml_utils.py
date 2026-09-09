"""Machine Learning and MLflow utility module.

Encapsulates PyCaret AutoML regression pipelines and MLflow Tracking
and Model Registry interactions.
"""

import os
from typing import Any, Tuple, List, Optional
import pandas as pd
import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from pycaret.regression import RegressionExperiment

# Default configurations (can be overridden via environment variables)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:8222")
USE_GPU = os.getenv("USE_GPU", "false").lower() in ("true", "1", "yes")
DEFAULT_EXPERIMENT_NAME = os.getenv("DEFAULT_EXPERIMENT_NAME", "reg-autotrain")
DEFAULT_MODEL_NAME = os.getenv("DEFAULT_MODEL_NAME", "autotrainstreamlit")


def setup_mlflow(tracking_uri: Optional[str] = None) -> str:
    """Configure MLflow tracking URI."""
    uri = tracking_uri or MLFLOW_TRACKING_URI
    mlflow.set_tracking_uri(uri=uri)
    return uri


def train_regression_models(
    data: pd.DataFrame,
    target_feature: str,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    use_gpu: bool = USE_GPU,
) -> Tuple[RegressionExperiment, Any]:
    """Configure and execute PyCaret AutoML comparison for regression.

    Args:
        data: Input DataFrame.
        target_feature: Column name to predict.
        experiment_name: MLflow experiment name for logging.
        use_gpu: Whether to use GPU acceleration (requires CUDA/cuML).

    Returns:
        Tuple containing the PyCaret RegressionExperiment instance and the best model.
    """
    setup_mlflow()
    experiment = RegressionExperiment()
    experiment.setup(
        data=data,
        target=target_feature,
        log_experiment=True,
        experiment_name=experiment_name,
        use_gpu=use_gpu,
        verbose=False,
    )
    best_model = experiment.compare_models()
    return experiment, best_model


def get_available_metrics(
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    pycaret_exp: Optional[RegressionExperiment] = None,
) -> List[str]:
    """Retrieve metric names available in experiment runs, with fallback to PyCaret metrics."""
    try:
        client = MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                run_view_type=ViewType.ACTIVE_ONLY,
                max_results=50,
            )
            metrics_set = set()
            for run in runs:
                metrics_set.update(run.data.metrics.keys())
            if metrics_set:
                # Prioritize standard regression metrics first
                standard_order = ["R2", "RMSE", "MAE", "MSE", "RMSLE", "MAPE"]
                ordered = [m for m in standard_order if m in metrics_set]
                others = sorted([m for m in metrics_set if m not in standard_order])
                return ordered + others
    except Exception:
        pass

    # Fallback to standard PyCaret metrics list if query fails
    return ["R2", "MAE", "RMSE", "MSE", "RMSLE", "MAPE"]


def register_best_model_from_experiment(
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    metric_name: str = "R2",
    registered_model_name: str = DEFAULT_MODEL_NAME,
    higher_is_better: bool = True,
    tag_or_alias: str = "production",
) -> Tuple[bool, str]:
    """Identify the top-performing run in an experiment and register it in the Model Registry.

    Args:
        experiment_name: Name of the MLflow experiment.
        metric_name: Metric key used to rank models.
        registered_model_name: Target model name in MLflow Model Registry.
        higher_is_better: True if larger metric values indicate better performance.
        tag_or_alias: Alias or stage tag to assign (e.g. 'production', 'staging').

    Returns:
        Tuple of (success_status, message).
    """
    setup_mlflow()
    client = MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return False, f"Experiment '{experiment_name}' not found in MLflow."

    experiment_id = experiment.experiment_id

    order_direction = "DESC" if higher_is_better else "ASC"
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=[f"metrics.{metric_name} {order_direction}"],
        run_view_type=ViewType.ACTIVE_ONLY,
    )

    if not runs:
        return False, f"No active runs found for experiment '{experiment_name}'."

    best_run = None
    best_metric_value = float("-inf") if higher_is_better else float("inf")

    for run in runs:
        if metric_name in run.data.metrics:
            current_val = run.data.metrics[metric_name]
            if (higher_is_better and current_val > best_metric_value) or (
                not higher_is_better and current_val < best_metric_value
            ):
                best_metric_value = current_val
                best_run = run

    if best_run is None:
        return False, f"No runs found containing metric '{metric_name}' in experiment '{experiment_name}'."

    model_uri = f"runs:/{best_run.info.run_id}/model"

    try:
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=registered_model_name,
        )

        # Assign alias or fallback to tag
        if hasattr(client, "set_registered_model_alias"):
            client.set_registered_model_alias(
                name=registered_model_name,
                alias=tag_or_alias,
                version=model_version.version,
            )
            msg = (
                f"Model '{registered_model_name}' version {model_version.version} successfully registered! "
                f"Alias '{tag_or_alias}' assigned (Best {metric_name}: {best_metric_value:.4f})."
            )
        else:
            client.set_model_version_tag(
                name=registered_model_name,
                version=model_version.version,
                key="stage",
                value=tag_or_alias,
            )
            msg = (
                f"Model '{registered_model_name}' version {model_version.version} successfully registered! "
                f"Tag 'stage:{tag_or_alias}' assigned (Best {metric_name}: {best_metric_value:.4f})."
            )

        return True, msg

    except Exception as exc:
        return False, f"Error registering model or assigning tag/alias: {exc}"

