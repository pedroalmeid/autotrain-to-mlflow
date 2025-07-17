import pandas as pd
import mlflow

from pycaret.regression import RegressionExperiment

mlflow.set_tracking_uri(uri='http://localhost:8222')

dataset = pd.read_csv('boston_weather_data.csv')

pycaret_experiment = RegressionExperiment()
pycaret_experiment.setup(data=dataset, target='tmax', log_experiment=True)

best_regressors = pycaret_experiment.compare_models()