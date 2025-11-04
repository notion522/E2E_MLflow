import os
import shutil
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from urllib.parse import urlparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mlProject.entity.config_entity import ModelEvaluationConfig
from mlProject.utils.common import save_json



class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        # ensure 1D arrays
        actual = np.ravel(np.array(actual))
        pred = np.ravel(np.array(pred))
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def log_into_mlflow(self):
        # Load test data and model
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]

        # set tracking URI (not registry URI)
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # start MLflow run
        with mlflow.start_run() as run:
            # predictions & metrics
            predicted_qualities = model.predict(test_x)
            rmse, mae, r2 = self.eval_metrics(test_y, predicted_qualities)

            # Save metrics locally
            scores = {"rmse": rmse, "mae": mae, "r2": r2}
            metric_path = Path(self.config.metric_file_name)
            metric_path.parent.mkdir(parents=True, exist_ok=True)
            save_json(path=metric_path, data=scores)   # your helper

            # Log params & metrics
            if isinstance(self.config.all_params, dict):
                mlflow.log_params(self.config.all_params)
            mlflow.log_metric('rmse', rmse)
            mlflow.log_metric('mae', mae)
            mlflow.log_metric('r2', r2)

            # Save & upload model in a DagsHub-friendly way
            # If the tracking URI contains "dagshub" avoid registered model APIs
            temp_model_dir = Path("temp_saved_model")
            try:
                if "dagshub" in mlflow.get_tracking_uri().lower():
                    # Save model in MLflow format locally then log artifacts
                    if temp_model_dir.exists():
                        shutil.rmtree(temp_model_dir)
                    mlflow.sklearn.save_model(sk_model=model, path=str(temp_model_dir))
                    mlflow.log_artifacts(str(temp_model_dir), artifact_path="model")
                else:
                    # If not DagsHub, try to use registry (if configured)
                    mlflow.sklearn.log_model(model, "model", registered_model_name="ElasticnetModel")
            finally:
                # cleanup temp dir if created
                if temp_model_dir.exists():
                    try:
                        shutil.rmtree(temp_model_dir)
                    except Exception:
                        pass

        # optionally return run id for reference
        return run.info.run_id
