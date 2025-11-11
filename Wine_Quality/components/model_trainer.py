import sys, os
sys.path.append(os.getcwd())
import sys
from typing import Tuple

import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import mlflow
import mlflow.sklearn

from wine_quality.utils.model_factory import ModelFactory
from wine_quality.exception import custom_Exception
from wine_quality.logger import logging
from wine_quality.utils.main_utils import load_numpy_array_data, load_object, save_object
from wine_quality.entity.config_entity import ModelTrainerConfig
from wine_quality.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    RegressionMetricArtifact,
)
from wine_quality.entity.estimator import combined_Model_preproccessing


class ModelTrainer:
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig,
    ):
        """
        ModelTrainer uses transformed data and model config to train
        the best regression model defined in model.yaml.
        """
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def get_model_object_and_report(
        self, train: np.ndarray, test: np.ndarray
    ) -> Tuple[dict, RegressionMetricArtifact]:
        """
        Run model selection & hyperparameter tuning using ModelFactory + GridSearchCV.
        Returns best model detail dict and regression metric artifact.
        """
        try:
            logging.info("Starting model selection using ModelFactory (GridSearchCV).")

            model_factory = ModelFactory(
                model_config_path=self.model_trainer_config.model_config_file_path
            )

            x_train, y_train = train[:, :-1], train[:, -1]
            x_test, y_test = test[:, :-1], test[:, -1]

            best_model_detail = model_factory.get_best_model(
                X=x_train,
                y=y_train,
                base_score=self.model_trainer_config.expected_accuracy,
            )

            best_model = best_model_detail["best_model"]

            #MLFlow logging
            with mlflow.start_run(run_name="wine_quality_training"):
                  mlflow.log_param("model_name", best_model_detail["best_model_name"])
                  mlflow.log_param("hyperparameters", best_model_detail["best_params"])

            if best_model is None:
                raise custom_Exception(
                    "No suitable model found above base score.", sys
                )

            y_pred = best_model.predict(x_test)

            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)

            metric_artifact = RegressionMetricArtifact(
                r2_score=r2,
                mae=mae,
                mse=mse,
            )


            # MLFlow logging Metric

            mlflow.log_metric("r2_score", metric_artifact.r2_score)
            mlflow.log_metric("mae", metric_artifact.mae)
            mlflow.log_metric("mse", metric_artifact.mse)


            logging.info(
                f"Best Model: {best_model_detail['best_model_name']}, "
                f"R2: {r2:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}"
            )
            mlflow.sklearn.log_model(best_model_detail["best_model"], "model")

            return best_model_detail, metric_artifact
        
           


        except Exception as e:
            raise custom_Exception(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Orchestrates loading data, training best model, wrapping with preprocessor,
        saving final model, and returning ModelTrainerArtifact.
        """
        try:
            logging.info("Entered initiate_model_trainer method of ModelTrainer class")

            train_arr = load_numpy_array_data(
                file_path=self.data_transformation_artifact.transformed_train_file_path
            )
            test_arr = load_numpy_array_data(
                file_path=self.data_transformation_artifact.transformed_test_file_path
            )

            best_model_detail, metric_artifact = self.get_model_object_and_report(
                train=train_arr, test=test_arr
            )

            if best_model_detail["best_score"] < self.model_trainer_config.expected_accuracy:
                logging.info(
                    "No best model found with score higher than expected_accuracy."
                )
                raise custom_Exception(
                    "Best model score below expected_accuracy threshold.", sys
                )

            preprocessing_obj = load_object(
                file_path=self.data_transformation_artifact.transformed_object_file_path
            )

            final_model = combined_Model_preproccessing(
                preprocessing_object=preprocessing_obj,
                trained_model_object=best_model_detail["best_model"],
            )

            logging.info("Created combined model (preprocessor + regressor).")

            save_object(
                self.model_trainer_config.trained_model_file_path,
                final_model,
            )

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact,
            )

            logging.info(f"Model trainer artifact: {model_trainer_artifact}")

            return model_trainer_artifact

        except Exception as e:
            raise custom_Exception(e, sys) from e
