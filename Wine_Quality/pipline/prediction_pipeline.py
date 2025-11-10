import os
import sys
import numpy as np
import pandas as pd

from wine_quality.entity.config_entity import wine_PredictorConfig
from wine_quality.entity.S3_estimator import WineEstimator
from wine_quality.exception import custom_Exception
from wine_quality.logger import logging
from pandas import DataFrame


class WineData:
    def __init__(self,
                 fixed_acidity,
                 volatile_acidity,
                 citric_acid,
                 residual_sugar,
                 chlorides,
                 free_sulfur_dioxide,
                 total_sulfur_dioxide,
                 density,
                 pH,
                 sulphates,
                 alcohol):
        """
        WineData constructor
        Input: all features used by the trained wine model for prediction
        """
        try:
            self.fixed_acidity = fixed_acidity
            self.volatile_acidity = volatile_acidity
            self.citric_acid = citric_acid
            self.residual_sugar = residual_sugar
            self.chlorides = chlorides
            self.free_sulfur_dioxide = free_sulfur_dioxide
            self.total_sulfur_dioxide = total_sulfur_dioxide
            self.density = density
            self.pH = pH
            self.sulphates = sulphates
            self.alcohol = alcohol

        except Exception as e:
            raise custom_Exception(e, sys) from e

    def get_wine_input_data_frame(self) -> DataFrame:
        """
        Returns a DataFrame from WineData class input
        """
        try:
            wine_input_dict = self.get_wine_data_as_dict()
            return DataFrame(wine_input_dict)
        except Exception as e:
            raise custom_Exception(e, sys) from e

    def get_wine_data_as_dict(self):
        """
        Returns a dictionary from WineData class input
        """
        logging.info("Entered get_wine_data_as_dict method of WineData class")

        try:
            input_data = {
                "fixed acidity": [self.fixed_acidity],
                "volatile acidity": [self.volatile_acidity],
                "citric acid": [self.citric_acid],
                "residual sugar": [self.residual_sugar],
                "chlorides": [self.chlorides],
                "free sulfur dioxide": [self.free_sulfur_dioxide],
                "total sulfur dioxide": [self.total_sulfur_dioxide],
                "density": [self.density],
                "pH": [self.pH],
                "sulphates": [self.sulphates],
                "alcohol": [self.alcohol]
            }

            logging.info("Created wine data dictionary successfully")
            return input_data

        except Exception as e:
            raise custom_Exception(e, sys) from e


class WineRegressor:
    def __init__(self, prediction_pipeline_config: wine_PredictorConfig = wine_PredictorConfig()) -> None:
        """
        :param prediction_pipeline_config: Configuration for wine prediction
        """
        try:
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise custom_Exception(e, sys)

    def predict(self, dataframe) -> str:
        """
        Loads the model from S3 and returns prediction
        """
        try:
            logging.info("Entered predict method of WineRegressor class")

            model = WineEstimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )

            result = model.predict(dataframe)
            return result

        except Exception as e:
            raise custom_Exception(e, sys)
