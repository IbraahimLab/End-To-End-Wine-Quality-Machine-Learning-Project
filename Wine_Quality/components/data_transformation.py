import sys
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer
)
from sklearn.compose import ColumnTransformer

from wine_quality.constants import TARGET_COLUMN, SCHEMA_FILE_PATH
from wine_quality.entity.config_entity import DataTransformationConfig
from wine_quality.entity.artifact_entity import (
    DataTransformationArtifact,
    DataIngestionArtifact,
    DataValidationArtifact,
)
from wine_quality.exception import custom_Exception
from wine_quality.logger import logging
from wine_quality.utils.main_utils import (
    save_object,
    save_numpy_array_data,
    read_yaml_file,
    drop_columns,
)
# from wine_quality.entity.estimator import TargetValueMapping  # Uncomment if you have string labels


class DataTransformation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_transformation_config: DataTransformationConfig,
        data_validation_artifact: DataValidationArtifact,
    ):
        """
        :param data_ingestion_artifact: paths for train/test data
        :param data_transformation_config: where to save transformed files and preprocessor
        :param data_validation_artifact: results from validation stage
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise custom_Exception(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        """Reads CSV into pandas DataFrame."""
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise custom_Exception(e, sys)

    def get_data_transformer_object(self) -> ColumnTransformer:
        """
        Creates preprocessing object dynamically using schema.yaml.
        Handles numerical, ordinal, and one-hot columns in one unified transformer.
        """
        logging.info("Entered get_data_transformer_object method")

        try:
            oh_columns = self._schema_config.get("oh_columns", [])
            or_columns = self._schema_config.get("or_columns", [])
            transform_columns = self._schema_config.get("transform_columns", [])
            num_features = self._schema_config.get("num_features", [])

            # Pipelines for different column groups
            numeric_pipeline = Pipeline(
                steps=[
                    ("power", PowerTransformer(method="yeo-johnson")),
                    ("scaler", StandardScaler())
                ]
            )

            one_hot_pipeline = Pipeline(
                steps=[
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
                ]
            )

            ordinal_pipeline = Pipeline(
                steps=[
                    ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
                ]
            )

            transform_pipeline = Pipeline(
                steps=[
                    ("transform", PowerTransformer(method="yeo-johnson"))
                ]
            )

            # Combine all in one ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", numeric_pipeline, num_features),
                    ("one_hot_pipeline", one_hot_pipeline, oh_columns),
                    ("ordinal_pipeline", ordinal_pipeline, or_columns),
                    ("transform_pipeline", transform_pipeline, transform_columns),
                ],
                remainder="passthrough"
            )

            logging.info("Preprocessor created successfully.")
            return preprocessor

        except Exception as e:
            raise custom_Exception(e, sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Executes transformation pipeline: preprocesses features, balances data,
        and saves all outputs for model training.
        """
        try:
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            logging.info("Starting data transformation...")

            preprocessor = self.get_data_transformer_object()

            # Load data
            train_df = self.read_data(self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)

            # Split input and target
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]

            # Drop columns specified in schema
            drop_cols = self._schema_config.get("drop_columns", [])
            input_feature_train_df = drop_columns(df=input_feature_train_df, cols=drop_cols)
            input_feature_test_df = drop_columns(df=input_feature_test_df, cols=drop_cols)

            # Optional target mapping (uncomment if your target is string)
            # target_feature_train_df = target_feature_train_df.replace(TargetValueMapping()._asdict())
            # target_feature_test_df = target_feature_test_df.replace(TargetValueMapping()._asdict())

            # Transform data
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            logging.info("Applied preprocessing transformations successfully.")

            # Balance dataset with SMOTEENN
            
            # smt = SMOTEENN(sampling_strategy="minority", smote=SMOTE(k_neighbors=1))

            

            # smt = SMOTEENN(
            # sampling_strategy="minority",
            # smote=SMOTE(k_neighbors=1),
            # enn=EditedNearestNeighbours(n_neighbors=1)
            #                 )

            smt = SMOTEENN(
            sampling_strategy="minority",
            smote=SMOTE(k_neighbors=1),
            enn=EditedNearestNeighbours(n_neighbors=1)
                )




            input_feature_train_final, target_feature_train_final = smt.fit_resample(
             input_feature_train_arr, target_feature_train_df
            )
            input_feature_test_final, target_feature_test_final = smt.fit_resample(
            input_feature_test_arr, target_feature_test_df
            )

            logging.info("Applied SMOTEENN to handle class imbalance.")

            # Combine processed features with targets
            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
            test_arr = np.c_[input_feature_test_final, np.array(target_feature_test_final)]

            # Save artifacts
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)

            logging.info("Saved preprocessor object and transformed arrays.")

            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )

            logging.info("Data transformation complete.")
            return data_transformation_artifact

        except Exception as e:
            raise custom_Exception(e, sys) from e
