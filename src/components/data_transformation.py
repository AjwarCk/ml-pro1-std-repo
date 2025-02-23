import sys  # Import sys for system-specific parameters and functions
from dataclasses import dataclass  # Import dataclass to simplify configuration class creation

import numpy as np  # Import numpy for numerical operations and array handling
import pandas as pd  # Import pandas for data manipulation and analysis
from sklearn.compose import ColumnTransformer  # Import ColumnTransformer to apply different transformations to columns
from sklearn.impute import SimpleImputer  # Import SimpleImputer to fill in missing values
from sklearn.pipeline import Pipeline  # Import Pipeline to chain processing steps
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # Import OneHotEncoder for categorical encoding and StandardScaler for scaling numerical data

from src.exception import CustomException  # Import custom exception class for error handling
from src.logger import logging  # Import logging for logging messages during processing
import os  # Import os for interacting with the operating system (e.g., file paths)

from src.utils import save_object  # Import a utility function to save objects (e.g., via pickle)

@dataclass  # Use dataclass to automatically generate class methods like __init__
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")  # Set file path for saving the preprocessor object

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()  # Initialize the data transformation configuration

    def get_data_transformer_object(self):
        """
        This function is responsible for creating and returning the preprocessor object,
        which applies transformations to both numerical and categorical data.
        """
        try:
            numerical_columns = ["writing_score", "reading_score"]  # Define numerical columns to transform
            categorical_columns = [  # Define categorical columns to transform
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            num_pipeline = Pipeline(  # Create a pipeline for processing numerical data
                steps=[
                    ("imputer", SimpleImputer(strategy='median')),  # Impute missing values using the median
                    ("scaler", StandardScaler())  # Scale numerical features with StandardScaler
                ]
            )

            cat_pipeline = Pipeline(  # Create a pipeline for processing categorical data
                steps=[
                    ("imputer", SimpleImputer(strategy='most_frequent')),  # Impute missing values using the most frequent value (note: "most_freequent" is likely a typo for "most_frequent")
                    ("one_hot_encoder", OneHotEncoder()),  # Encode categorical features using OneHotEncoder
                    ("scaler", StandardScaler(with_mean=False))  # Scale encoded features using StandardScaler without centering
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")  # Log the list of categorical columns
            logging.info(f"Numerical columns: {numerical_columns}")  # Log the list of numerical columns

            preprocessor = ColumnTransformer(  # Combine the numerical and categorical pipelines into one preprocessor object
                [
                    ("num_pipeline", num_pipeline, numerical_columns),  # Apply numerical pipeline to numerical columns
                    ("cat_pipeline", cat_pipeline, categorical_columns)  # Apply categorical pipeline to categorical columns
                ]
            )

            return preprocessor  # Return the constructed preprocessor object

        except Exception as e:
            raise CustomException(e, sys)  # Raise a custom exception if any error occurs

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)  # Read the training data CSV into a pandas DataFrame
            test_df = pd.read_csv(test_path)  # Read the testing data CSV into a pandas DataFrame

            logging.info("Read train and test data completed")  # Log that both train and test data have been read

            logging.info("Obtaining preprocessing object")  # Log that the preprocessor object is being obtained
            preprocessing_obj = self.get_data_transformer_object()  # Get the preprocessor object using the method above

            target_column_name = "math_score"  # Define the target column for prediction
            numerical_columns = ["writing_score", "reading_score"]  # Define numerical columns (can be used later if needed)

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)  # Drop target column from training data to get input features
            target_feature_train_df = train_df[target_column_name]  # Extract the target feature from training data

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)  # Drop target column from testing data to get input features
            target_feature_test_df = test_df[target_column_name]  # Extract the target feature from testing data

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")  # Log that preprocessing is being applied

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)  # Fit and transform training input features
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)  # Transform testing input features using the fitted preprocessor

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]  # Concatenate processed training features and target into a single array
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]  # Concatenate processed testing features and target into a single array

            logging.info(f"Saved preprocessing object.")  # Log that the preprocessor object will be saved

            save_object(  # Save the preprocessor object to the specified file path for future use
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (  # Return the training array, testing array, and the file path where the preprocessor object is saved
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)  # Raise a custom exception if an error occurs during transformation