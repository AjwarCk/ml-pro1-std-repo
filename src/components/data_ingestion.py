import os  # Import os module for operating system related operations (e.g., file paths)
import sys  # Import sys module to interact with the Python interpreter and system-specific parameters
from src.exception import CustomException  # Import a custom exception class for handling errors in our project
from src.logger import logging  # Import our logging utility for logging messages (info, error, etc.)
import pandas as pd  # Import pandas library as pd for data manipulation and analysis

from sklearn.model_selection import train_test_split  # Import function to split datasets into training and testing subsets
from dataclasses import dataclass  # Import dataclass decorator to easily create classes for storing configuration data

from src.components.data_transformation import DataTransformation  # Import the DataTransformation class for transforming data
from src.components.data_transformation import DataTransformationConfig  # Import configuration for data transformation

from src.components.model_trainer import ModelTrainerConfig  # Import configuration for model training
from src.components.model_trainer import ModelTrainer  # Import the ModelTrainer class for training our machine learning model

@dataclass  # Use dataclass decorator to automatically generate init, repr, etc.
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")  # File path to save the training data
    test_data_path: str = os.path.join('artifacts', "test.csv")    # File path to save the testing data
    raw_data_path: str = os.path.join('artifacts', "data.csv")      # File path to save the raw data

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()  # Initialize data ingestion configuration

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")  # Log the start of the data ingestion process
        try:
            df = pd.read_csv('notebook/data/stud.csv')  # Read the dataset from CSV into a pandas DataFrame
            logging.info('Read the dataset as dataframe')  # Log that the dataset was read successfully

            # Create the directory for the training data file if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save the raw data DataFrame to a CSV file
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")  # Log that train-test splitting is starting
            # Split the DataFrame into training and testing sets (80% train, 20% test)
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save the training set to a CSV file
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            # Save the testing set to a CSV file
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")  # Log that data ingestion has been completed

            # Return the file paths for the training and testing datasets
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            # If an exception occurs, raise a CustomException with the error and system info
            raise CustomException(e, sys)

if __name__ == "__main__":  # If this script is run directly (not imported as a module)
    obj = DataIngestion()  # Create an instance of the DataIngestion class
    train_data, test_data = obj.initiate_data_ingestion()  # Execute the data ingestion process and get the train/test file paths

    data_transformation = DataTransformation()  # Create an instance of DataTransformation to process the data further
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)  # Transform the data into arrays

    modeltrainer = ModelTrainer()  # Create an instance of ModelTrainer to train the machine learning model
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))  # Train the model and print the output/result