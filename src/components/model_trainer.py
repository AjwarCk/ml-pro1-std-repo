import os  # Import os module for interacting with the operating system (e.g., file paths)
import sys  # Import sys module for system-specific parameters and functions (e.g., exception info)
from dataclasses import dataclass  # Import dataclass decorator to simplify class creation for storing data

# Import CatBoostRegressor for gradient boosting on decision trees from the CatBoost library
from catboost import CatBoostRegressor

# Import various ensemble regressors from scikit-learn
from sklearn.ensemble import (
    AdaBoostRegressor,         # Import AdaBoostRegressor for boosting-based regression
    GradientBoostingRegressor, # Import GradientBoostingRegressor for gradient boosting regression
    RandomForestRegressor      # Import RandomForestRegressor for ensemble learning using decision trees
)

from sklearn.linear_model import LinearRegression  # Import LinearRegression for simple linear regression
from sklearn.metrics import r2_score  # Import r2_score to evaluate regression model performance
from sklearn.neighbors import KNeighborsRegressor  # Import KNeighborsRegressor (not used in current script)
from sklearn.tree import DecisionTreeRegressor  # Import DecisionTreeRegressor for regression using decision trees
from xgboost import XGBRegressor  # Import XGBRegressor from the XGBoost library for gradient boosting regression

# Import custom exception and logger for project-specific error handling and logging
from src.exception import CustomException  # Import CustomException for raising project-specific exceptions
from src.logger import logging  # Import logging for standardized logging throughout the project

# Import utility functions for saving objects and evaluating models
from src.utils import save_object, evaluate_models  # Utility functions for model persistence and evaluation

@dataclass  # Use dataclass to automatically generate init and other methods for configuration class
class ModelTrainerConfig:
    # Define the file path where the trained model will be saved
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        # Initialize the configuration for model training using ModelTrainerConfig dataclass
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            # Log the start of data splitting process for train and test arrays
            logging.info("Split train and test input data")
            
            # Split the input arrays into features (X) and labels (y)
            # For both train and test arrays, all columns except the last are features and the last column is the target
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],  # Training features: all columns except the last
                train_array[:, -1],   # Training target: the last column
                test_array[:, :-1],   # Testing features: all columns except the last
                test_array[:, -1]     # Testing target: the last column
            )
            
            # Define a dictionary of regression models to be evaluated
            models = {
                "Random Forest": RandomForestRegressor(),  # Random Forest model instance
                "Decision Tree": DecisionTreeRegressor(),  # Decision Tree model instance
                "Gradient Boosting": GradientBoostingRegressor(),  # Gradient Boosting model instance
                "Linear Regression": LinearRegression(),  # Linear Regression model instance
                "XGBRegressor": XGBRegressor(),  # XGBoost regressor model instance
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),  # CatBoost regressor with verbosity disabled
                "AdaBoost Regressor": AdaBoostRegressor(),  # AdaBoost regressor model instance
            }
            
            # Define hyperparameter grids for each model to perform model tuning
            params = {
                "Decision Tree": {
                    "criterion": ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # "splitter": ['best', 'random'],  # (Commented out) Potential hyperparameter for splitting strategy
                    # "max_features": ['sqrt', 'log2']  # (Commented out) Potential hyperparameter for feature selection
                },
                "Random Forest": {
                    # "criterion": ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],  # (Commented out)
                    # "max_features": ['sqrt', 'log2', 'None'],  # (Commented out)
                    "n_estimators": [8, 16, 32, 64, 128, 256]  # Number of trees in the forest to try during tuning
                },
                "Gradient Boosting": {
                    # "loss": ['squared_error', 'huber', 'absolute_error', 'quantile'],  # (Commented out)
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],  # Learning rates to try for boosting
                    "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],  # Subsample ratios to use during training
                    # "criterion": ['squared_error', 'friedman_mse'],  # (Commented out)
                    # "max_features": ['auto', 'sqrt', 'log2'],  # (Commented out)
                    "n_estimators": [8, 16, 32, 64, 128, 256]  # Number of boosting stages to try
                },
                "Linear Regression": {},  # No hyperparameters to tune for Linear Regression
                "XGBRegressor": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],  # Learning rates to try for XGBoost
                    "n_estimators": [8, 16, 32, 64, 128, 256]  # Number of boosting rounds to try for XGBoost
                },
                "CatBoosting Regressor": {
                    "depth": [6, 8, 10],  # Depths of trees to try for CatBoost
                    "learning_rate": [0.01, 0.05, 0.1],  # Learning rates to try for CatBoost
                    "iterations": [30, 50, 100]  # Number of iterations (trees) to try for CatBoost
                },
                "AdaBoost Regressor": {
                    "learning_rate": [0.1, 0.01, 0.5, 0.001],  # Learning rates to try for AdaBoost
                    # "loss": ['linear', 'square', 'exponential'],  # (Commented out) Potential loss functions
                    "n_estimators": [8, 16, 32, 64, 128, 256]  # Number of boosting rounds to try for AdaBoost
                }
            }

            # Evaluate all the models using a utility function that performs training, hyperparameter tuning, and testing.
            model_report: dict = evaluate_models(
                X_train=X_train, 
                y_train=y_train, 
                X_test=X_test, 
                y_test=y_test,
                models=models, 
                param=params
            )
            # The model_report is a dictionary with model names as keys and their evaluation scores as values.

            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))
            # Identify the best score among all evaluated models.

            ## To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            # Retrieve the name of the model corresponding to the best score.

            best_model = models[best_model_name]
            # Get the best model instance from the models dictionary using its name.

            # Check if the best model's score meets a minimum threshold (here, 0.6)
            if best_model_score < 0.6:
                raise CustomException("No best model found")
                # If no model performs sufficiently well, raise a CustomException

            logging.info(f"Best found model on both training and testing dataset")
            # Log that the best model has been successfully identified

            # Save the best model object to disk using the specified file path in the configuration
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Use the best model to make predictions on the test dataset
            predicted = best_model.predict(X_test)

            # Calculate the R^2 score to assess the performance of the model's predictions
            r2_square = r2_score(y_test, predicted)
            
            return r2_square  # Return the R^2 score as the performance metric

        except Exception as e:
            # In case of any exception, raise a CustomException with details about the error and system info
            raise CustomException(e, sys)