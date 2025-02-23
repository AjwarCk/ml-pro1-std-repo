import os  # Import the os module for interacting with the operating system
import sys  # Import the sys module for system-specific parameters and functions

import numpy as np  # Import numpy for numerical operations
import pandas as pd  # Import pandas for data manipulation and analysis
import dill  # Import dill for advanced object serialization (alternative to pickle)
import pickle  # Import pickle for object serialization

from sklearn.metrics import r2_score  # Import r2_score to evaluate regression model performance
from sklearn.model_selection import GridSearchCV  # Import GridSearchCV for hyperparameter tuning using cross-validation

from src.exception import CustomException  # Import a custom exception class for tailored error handling

def save_object(file_path, obj):  # Define a function to serialize and save an object to a file
    try:  # Start a try block to handle exceptions during saving
        dir_path = os.path.dirname(file_path)  # Extract the directory path from the given file path

        os.makedirs(dir_path, exist_ok=True)  # Create the directory if it doesn't exist; do nothing if it does

        with open(file_path, "wb") as file_obj:  # Open the file in write-binary mode
            pickle.dump(obj, file_obj)  # Serialize the object and write it to the file

    except Exception as e:  # If any exception occurs during saving
        raise CustomException(e, sys)  # Raise a custom exception with error details and system info
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param):  # Define a function to evaluate and tune multiple models
    try:  # Start a try block to handle exceptions during model evaluation
        report = {}  # Initialize an empty dictionary to store model evaluation scores

        for i in range(len(list(models))):  # Loop over each model using its index in the models dictionary
            model = list(models.values())[i]  # Retrieve the model object from the dictionary by index
            para = param[list(models.keys())[i]]  # Retrieve the hyperparameters corresponding to the model

            gs = GridSearchCV(model, para, cv=3)  # Initialize GridSearchCV with the model, its parameters, and 3-fold cross-validation
            gs.fit(X_train, y_train)  # Fit GridSearchCV on the training data to find the best hyperparameters

            model.set_params(**gs.best_params_)  # Update the model with the best hyperparameters found
            model.fit(X_train, y_train)  # Train the model on the entire training dataset

            # model.fit(X_train, train_y) # Train model  # (Alternative training method commented out)

            y_train_pred = model.predict(X_train)  # Generate predictions on the training data
            y_test_pred = model.predict(X_test)  # Generate predictions on the test data

            train_model_score = r2_score(y_train, y_train_pred)  # Calculate the R^2 score for the training predictions
            test_model_score = r2_score(y_test, y_test_pred)  # Calculate the R^2 score for the test predictions

            report[list(models.keys())[i]] = test_model_score  # Store the test R^2 score in the report dictionary with the model's key

        return report  # Return the dictionary containing evaluation scores for all models
    
    except Exception as e:  # If any exception occurs during evaluation
        raise CustomException(e, sys)  # Raise a custom exception with error details and system info
    
def load_object(file_path):  # Define a function to load a serialized object from a file
    try:  # Start a try block to handle exceptions during loading
        with open(file_path, "rb") as file_obj:  # Open the file in read-binary mode
            return pickle.load(file_obj)  # Deserialize and return the object from the file
        
    except Exception as e:  # If any exception occurs during loading
        raise CustomException(e, sys)  # Raise a custom exception with error details and system info