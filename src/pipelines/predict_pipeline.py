import sys  # Import sys module for system-specific parameters and functions
import os  # Import os module for operating system interactions (e.g., file paths)
import pandas as pd  # Import pandas for data manipulation and DataFrame operations
from src.exception import CustomException  # Import CustomException for customized error handling
from src.utils import load_object  # Import load_object function to load serialized objects (e.g., model, preprocessor)

class PredictPipeline:
    def __init__(self):
        pass  # No initialization needed for this pipeline class

    def predict(self, features):  # Method to make predictions given input features
        try:  # Try block to catch and handle any errors during prediction
            model_path = os.path.join("artifacts", "model.pkl")  # Build file path for the serialized model
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')  # Build file path for the serialized preprocessor
            print("Before Loading")  # Debug print before loading objects
            model = load_object(file_path=model_path)  # Load the model from the file using load_object utility
            preprocessor = load_object(file_path=preprocessor_path)  # Load the preprocessor from the file using load_object utility
            print("After Loading")  # Debug print after loading objects
            data_scaled = preprocessor.transform(features)  # Transform the input features using the preprocessor
            preds = model.predict(data_scaled)  # Generate predictions using the loaded model
            return preds  # Return the prediction results
        
        except Exception as e:  # If an exception occurs during prediction
            raise CustomException(e, sys)  # Raise a custom exception with the error details and system info

class CustomData:
    def __init__(self,
                 gender: str,  # Gender of the individual
                 race_ethnicity: str,  # Race or ethnicity of the individual
                 parental_level_of_education,  # Parental level of education
                 lunch: str,  # Type of lunch (e.g., standard, free/reduced)
                 test_preparation_course: str,  # Status of test preparation course (e.g., completed, none)
                 reading_score: int,  # Reading score as an integer
                 writing_score: int):  # Writing score as an integer

        self.gender = gender  # Store gender in the instance variable
        self.race_ethnicity = race_ethnicity  # Store race/ethnicity in the instance variable
        self.parental_level_of_education = parental_level_of_education  # Store parental education level
        self.lunch = lunch  # Store lunch type
        self.test_preparation_course = test_preparation_course  # Store test preparation course status
        self.reading_score = reading_score  # Store reading score
        self.writing_score = writing_score  # Store writing score

    def get_data_as_data_frame(self):  # Method to convert custom data into a pandas DataFrame
        try:  # Try block to catch and handle errors during DataFrame creation
            custom_data_input_dict = {  # Create a dictionary with the custom data values
                "gender": [self.gender],  # 'gender' key with its value in a list
                "race_ethnicity": [self.race_ethnicity],  # 'race_ethnicity' key with its value in a list
                "parental_level_of_education": [self.parental_level_of_education],  # 'parental_level_of_education' key with its value in a list
                "lunch": [self.lunch],  # 'lunch' key with its value in a list
                "test_preparation_course": [self.test_preparation_course],  # 'test_preparation_course' key with its value in a list
                "reading_score": [self.reading_score],  # 'reading_score' key with its value in a list
                "writing_score": [self.writing_score],  # 'writing_score' key with its value in a list
            }
            return pd.DataFrame(custom_data_input_dict)  # Convert the dictionary into a pandas DataFrame and return it

        except Exception as e:  # If an error occurs while creating the DataFrame
            raise CustomException(e, sys)  # Raise a custom exception with error details and system info