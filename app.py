from flask import Flask, request, render_template  # Import Flask, request, and render_template for building the web app
import numpy as np  # Import numpy for numerical operations (if needed)
import pandas as pd  # Import pandas for data manipulation

from sklearn.preprocessing import StandardScaler  # Import StandardScaler for potential data scaling (not used in this snippet)
from src.pipelines.predict_pipeline import CustomData, PredictPipeline  # Import CustomData and PredictPipeline from our custom module

application = Flask(__name__)  # Create a Flask application instance using the current module's name
app = application  # Alias the application instance as 'app' for convenience

## Route for a home page
@app.route('/')  # Define a route for the root URL ('/')
def index():  # Define the view function for the home page
    return render_template('index.html')  # Render and return the 'index.html' template for the home page

@app.route('/predictdata', methods=['GET', 'POST'])  # Define a route '/predictdata' that accepts both GET and POST requests
def predict_datapoint():  # Define the view function to handle prediction requests
    if request.method == 'GET':  # Check if the incoming request is a GET request
        return render_template('home.html')  # For GET requests, render and return the 'home.html' template
    else:  # Otherwise, handle the POST request (when form data is submitted)
        data = CustomData(  # Create an instance of CustomData using values from the submitted form data
            gender=request.form.get('gender'),  # Retrieve the 'gender' field from the form data
            race_ethnicity=request.form.get('ethnicity'),  # Retrieve the 'ethnicity' field from the form data
            parental_level_of_education=request.form.get('parental_level_of_education'),  # Retrieve the 'parental_level_of_education' field
            lunch=request.form.get('lunch'),  # Retrieve the 'lunch' field from the form data
            test_preparation_course=request.form.get('test_preparation_course'),  # Retrieve the 'test_preparation_course' field
            reading_score=float(request.form.get('reading_score')),  # Retrieve and convert 'reading_score' to float
            writing_score=float(request.form.get('writing_score'))  # Retrieve and convert 'writing_score' to float
        )  # End of CustomData instantiation

        pred_df = data.get_data_as_data_frame()  # Convert the custom data into a pandas DataFrame
        print(pred_df)  # Print the DataFrame to the console for debugging purposes
        print("Before Prediction")  # Debug message before starting prediction

        predict_pipeline = PredictPipeline()  # Create an instance of PredictPipeline to handle prediction logic
        print("Mid Prediction")  # Debug message indicating mid-point of prediction process
        results = predict_pipeline.predict(pred_df)  # Generate predictions using the PredictPipeline on the DataFrame
        print("after Prediction")  # Debug message after prediction is complete
        print("Prediction Result:", results)  # Print the full prediction results to the console for debugging
        return render_template('home.html', results=results[0])  # Render 'home.html' and pass the first prediction result to the template

if __name__ == "__main__":  # Check if this script is executed as the main program (not imported as a module)
    app.run(host="0.0.0.0", debug=True)  # Run the Flask app on all available network interfaces with debug mode enabled