import sys             # Import the sys module to access system-specific functions (like sys.exc_info())
from src.logger import logging         # Import the logging module to log messages (ensure logging is configured before use)
def error_message_detail(error, error_details: sys):
    """
    Generate a detailed error message using the provided error and error details.
    
    Parameters:
    - error: The exception object that was raised.
    - error_details: The sys module (passed in so that we can call sys.exc_info()).
    
    Returns:
    - A formatted string containing the filename, line number, and error message.
    """
    # Retrieve the traceback info using sys.exc_info() and unpack it.
    # sys.exc_info() returns a tuple: (exception type, exception instance, traceback object).
    _, _, exc_tb = error_details.exc_info()  # We ignore the exception type and instance; we only need the traceback.
    
    # Extract the filename from the traceback object where the exception occurred.
    file_name = exc_tb.tb_frame.f_code.co_filename  # Get the name of the file from the traceback frame.
    
    # Create a detailed error message including the filename, the line number, and the error message.
    error_message = (
        f"Error occurred in python script: [{file_name}] "
        f"at line number: [{exc_tb.tb_lineno}] with error: [{str(error)}]"
    )
    
    # Return the detailed error message.
    return error_message
class CustomException(Exception):
    """
    A custom exception class that extends the built-in Exception class.
    It uses the error_message_detail function to generate a detailed error message.
    """
    def __init__(self, error_message, error_detail: sys):
        # Initialize the base Exception class with the basic error message.
        super().__init__(error_message)
        # Generate and store a detailed error message using the error_message_detail function.
        self.error_message = error_message_detail(error_message, error_detail)
    
    def __str__(self):
        # When the exception is converted to a string (for example, when printed), return the detailed error message.
        return self.error_message
# if __name__ == "__main__":
#     # Configure the logging module to log messages at the INFO level.
#     # This is necessary to ensure that our logging.info() calls output to the console (or file, if configured).
#     logging.basicConfig(level=logging.INFO)
    
#     try:
#         # Intentionally perform a division by zero to trigger an exception.
#         a = 1 / 0
#     except Exception as e:  # Catch any exception and store it in variable 'e'
#         # Log an informational message indicating that a divide-by-zero error occurred.
#         logging.info("Divide by Zero")
#         # Raise a CustomException with the caught exception 'e' and pass the sys module to provide traceback details.
#         raise CustomException(e, sys)