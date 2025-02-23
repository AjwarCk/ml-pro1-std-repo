import logging                # Import the logging module to handle logging functionality
import os                     # Import os for operating system dependent functionality (e.g., file paths)
from datetime import datetime # Import datetime to get the current time for naming the log file
# Create a log file name using the current date and time, formatted as MM_DD_YYYY_HH_MM_SS.log
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"  # Log file name
# Define the directory where log files will be stored (a "logs" folder in the current working directory)
logs_dir = os.path.join(os.getcwd(), "logs")  # Construct the logs directory path
# Create the logs directory if it does not already exist
os.makedirs(logs_dir, exist_ok=True)  # exist_ok=True prevents an error if the directory exists
# Combine the logs directory path and log file name to get the full path for the log file
LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)  # Complete log file path
# Configure the logging module:
#   - 'filename' specifies the file where logs are written
#   - 'format' defines the log message format
#   - 'level' sets the minimum log level to capture (INFO and above in this case)
logging.basicConfig(
    filename=LOG_FILE_PATH,  # Path to the log file
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",  # Log message format
    level=logging.INFO,  # Minimum level of messages to log (INFO, WARNING, ERROR, etc.)
)
# # Check if this script is being run directly (not imported as a module)
# if __name__ == "__main__":
#    # Write an informational log message to indicate that logging has started
#     logging.info("Logging has started")