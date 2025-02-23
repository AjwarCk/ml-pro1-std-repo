# Import find_packages and setup from setuptools.
# - find_packages: Automatically discovers all packages and sub-packages.
# - setup: Used to define the package metadata and configuration for distribution.
from setuptools import find_packages, setup
# Import List from the typing module for type annotations.
from typing import List
# Define a constant for the editable installation flag used in requirements files.
# '-e .' indicates an editable install of the current package.
HYPEN_E_DOT = '-e .'
def get_requirements(file_path: str) -> List[str]:
    """
    Reads the file specified by 'file_path' (typically requirements.txt) and returns a list of requirements.
    If the editable flag ('-e .') is present, it is removed from the list.
    """
    # Initialize an empty list to store requirements.
    requirements = []
    
    # Open the requirements file for reading.
    with open(file_path) as file_obj:
        # Read all lines from the file. Each line should contain one package requirement.
        requirements = file_obj.readlines()
        # Remove newline characters from each requirement.
        requirements = [req.replace("\n", "") for req in requirements]
        
        # Check if the editable flag is in the list of requirements.
        # If found, remove it since it's not needed in the install_requires list.
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    # Return the cleaned list of requirements.
    return requirements
# Call the setup() function to define your package's metadata and configuration.
setup(
    # Name of the package.
    name='ml-project1',
    
    # Version of the package.
    version='0.0.1',
    
    # Name of the author.
    author='AjwarCK',
    
    # Author's email address.
    author_email='ajwar.trn@outlook.com',
    
    # Automatically discover and include all packages and sub-packages.
    packages=find_packages(),
    
    # Specify the external packages required for this package to work.
    # The get_requirements function reads the requirements.txt file and returns them as a list.
    install_requires=get_requirements('requirements.txt')
)