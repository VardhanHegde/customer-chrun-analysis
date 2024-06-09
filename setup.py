# setup file helps us to install the project as a package , and can be released
from setuptools import find_packages,setup
from typing import List

HYPHON_E_DOT = "-e ." # -e . is to install the setup file

def get_requirements(filepath:str ) -> List[str]:
    requirements = []
    
    with open(filepath) as file_obj:
        requirements = file_obj.readlines()
        requirements = [i.replace("\n","") for i in requirements]
        
        if HYPHON_E_DOT in requirements:
            requirements.remove(HYPHON_E_DOT)

setup(name = 'Customer Churn Project',
    version = '0.0.1',
    description= 'ML python Project',
    author = 'Vardhan Hegde',
    author_email='hegdevardhan@gmail.com',
    # url = '',
    packages = find_packages(), 
    install_requires = get_requirements('requirement.txt')
)