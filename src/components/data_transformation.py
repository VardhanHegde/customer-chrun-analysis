# Handle missing vaalue
# Outliers treatment
# Handle Imbalance dataset
# Convert catagorical columns into numerical columns

import os,sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocess_obj_file_path = os.path.join("artifacts/data_transformation","preprocessor.pkl")
    
class DataTransformation:
    def __init__(self):
        self.data_transforamtion_config = DataTransformationConfig()
        
    def get_data_transformation_obj(self):
        try:
            pass
        except Exception as e :
            raise CustomException(e,sys)