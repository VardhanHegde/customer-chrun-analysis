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
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocess_obj_file_path = os.path.join("artifacts/data_transformation","preprocessor.pkl")
    
class DataTransformation:
    def __init__(self):
        self.data_transforamtion_config = DataTransformationConfig()
        
    def get_data_transformation_obj(self):
        try:
            logging.info("Data trasformation started")
            
            numerical_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
            'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
            'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
            'MonthlyCharges', 'TotalCharges','tenure_group']
            
            num_pipline = Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy='median')),
                    ("scalar",StandardScaler()),
                ]
            )
            
            cat_pipline = Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy='most_frequent')),
                ]
            )
              
            preprocessor = ColumnTransformer([
                ("cat_pipline",cat_pipline,numerical_features)
            ])
            
            return preprocessor
        
        except Exception as e :
            raise CustomException(e,sys)
        
    def remote_outliers_IQR(self,col,df):
        try:
            logging.info("Outliers handling code")
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            
            iqr = Q3 -Q1            
            upper_limit = Q3 + 1.5*iqr
            lower_limit = Q1 - 1.5*iqr
               
            df.loc[(df[col]>upper_limit),col] = upper_limit
            df.loc[(df[col]<lower_limit),col] = lower_limit
            
            return df

        except Exception as e :
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            
            numerical_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
            'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
            'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
            'MonthlyCharges', 'TotalCharges','tenure_group']
            
            for col in numerical_features:
                self.remote_outliers_IQR(col = col,df = train_data)
            
            logging.info("Outlier capped on out train data")
            
            for col in numerical_features:
                self.remote_outliers_IQR(col = col,df = test_data)
                
            logging.info("Outlier capped on out test data")
            
            preprocessor_obj = self.get_data_transformation_obj()
            
            
            target_column = "Churn"
            drop_column = [target_column]
            
            logging.info("Splitting data into dependent and independent features")
            input_feature_train_data = train_data.drop(drop_column,axis = 1)
            target_feature_train_data = train_data[target_column]
            
            logging.info("Splitting data into dependent and independent features")
            input_feature_test_data = test_data.drop(drop_column,axis = 1)
            target_feature_test_data = test_data[target_column]
            
            #apply transformation on train and test data
            input_train_arr = preprocessor_obj.fit_transform(input_feature_train_data)
            input_test_arr = preprocessor_obj.transform(input_feature_test_data)
            
            # apply preprocessor object on train and test data
            train_arr = np.c_[input_train_arr,np.array(target_feature_train_data)]
            test_arr = np.c_[input_test_arr,np.array(target_feature_test_data)]
            
            save_object(file_path = self.data_transforamtion_config.preprocess_obj_file_path,
                        obj = preprocessor_obj)
            
            return(train_arr,
                   test_arr,
                   self.data_transforamtion_config.preprocess_obj_file_path)
            
        except Exception as e :
            raise CustomException(e,sys)