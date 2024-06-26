import os,sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from imblearn.combine import SMOTEENN

@dataclass
class DataIngetionConfig:
    train_data_path = os.path.join("artifacts\data_ingetion","train.csv")
    test_data_path = os.path.join("artifacts\data_ingetion","test.csv")
    raw_data_path = os.path.join("artifacts\data_ingetion","raw.csv")
    
# \notbook\data\tel_churn.csv

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngetionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("Data ingestion started")
        try:
            logging.info("Data read using pands library from local system")
            data = pd.read_csv(os.path.join("notebook\data","tel_churn.csv"))
            logging.info("Data reading completed")
            
            y = []
            y = pd.DataFrame(data["Churn"])
            x = data.drop(["Churn"],axis = 1)
            x = pd.DataFrame(x)

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path,index =False)
            logging.info("Data splitted into train and test")
            
            #smotthning the dataframe
            sm = SMOTEENN()
            X_resampled, y_resampled = sm.fit_resample(x,y)
            print(X_resampled.shape)
            print(y_resampled.shape)
            data1 = pd.concat((X_resampled,y_resampled),axis=1,)
            
            train_set ,test_set = train_test_split(data1,test_size=0.3,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header =True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header =True)
            
            logging.info("Data ingetion completed")
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path,test_data_path = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr , test_arr , _ =  data_transformation.initiate_data_transformation(train_data_path,test_data_path)
    
    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr,test_arr))