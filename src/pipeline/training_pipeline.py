import os,sys

from src.logger import logging
from src.exception import CustomException
from src.components.data_ingetion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from dataclasses import dataclass

if __name__ == "__main__":
    
    obj = DataIngestion()
    train_data_path,test_data_path = obj.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    train_array,test_array, _ = data_transformation.initiate_data_transformation(train_data_path,test_data_path)
    
    model_training = ModelTrainer()
    model_training.initiate_model_trainer(train_array,test_array)
 