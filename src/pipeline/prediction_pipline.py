#Create prediction peipeline class
#create function for load a object
#Create custome class basd upon our dataset
#Create function to convert data into Dataframe with the help of Dict

import os, sys
from src.logger import logging
from src.exception import CustomException
import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.utils import load_object

class PredictionPipeline:
    def _init__(self):
        pass

    def predict(self, features):
        preprocessor_path = os.path.join("artifacts/data_transformation","preprocessor.pkl")
        model_path = os.path.join('artifacts/model_trainer','model.pkl')
        
        processor = load_object(preprocessor_path)
        model = load_object(model_path)
        
        # scaled = processor.transform(features)
        pred = model.predict(features)
        
        return pred

class CustomClass:
    def __init__(
        self,
        gender:int, 
        SeniorCitizen:int, 
        Partner:int,
        Dependents:int,
        PhoneService:int,
        MultipleLines:int,
        InternetService:int,
        OnlineSecurity:int,
        OnlineBackup:int,
        DeviceProtection:int,
        TechSupport:int,
        StreamingTV:int,
        StreamingMovies:int,
        Contract:int,
        PaperlessBilling:int,
        PaymentMethod:int,
        MonthlyCharges:float,
        TotalCharges:float,
        tenure_group:int
        ):
        
        self.gender = gender  
        self.SeniorCitizen = SeniorCitizen  
        self.Partner = Partner 
        self.Dependents = Dependents 
        self.PhoneService = PhoneService 
        self.MultipleLines = MultipleLines 
        self.InternetService = InternetService 
        self.OnlineSecurity = OnlineSecurity 
        self.OnlineBackup = OnlineBackup 
        self.DeviceProtection = DeviceProtection 
        self.TechSupport = TechSupport 
        self.StreamingTV = StreamingTV 
        self.StreamingMovies = StreamingMovies 
        self.Contract = Contract 
        self.PaperlessBilling = PaperlessBilling 
        self.PaymentMethod = PaymentMethod 
        self.MonthlyCharges = MonthlyCharges 
        self.TotalCharges = TotalCharges 
        self.tenure_group = tenure_group
        
    def get_data_into_dataframe(self):
        try:
            custom_input = {
                "gender" : [self.gender],
                "SeniorCitizen" : [self.SeniorCitizen],
                "Partner" : [self.Partner],
                "Dependents" : [self.Dependents],
                "PhoneService" : [self.PhoneService],
                "MultipleLines" : [self.MultipleLines],
                "InternetService" : [self.InternetService],
                "OnlineSecurity" : [self.OnlineSecurity],
                "OnlineBackup" : [self.OnlineBackup],
                "DeviceProtection" : [self.DeviceProtection],
                "TechSupport" : [self.TechSupport],
                "StreamingTV" : [self.StreamingTV],
                "StreamingMovies" : [self.StreamingMovies],
                "Contract" : [self.Contract],
                "PaperlessBilling" : [self.PaperlessBilling],
                "PaymentMethod" : [self.PaymentMethod],
                "MonthlyCharges" : [self.MonthlyCharges],
                "TotalCharges" : [self.TotalCharges],
                "tenure_group" : [self.tenure_group]
            }
            
            data = pd.DataFrame(custom_input)
            return data
        
        except Exception as e:
            raise CustomException(e,sys)