from src.logger import logging
from src.exception import CustomException
import os,sys
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,fbeta_score,classification_report,confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from imblearn.combine import SMOTEENN

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path,"wb") as file_obj: 
            pickle.dump(obj,file_obj)
        
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(x_train,x_test,y_train,y_test,models,params):
    try:
        j = 0
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = params[list(models.keys())[i]]
            
            GS = GridSearchCV(model,para,cv = 5,scoring= 'f1_weighted')
            GS.fit(x_train,y_train)
            
            model.set_params(**GS.best_params_)
            model.fit(x_train,y_train)
            
            # /prediction 
            y_pred = model.predict(x_test)   
            test_model_accuracy = accuracy_score(y_test,y_pred)
            
            # if j == 0:
            #     print('*'*190)
            #     model_rf=RandomForestClassifier(n_estimators=100, criterion='gini', random_state = 100,max_depth=6, min_samples_leaf=8)
            #     model_rf.fit(x_train,y_train)
            #     y_pred1=model_rf.predict(x_test)
            #     print(confusion_matrix(y_test, y_pred1))
            #     print(classification_report(y_test, y_pred1, labels=[0,1]))
            #     print(f1_score(y_test,y_pred1))
            #     print('*'*190)
                
                

            #     x = np.concatenate((x_train, x_test), axis=0)
            #     y = np.concatenate((y_train, y_test), axis=0)
                
            #     sm = SMOTEENN()
            #     X_resampled, y_resampled = sm.fit_resample(x,y)
                
            #     xr_train,xr_test,yr_train,yr_test=train_test_split(X_resampled, y_resampled,test_size=0.2)
                
            #     model_dt_smote=RandomForestClassifier(criterion='gini', random_state = 100,max_depth=6, min_samples_leaf=8)
            #     model_dt_smote.fit(xr_train,yr_train)
            #     yr_predict = model_dt_smote.predict(xr_test)
            #     model_score_r = model_dt_smote.score(xr_test, yr_test)
                
            #     print(model_score_r)
            #     print(classification_report(yr_test, yr_predict))   
            #     print(confusion_matrix(yr_test, yr_predict))
                
            #     j = 100          
            #     print('*'*190)

            report[list(models.values())[i]] = test_model_accuracy
            return report
        
        
                        
    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
         
    except Exception as e:
        raise CustomException(e,sys)