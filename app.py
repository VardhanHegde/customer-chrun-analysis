from flask import Flask,render_template,request,jsonify
from src.pipeline.prediction_pipline import PredictionPipeline,CustomClass
# from src.logger import logging
# import os,sys

app = Flask(__name__)

@app.route('/',methods = ['GET','POST'])
def prediction_data():
    if request.method == "GET" : 
        return render_template("home.html")
    
    else:
        data = CustomClass( 
            gender = int(request.form.get("gender")), 
            SeniorCitizen = int(request.form.get("SeniorCitizen")), 
            Partner = int(request.form.get("Partner")),
            Dependents = int(request.form.get("Dependents")),
            PhoneService = int(request.form.get("PhoneService")),
            MultipleLines = int(request.form.get("MultipleLines")),
            InternetService = int(request.form.get("InternetService")),
            OnlineSecurity = int(request.form.get("OnlineSecurity")),
            OnlineBackup = int(request.form.get("OnlineBackup")),
            DeviceProtection = int(request.form.get("DeviceProtection")),
            TechSupport = int(request.form.get("TechSupport")),
            StreamingTV = int(request.form.get("StreamingTV")),
            StreamingMovies = int(request.form.get("StreamingMovies")),
            Contract = int(request.form.get("Contract")),
            PaperlessBilling = int(request.form.get("PaperlessBilling")),
            PaymentMethod = int(request.form.get("PaymentMethod")),
            MonthlyCharges = float(request.form.get("MonthlyCharges")),
            TotalCharges = float(request.form.get("TotalCharges")),
            tenure_group = int(request.form.get("tenure_group"))
        )
        
    final_data = data.get_data_into_dataframe()
    print(final_data)
    pipeline_prediction = PredictionPipeline()
    pred = pipeline_prediction.predict(final_data)
    result = pred
    
    if result == 0:
        return render_template("result.html",final_result = "Customer Will Not Churn : {}".format(result))
    elif result ==1:
        return render_template("result.html",final_result = "Customer will Churn {}".format(result))

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5000,debug = True)
