from flask import Flask, request
import joblib
import pandas as pd
import os

#instanciate o Flask
app = Flask(__name__)

#Load machine learning model
model = joblib.load(open('model.pkl', 'rb'))

@app.route("/")
def home():
    return "Hello, World!"

@app.route('/predict', methods = ['POST'])
def predict():
    
    #collect data
    archive_json = request.get_json()
    
    #transform data
    if archive_json:#if not null
        if isinstance(archive_json, dict):#unique value
           df_raw = pd.DataFrame(archive_json, index=[0])
        else:
           df_raw = pd.DataFrame(archive_json,#more than one raw
               columns = archive_json[0].keys())
    
        # prediction
        result = model.predict(df_raw)
        
        # add to column
        df_raw['prediction'] = result
        
        # return original df with predictions
        return df_raw.to_json(orient = 'records')
    
    return archive_json

if __name__ == '__main__':
    # start flask
    app.run(debug=True)
