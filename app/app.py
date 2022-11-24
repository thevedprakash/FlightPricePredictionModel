import sys
sys.path.append('src')
# https://stackoverflow.com/questions/4761041/python-import-src-modules-when-running-tests


import pandas as pd
from flask import Flask, json, request, jsonify , render_template
from flask_ngrok import run_with_ngrok
import os
import urllib.request
from werkzeug.utils import secure_filename

from predict import preprocess_and_predict

from variables import airlines, sources, destinations, routes

import joblib
import pickle

app = Flask(__name__)

# run_with_ngrok(app)
# app.secret_key = "caircocoders-ednalan"

# UPLOAD_FOLDER = 'static/uploads'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load data (deserialize)
with open('models/encoded.pickle', 'rb') as handle:
    encoded_dict = pickle.load(handle)

model_path = "models/randomForestModel.pickle"
model= joblib.load(model_path)

import json


@app.route('/api', methods=['POST','GET'])
def predict():
    # Get the data from the POST request.
    json_data = request.get_json(force=True)

    # Convert json data to dataframe
    df = pd.DataFrame.from_dict(pd.json_normalize(json_data), orient='columns')
    print("-"*80)
    print(df)

    # Pre-process and make prediction using model loaded from disk as per the data.
    data = preprocess_and_predict(df,encoded_dict)
    print("-"*80)
    print(data)
    print("-"*80)
 
    prediction = model.predict(data)


    # Take the first value of prediction
    output = prediction[0]
    print("price : ",output)
    return jsonify(output)

@app.route("/")
def root():
    return 'Root of Flask WebApp!'


@app.route("/prediction")
def prediction():

    Airline = request.args.get('Airline')
    Source = request.args.get('Source')
    Destination = request.args.get('Destination')
    Departure_Date = request.args.get('Departure_Date')
    Arrival_Date = request.args.get('Arrival_Date')
    Departure_Time = request.args.get('Departure_Time')
    Arrival_Time = request.args.get('Arrival_Time')
    Route = request.args.get('Route')
    Stops = request.args.get('Stops')
    Additional = request.args.get('Additional')

    data_dict = {
            'Airline'  : Airline,
            'Source'  : Source,
            'Destination'  : Destination,
            'Date_of_Journey'  : Departure_Date,
            'Dep_Time'  : Departure_Time,
            # 'Arrival_Date'  : Arrival_Date,
            'Duration'  : '2h 50m',
            'Arrival_Time'  : Arrival_Time,
            'Route'  : Route,
            'Total_Stops' : Stops,
            'Additional_Info'  : Additional,
        }

    print(data_dict)

    # Convert json data to dataframe
    df = pd.DataFrame.from_dict([data_dict],orient='columns')
    # print("-"*80)
    print(df)

    # Pre-process and make prediction using model loaded from disk as per the data.
    data = preprocess_and_predict(df,encoded_dict)
    print("-"*80)
    print(data)
    print("-"*80)
 
    prediction = model.predict(data)


    return str(prediction[0])

@app.route('/index', methods=['GET', 'POST'])
def index():
    return render_template('index.html', airlines=airlines, 
    sources=sources, destinations=destinations,
    routes=routes , pred=str(9999))

if __name__ == '__main__':
    app.run(debug=True)