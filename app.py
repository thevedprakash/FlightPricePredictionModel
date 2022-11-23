from flask import Flask, json, request, jsonify
from flask_ngrok import run_with_ngrok
import os
import urllib.request
from werkzeug.utils import secure_filename

app = Flask(__name__)
run_with_ngrok(app)
app.secret_key = "caircocoders-ednalan"

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = saved_model
encoded_dict


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

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        Airline = request.form['Airline']
        Source = request.form['baths']
        Destination = request.form['sqft']
        Departure_Date = request.form['Airline']
        Arrival_Date = request.form['baths']
        Departure_Time = request.form['Airline']
        Arrival_Time = request.form['baths']
        Route = request.form['sqft']
        Stops = request.form['Airline']
        Additional = request.form['baths']

        test_json = xxx
        test_input = preprocess_and_predict(test_json,encoded_dict)
        pred = model.predict(test_input)

        return render_template('index.html', pred=str(pred))
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run()