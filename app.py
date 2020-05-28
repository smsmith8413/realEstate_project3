from flask import Flask,request, url_for, redirect, render_template, jsonify
from pycaret.regression import *
import pandas as pd
import pickle
import numpy as np


app = Flask(__name__)

model = load_model('deployment_1')
cols = ['PostalCode', 'BathsTotal', 'BedsTotal', 'DOM',
       'NumberOfDiningAreas', 'NumberOfLivingAreas', 'NumberOfStories',
       'OriginalListPrice', 'ParkingSpacesGarage', 'PoolYN', 'SqFtTotal',
       'YearBuilt']

# open home page
@app.route("/")
def index():

    return render_template("index.html")

# process flow path
@app.route("/process")
def process():

    return render_template("process.html")

# process flow path
@app.route("/data")
def data():

    return render_template("datadetails.html")

# machine learning path
@app.route("/machine")
def machine():

    return render_template("machine.html")

# tableau path
@app.route("/tableau")
def tableau():

    return render_template("tableau.html")
@app.route("/before")
def before():

    return render_template("Collin_County_RE_Data_Analysis-Before.html")

@app.route("/after")
def after():

    return render_template("Collin_County_RE_Data_Analysis-After.html")

# price prediction page with user inputs path, and function code
@app.route("/calculator")
def calc():

    return render_template("calculator.html", blah="")

@app.route('/calculator_predictor',methods=['POST'])
def calculator():
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final], columns = cols)
    prediction = predict_model(model, data=data_unseen, round = 0)
    prediction = int(prediction.Label[0])
    return render_template('calculator.html',pred='   Predicted Sales Price of the house will be ${:,}'.format(prediction))


if __name__ == "__main__":
    app.run(debug=True)