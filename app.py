import tensorflow as tf
from tensorflow.keras.models import Sequential,load_model,model_from_json
from tensorflow.keras.layers import Dense, Dropout,Activation,MaxPooling2D,Conv2D,Flatten
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing import image
import numpy as np
import h5py
import os
import sys
import json
from sklearn.preprocessing import StandardScaler
from predictor import sc
import pandas as pd
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename


# Define a flask app
app = Flask(__name__)

with open('heart_diseases.json','r') as f:
    model = model_from_json(f.read())


# Load your trained model
model.load_weights('heart_diseases.h5')   

 


@app.route('/')
 

@app.route('/index')
def index():
	return render_template('index.html')

@app.route('/chart')
def chart():
	return render_template('chart.html')

#@app.route('/future')
#def future():
#	return render_template('future.html')    

@app.route('/login')
def login():
	return render_template('login.html')
@app.route('/upload')
def upload():
    return render_template('upload.html')  
@app.route('/preview',methods=["POST"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset,encoding = 'unicode_escape')
        df.set_index('Id', inplace=True)
        return render_template("preview.html",df_view = df)	 

@app.route('/prediction', methods = ['GET', 'POST'])
def prediction():
    # Main page
    return render_template('prediction.html')

@app.route('/predict',methods=['POST'])
def predict():
	int_feature = [x for x in request.form.values()]
	print(int_feature)
        # Make Prediction
	prediction = model.predict(sc.transform(np.array([int_feature])))
	print(prediction)
        # Process your result for human
	if prediction > 0.5:
		result = "Heart diseases"
	else:
		result = "No diseases"
	return render_template('prediction.html', prediction_value=result)
 
@app.route('/performance')
def performance():
	return render_template('performance.html')         

if __name__ == '__main__':
    app.run()
