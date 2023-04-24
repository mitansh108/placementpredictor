from flask import Flask,render_template,request
import pickle
import pandas as pd
import numpy as np
import xgboost as xg

model = pickle.load(open('/Users/mitanshpatel/Downloads/xgb.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict_placement():
    cgpa = (request.form.get('cgpa'))
    iq = (request.form.get('iq'))
    profile_score = (request.form.get('profile_score'))

    # prediction
    result = model.predict(np.array([cgpa,iq,profile_score]).reshape(1,3))

    if result[0] == 1:
        result = 'placed'
    else:
        result = 'not placed'

    return str(result)


if __name__ == '__main__':
    app.run(debug=True,port=8002)