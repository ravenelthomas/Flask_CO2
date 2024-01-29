from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd 
import pickle
from sklearn.preprocessing import PolynomialFeatures

app = Flask(__name__)

model = pickle.load(open('model.pickle',"rb"))
cols = ['MODELYEAR', 'ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']

@app.route('/')
def home():
    return render_template("input.html")

@app.route('/predict',methods=['POST'])
def predict():
    features = [x for x in request.form.values()]
    final = np.array(features)
    final= final.astype(np.float32)
    poly_features = PolynomialFeatures(degree=2)
    final_poly = poly_features.fit_transform([final])
    data_unseen = pd.DataFrame(final_poly)
    prediction=model.predict(data_unseen)
    result=prediction[0]
    return render_template('input.html',resultat=f"Les émissions de CO2 seront de {result:.2f}")

@app.route('/predict_api',methods=['POST'])
def predict_api():
    features = [x for x in request.args.values()]
    final = np.array(features)
    data_unseen = pd.DataFrame([final], columns = cols)
    prediction=model.predict(data_unseen)
    result=prediction[0]
    return jsonify(result)


if __name__ == '__main__':
    app.run()

#lancer l'application (une des trois possibilités)
# flask --app insurance_app run
# python insurance_app.py
# avec
# gunicorn insurance_app:app