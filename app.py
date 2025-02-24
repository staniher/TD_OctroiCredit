from flask import Flask, request, render_template

app=Flask(__name__)

import joblib

modele_logistique=joblib.load("modele logistique.plk")

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    import numpy as np
    features=[float(i) for i in request.form.values()]
    mesfeatures=np.array(features).reshape(1,-1)
    prediction=modele_logistique.predict(mesfeatures)
    resultat="Yes" if prediction[0]==1 else "No"
    return render_template("index.html", resultat=f"credit octroyer {resultat}")
if __name__=="__main__":
    app.run(debug=False)
