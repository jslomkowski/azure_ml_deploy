import numpy as np
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load('boston_model.sav')


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    return render_template(
        'home.html', prediction_text=f"Expected value:{round(prediction[0]*1000,2)} USD")


if __name__ == '__main__':
    app.run(debug=True)
