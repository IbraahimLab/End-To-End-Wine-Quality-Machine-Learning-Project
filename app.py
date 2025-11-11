from flask import Flask, render_template, request
import pandas as pd
from wine_quality.entity.S3_estimator import WineEstimator
from wine_quality.constants import *

app = Flask(__name__)

# Initialize model from S3
model = WineEstimator(
    bucket_name="wine-project-s3-bucket",  # replace with your actual bucket name
    model_path="model.pkl"               # replace if your path is different
)

@app.route('/')
def home():
    return render_template('wine.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from HTML form
        fixed_acidity = float(request.form['fixed_acidity'])
        volatile_acidity = float(request.form['volatile_acidity'])
        citric_acid = float(request.form['citric_acid'])
        residual_sugar = float(request.form['residual_sugar'])
        chlorides = float(request.form['chlorides'])
        free_sulfur_dioxide = float(request.form['free_sulfur_dioxide'])
        total_sulfur_dioxide = float(request.form['total_sulfur_dioxide'])
        density = float(request.form['density'])
        pH = float(request.form['pH'])
        sulphates = float(request.form['sulphates'])
        alcohol = float(request.form['alcohol'])

        # Create dataframe
        input_data = pd.DataFrame([{
            'fixed acidity': fixed_acidity,
            'volatile acidity': volatile_acidity,
            'citric acid': citric_acid,
            'residual sugar': residual_sugar,
            'chlorides': chlorides,
            'free sulfur dioxide': free_sulfur_dioxide,
            'total sulfur dioxide': total_sulfur_dioxide,
            'density': density,
            'pH': pH,
            'sulphates': sulphates,
            'alcohol': alcohol
        }])

        # Make prediction
        prediction = model.predict(input_data)
        prediction = int(prediction[0])  # assume output is quality score

        return render_template('wine.html', result=f"Predicted Wine Quality: {prediction}")

    except Exception as e:
        return render_template('wine.html', result=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
