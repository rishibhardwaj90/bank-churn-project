from flask import Flask, render_template, request
import joblib
import numpy as np
import os
import logging

app = Flask(__name__)

# Logging setup
logging.basicConfig(level=logging.INFO)

# Load model
model_path = os.path.join("models", "best_model.pkl")
model = joblib.load(model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get inputs
        credit_score = request.form.get('credit_score')
        age = request.form.get('age')
        balance = request.form.get('balance')
        gender = request.form.get('gender')

        # Validation
        if not all([credit_score, age, balance, gender]):
            return render_template('index.html', error="All fields are required")

        credit_score = int(credit_score)
        age = int(age)
        balance = float(balance)
        gender = int(gender)

        # Range validation
        if age < 18 or age > 100:
            return render_template('index.html', error="Invalid age")

        if credit_score < 300 or credit_score > 900:
            return render_template('index.html', error="Invalid credit score")

        # Model input
        features = np.array([[credit_score, age, balance, gender]])

        prediction = model.predict(features)[0]

        result = "Customer will churn !!" if prediction == 1 else "Customer will stay."

        logging.info(f"Prediction success: {features} → {prediction}")

        return render_template('index.html', result=result)

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return render_template('index.html', error="Something went wrong")

if __name__ == "__main__":
    app.run(debug=True)
