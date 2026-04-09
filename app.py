from flask import Flask, request, jsonify
import pandas as pd
import traceback
from churn_model import predict_churn
app = Flask(__name__)
from flask import render_template
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form.to_dict()
        numeric_cols = [
            "CreditScore",
            "Age",
            "Tenure",
            "Balance",
            "NumOfProducts",
            "EstimatedSalary"
        ]
        for col in numeric_cols:
            if col in data:
                data[col] = float(data[col])
        input_df = pd.DataFrame([data])
        result = predict_churn(input_df)
        if isinstance(result, str):
            return result
        prediction = int(result["Churn_Prediction"].iloc[0])
        probability = float(result["Churn_Probability"].iloc[0])
        return f"""
        <h2>Prediction Result</h2>
        <p>Churn Prediction: {prediction}</p>
        <p>Churn Probability: {probability:.2f}</p>
        <a href="/">Go Back</a>
        """
    except Exception as e:
        return f"Error: {str(e)}"
if __name__ == "__main__":
    app.run(debug=True)
