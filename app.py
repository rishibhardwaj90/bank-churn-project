from flask import Flask, request, jsonify
import pandas as pd
import traceback
from churn_model import predict_churn
app = Flask(__name__)
from flask import render_template
@app.route('/')
def home():
    return render_template("index.html")
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        input_df = pd.DataFrame([data])
        result = predict_churn(input_df)
        if isinstance(result, str):
            return jsonify({"error": result})
        output = {
            "Churn_Prediction": int(result["Churn_Prediction"].iloc[0]),
            "Churn_Probability": float(result["Churn_Probability"].iloc[0])
        }
        return jsonify(output)
    except Exception:
        return jsonify({"error": traceback.format_exc()})
if __name__ == "__main__":
    app.run(debug=True)