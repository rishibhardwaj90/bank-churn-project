from flask import Flask, render_template, request, jsonify
import pandas as pd
import churn_model
app = Flask(__name__)
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict_churn", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        input_df = pd.DataFrame([data])
        result = churn_model.predict_churn(input_df)
        if isinstance(result, str):
            return jsonify({
                "error": result
            })
        prediction = int(
            result["Churn_Prediction"].iloc[0]
        )
        probability = float(
            result["Churn_Probability"].iloc[0]
        )
        return jsonify({
            "Churn_Prediction": prediction,
            "Churn_Probability": probability
        })
    except Exception as e:
        return jsonify({
            "error": str(e)
        })
if __name__ == "__main__":
    app.run(debug=True)