from flask import Flask, render_template, request, jsonify
import pandas as pd
import churn_model
app = Flask(__name__)
@app.route("/")
def home():
    return render_template("index.html")
def generate_explanation(df):
    reasons = []
    if df["Age"].iloc[0] > 50:
        reasons.append("Higher age increases churn risk")
    if df["Balance"].iloc[0] > 100000:
        reasons.append("High balance customer")
    if df["NumOfProducts"].iloc[0] <= 1:
        reasons.append("Low product usage")
    if len(reasons) == 0:
        return "No strong churn indicators"
    return ", ".join(reasons)
@app.route("/predict_churn", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        input_df = pd.DataFrame([data])
        result = churn_model.predict_churn(input_df)
        prediction = int(result["Churn_Prediction"].iloc[0])
        probability = float(result["Churn_Probability"].iloc[0])
        explanation = generate_explanation(input_df)
        return jsonify({
            "Churn_Prediction": prediction,
            "Churn_Probability": probability,
            "explanation": explanation
        })
    except Exception as e:
        return jsonify({"error": str(e)})
if __name__ == "__main__":
    app.run(debug=True)