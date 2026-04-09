from flask import Flask, render_template, request, jsonify
import churn_model
app = Flask(__name__)
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict_churn", methods=["POST"])
def predict():
    data = request.get_json()
    input_data = {
        "CreditScore": data["CreditScore"],
        "Gender": data["Gender"],
        "Geography": data["Geography"],
        "Tenure": data["Tenure"],
        "Balance": data["Balance"],
        "NumOfProducts": data["NumOfProducts"],
        "HasCrCard": data["HasCrCard"],
        "IsActiveMember": data["IsActiveMember"],
        "EstimatedSalary": data["EstimatedSalary"]
    }
    prediction, probability = predict_churn.predict(input_data)
    return jsonify({
        "Churn_Prediction": int(prediction),
        "Churn_Probability": float(probability)
    })
if __name__ == "__main__":
    app.run(debug=True)