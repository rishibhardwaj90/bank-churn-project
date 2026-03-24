from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load model & scaler
model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Load feature columns
feature_columns = pd.read_csv("data/processed/features.csv").columns

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_dict = {
            "CreditScore": int(request.form["creditscore"]),
            "Gender": int(request.form["gender"]),
            "Age": int(request.form["age"]),
            "Tenure": int(request.form["tenure"]),
            "Balance": float(request.form["balance"]),
            "NumOfProducts": int(request.form["products"]),
            "HasCrCard": int(request.form["card"]),
            "IsActiveMember": int(request.form["active"]),
            "EstimatedSalary": float(request.form["salary"]),
            "Geography_Germany": 1 if request.form["geography"] == "Germany" else 0,
            "Geography_Spain": 1 if request.form["geography"] == "Spain" else 0
        }

        input_df = pd.DataFrame([input_dict])

        # Match training columns
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)

        # Apply scaling only if needed
        if model.__class__.__name__ in ["LogisticRegression", "SVC"]:
            input_processed = scaler.transform(input_df)
        else:
            input_processed = input_df

        prediction = model.predict(input_processed)[0]
        probability = model.predict_proba(input_processed)[0][1]

        if probability > 0.7:
            risk = "High Risk"
        elif probability > 0.3:
            risk = "Medium Risk"
        else:
            risk = "Low Risk"

        return render_template("index.html",
                               prediction_text=f"Churn Probability: {round(probability*100,2)}%",
                               risk=risk)

    except Exception as e:
        return str(e)

port = int(os.environ.get("PORT", 10000))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port)
