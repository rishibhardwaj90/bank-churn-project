from flask import Flask, render_template, request, jsonify
import pandas as pd
import churn_model
import os
app = Flask(__name__)
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict_churn", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if data["Gender"] == "Male":
            data["Gender"] = 1
        else:
            data["Gender"] = 0
        geo_map = {"France":0, "Germany":1, "Spain":2}
        data["Geography"] = geo_map.get(data["Geography"],0)
        input_df = pd.DataFrame([data])
        result = churn_model.predict_churn(input_df)
        prediction = int(result["Churn_Prediction"].iloc[0])
        probability = float(result["Churn_Probability"].iloc[0])
        return jsonify({
            "Churn_Prediction": prediction,
            "Churn_Probability": probability
        })
    except Exception as e:
        return jsonify({"error": str(e)})
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)