import pandas as pd
import joblib
import os
import traceback

def predict_churn(InputData):
    try:
        # 1. Validate Input
        if not isinstance(InputData, pd.DataFrame):
            return "Error: InputData must be a pandas DataFrame"

        # 2. Load Feature Columns
        feature_path = "data/processed/features.csv"
        if not os.path.exists(feature_path):
            return "Error: features.csv not found"

        feature_df = pd.read_csv(feature_path)
        feature_columns = feature_df.columns

        # 3. Preprocess Input
        InputData = InputData.drop(columns=["CustomerId"], errors="ignore")

        if "Gender" in InputData.columns:
            InputData["Gender"] = InputData["Gender"].map({"Male": 1, "Female": 0})

        if "Geography" in InputData.columns:
            InputData = pd.get_dummies(InputData, columns=["Geography"], drop_first=True)

        # 4. Align Columns
        InputData = InputData.reindex(columns=feature_columns, fill_value=0)

        # 5. Load Model
        model_path = "models/best_model.pkl"
        if not os.path.exists(model_path):
            return "Error: best_model.pkl not found"

        model = joblib.load(model_path)

        # 6. Prediction
        scaler_path = "models/scaler.pkl"

        if model.__class__.__name__ in ["LogisticRegression", "SVC"]:
            if not os.path.exists(scaler_path):
                return "Error: scaler.pkl not found"

            scaler = joblib.load(scaler_path)
            Input_scaled = scaler.transform(InputData)

            prediction = model.predict(Input_scaled)
            probability = model.predict_proba(Input_scaled)[:, 1]
        else:
            prediction = model.predict(InputData)
            probability = model.predict_proba(InputData)[:, 1]

        # 7. Output
        result = InputData.copy()
        result["Churn_Prediction"] = prediction
        result["Churn_Probability"] = probability

        return result[["Churn_Prediction", "Churn_Probability"]]

    except Exception:
        return traceback.format_exc()