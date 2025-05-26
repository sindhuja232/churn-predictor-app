import os
import pickle
import numpy as np
from flask import Flask, request, jsonify
import xgboost as xgb

app = Flask(__name__)

# Load the trained model
model = xgb.XGBClassifier()
model.load_model("churn_model.json")

# Load encoders
with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# Define the expected input features
input_features = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
    'PaymentMethod', 'MonthlyCharges', 'TotalCharges'
]

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        print("Received input:", data)

        features = []
        for col in input_features:
            value = data.get(col)

            if col in ['MonthlyCharges', 'TotalCharges']:
                value = float(value)
            elif col in ['tenure', 'SeniorCitizen']:
                value = int(value)
            elif col in encoders:
                print(f"Encoding {col}: {value}")
                value = encoders[col].transform([value])[0]

            features.append(value)

        print("Processed features:", features)

        input_array = np.array(features).reshape(1, -1)
        prediction = model.predict(input_array)[0]

        print("Prediction result:", prediction)
        return jsonify({"Churn": str(prediction)})

    except Exception as e:
        print("Prediction error:", str(e))
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
