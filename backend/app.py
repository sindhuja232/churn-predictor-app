from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os

app = Flask(__name__)
CORS(app)

model = pickle.load(open("churn_model.pkl", "rb"))

@app.route('/')
def home():
    return "Churn Prediction API is running"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = [
            data['gender'],
            data['SeniorCitizen'],
            data['Partner'],
            data['Dependents'],
            data['tenure'],
            data['PhoneService'],
            data['MultipleLines'],
            data['InternetService'],
            data['OnlineSecurity'],
            data['OnlineBackup'],
            data['DeviceProtection'],
            data['TechSupport'],
            data['StreamingTV'],
            data['StreamingMovies'],
            data['Contract'],
            data['PaperlessBilling'],
            data['PaymentMethod'],
            float(data['MonthlyCharges']),
            float(data['TotalCharges'])
        ]
        prediction = model.predict([features])[0]
        return jsonify({'Churn': str(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)})

port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port, debug=True)
