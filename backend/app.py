from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle
import os

app = Flask(__name__, static_folder="build", static_url_path="/")
CORS(app)

model = pickle.load(open("churn_model.pkl", "rb"))

@app.route('/')
def serve():
    return send_from_directory(app.static_folder, 'index.html')

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

@app.errorhandler(404)
def not_found(e):
    return send_from_directory(app.static_folder, 'index.html')

port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port)
