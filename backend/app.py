from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle
import os

app = Flask(__name__, static_folder="build", static_url_path="/")
CORS(app)

model_path = os.path.join(os.path.dirname(__file__), "churn_model.pkl")
model = pickle.load(open(model_path, "rb"))

@app.route('/')
def serve():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = [
            int(data['gender']),
            int(data['SeniorCitizen']),
            int(data['Partner']),
            int(data['Dependents']),
            int(data['tenure']),
            int(data['PhoneService']),
            int(data['MultipleLines']),
            int(data['InternetService']),
            int(data['OnlineSecurity']),
            int(data['OnlineBackup']),
            int(data['DeviceProtection']),
            int(data['TechSupport']),
            int(data['StreamingTV']),
            int(data['StreamingMovies']),
            int(data['Contract']),
            int(data['PaperlessBilling']),
            int(data['PaymentMethod']),
            float(data['MonthlyCharges']),
            float(data['TotalCharges'])
        ]
        prediction = model.predict([features])[0]
        return jsonify({'Churn': str(prediction)})
    except Exception as e:
        print("Prediction Error:", e)
        return jsonify({'error': str(e)})

@app.errorhandler(404)
def not_found(e):
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
