from flask import Flask, request, jsonify
import joblib
import json
import pandas as pd
import os

app = Flask(__name__)

# Load models and metadata on startup
try:
    # Use absolute or relative paths correctly, assuming gunicorn starts in pest_prediction_api/
    pest_type_model = joblib.load('models/pest_type_model.joblib')
    pest_risk_model = joblib.load('models/pest_risk_model.joblib')
    
    with open('model_metadata.json', 'r') as f:
        metadata = json.load(f)
except Exception as e:
    print(f"Failed to load models or metadata: {e}")
    pest_type_model = None
    pest_risk_model = None
    metadata = None

@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "Pest Prediction API running"})

@app.route('/predict', methods=['POST'])
def predict():
    if not pest_type_model or not pest_risk_model or not metadata:
        return jsonify({"error": "Models or metadata not loaded. Check server logs."}), 500
        
    data = request.json
    
    if not data:
         return jsonify({"error": "No JSON payload provided"}), 400
         
    # Expected fields
    required_fields = ["District", "Crop_Type", "Soil_Temperature", "Humidity", "Rainfall", "Season"]
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400
            
    # Prepare dataframe for prediction
    input_df = pd.DataFrame([{
        "District": data["District"],
        "Crop_Type": data["Crop_Type"],
        "Soil_Temperature": float(data["Soil_Temperature"]),
        "Humidity": float(data["Humidity"]),
        "Rainfall": float(data["Rainfall"]),
        "Season": data["Season"]
    }])
    
    try:
        pest_prediction = pest_type_model.predict(input_df)[0]
        risk_prediction = pest_risk_model.predict(input_df)[0]
        
        return jsonify({
            "prediction_pest": str(pest_prediction),
            "risk_level": str(risk_prediction),
            "pest_model_accuracy": metadata["pest_type_model_accuracy"],
            "risk_model_accuracy": metadata["pest_risk_model_accuracy"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
