from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

pest_type_model = joblib.load("models/pest_type_model.joblib")
pest_risk_model = joblib.load("models/pest_risk_model.joblib")

@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "Pest Prediction API running"})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    if not data:
         return jsonify({"error": "No JSON payload provided"}), 400
         
    # Expected fields
    required_fields = ["District", "Crop_Type", "Soil_Temperature", "Humidity", "Rainfall", "Season"]
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400
            
    # Map categorical values
    district_map = {
        "Thanjavur":0,
        "Salem":1,
        "Madurai":2
    }
    crop_map = {
        "Rice":0,
        "Maize":1,
        "Groundnut":2
    }
    season_map = {
        "Kharif":0,
        "Rabi":1,
        "Summer":2
    }

    district = district_map.get(data["District"], 0)
    crop = crop_map.get(data["Crop_Type"], 0)
    soil_temp = float(data["Soil_Temperature"])
    humidity = float(data["Humidity"])
    rainfall = float(data["Rainfall"])
    season = season_map.get(data["Season"], 0)

    # Prepare array for prediction
    X = np.array([[district, crop, soil_temp, humidity, rainfall, season]])
    
    try:
        pest_prediction = pest_type_model.predict(X)[0]
        risk_prediction = pest_risk_model.predict(X)[0]
        
        return jsonify({
            "prediction_pest": str(pest_prediction),
            "risk_level": str(risk_prediction)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
