from flask import Flask, request, jsonify
import joblib
import pandas as pd

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
            "risk_level": str(risk_prediction)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
