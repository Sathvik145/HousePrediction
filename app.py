from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load('best_house_price_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(request.form[feature]) for feature in ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]]
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]
        return jsonify({"prediction": f"{prediction:.2f}"})
    except Exception as e:
        return jsonify({"prediction": f"Error: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True)