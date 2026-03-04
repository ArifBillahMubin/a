from flask import Flask, render_template, request
import numpy as np
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load saved model and scaler
# Ensure these files are in the same directory as app.py
try:
    model = joblib.load("ev_model.pkl")
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    print("Error: .pkl files not found. Run your training script first.")

# Define the exact feature names used during training (to ensure correct order)
FEATURES = [
    'Trip Distance', 'Time of Day', 'Day of the Week', 'Longitude', 'Latitude',
    'Speed', 'Current', 'Total Voltage', 'Maximum Cell Temperature of Battery',
    'Minimum Cell Temperature of Battery', 'Trip Time Length'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect all inputs from the form in the correct order
        input_data = []
        for feature in FEATURES:
            # We use the 'name' attribute from index.html to fetch values
            val = float(request.form.get(feature))
            input_data.append(val)

        # Convert to numpy array and reshape for the scaler (1 row, N columns)
        input_array = np.array(input_data).reshape(1, -1)

        # IMPORTANT: Scale the input using the same scaler used during training
        input_scaled = scaler.transform(input_array)

        # Make prediction
        prediction = model.predict(input_scaled)

        # Return the result back to the HTML page
        result_text = f"Estimated Energy Consumption: {prediction[0]:.4f} kWh"
        return render_template('index.html', prediction_text=result_text)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

