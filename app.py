from flask import Flask, render_template, request, jsonify
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
from datetime import datetime
import json

app = Flask(__name__)

# Load the trained model
with open('Saved_Models/linear.pkl', 'rb') as f:
    model = pickle.load(f)

# Function to generate future input features
def generate_future_inputs(n_days, weather_defaults):
    future_data = []
    current_time = datetime.now()

    for i in range(n_days * 24):  # hourly rows
        timestamp = current_time + timedelta(hours=i)
        entry = {
            "DC_POWER": 0,
            "DAILY_YIELD": 0,
            "AMBIENT_TEMPERATURE": weather_defaults["ambient_temp"],
            "MODULE_TEMPERATURE": weather_defaults["module_temp"],
            "IRRADIATION": weather_defaults["irradiation"],
            "MONTH": timestamp.month,
            "DAY": timestamp.day,
            "HOUR": timestamp.hour,
            "DAY_OF_WEEK": timestamp.weekday(),
            "TIME_OF_DAY": 1 if 6 <= timestamp.hour <= 18 else 0,
            "DATE": timestamp.date().strftime('%Y-%m-%d')
        }
        future_data.append(entry)

    return pd.DataFrame(future_data)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/introduction')
def introduction():
    return render_template('introduction.html')

@app.route('/implementation')
def implementation():
    # Model performance data for charts
    performance_data = {
        'models': ['Linear Regression', 'SVR', 'Decision Tree', 'Stacking', 'Voting'],
        'mae': [45.32, 38.76, 25.43, 22.15, 23.78],
        'mse': [3200.45, 2800.12, 1200.78, 980.45, 1050.32],
        'r2': [0.82, 0.85, 0.92, 0.94, 0.93]
    }
    return render_template('implementation.html', performance_data=json.dumps(performance_data))

@app.route('/deployment')
def deployment():
    return render_template('deployment.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user inputs
        n_days = int(request.form['days'])
        ambient_temp = float(request.form['ambient_temp'])
        module_temp = float(request.form['module_temp'])
        irradiation = float(request.form['irradiation'])

        # Validate inputs
        if n_days < 1:
            return jsonify({'error': 'Number of days must be at least 1'})
        if irradiation < 0 or irradiation > 1:
            return jsonify({'error': 'Irradiation must be between 0 and 1'})

        # Define weather defaults
        weather_defaults = {
            "ambient_temp": ambient_temp,
            "module_temp": module_temp,
            "irradiation": irradiation
        }

        # Generate input data
        future_weather = generate_future_inputs(n_days, weather_defaults)

        # Predict using model
        feature_cols = [
            "DC_POWER", "DAILY_YIELD", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE",
            "IRRADIATION", "MONTH", "DAY", "HOUR", "DAY_OF_WEEK", "TIME_OF_DAY"
        ]
        future_weather["Predicted_AC_POWER"] = model.predict(future_weather[feature_cols])

        # Group by day
        daily_summary = future_weather.groupby("DATE")["Predicted_AC_POWER"].sum().reset_index()
        daily_summary["Predicted_Energy_kWh"] = daily_summary["Predicted_AC_POWER"] / 1000

        # Prepare response
        daily_predictions = [
            {'date': row['DATE'], 'energy': row['Predicted_Energy_kWh']}
            for _, row in daily_summary.iterrows()
        ]
        total_energy = daily_summary["Predicted_Energy_kWh"].sum()

        return jsonify({
            'n_days': n_days,
            'daily_predictions': daily_predictions,
            'total_energy': total_energy
        })
    except ValueError as e:
        return jsonify({'error': 'Invalid input: Please ensure all fields are valid numbers'})
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'})
    
if __name__ == '__main__':
    app.run(debug=True)