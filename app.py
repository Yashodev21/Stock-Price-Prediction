from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load models and scalers
models = {
    "NIFTY50": pickle.load(open("modell/nifty_model.pkl", "rb")),
    "TCS": pickle.load(open("modell/tcs_model.pkl", "rb")),
    "ITC": pickle.load(open("modell/itc_model.pkl", "rb")),
    "HYUNDAI": pickle.load(open("modell/hyundai_model.pkl", "rb")),
    "HAL": pickle.load(open("modell/hal_model.pkl", "rb"))
}

scalers = {
    "NIFTY50": pickle.load(open("scalers/nifty_scaler.pkl", "rb")),
    "TCS": pickle.load(open("scalers/tcs_scaler.pkl", "rb")),
    "HAL": pickle.load(open("scalers/hal_scaler.pkl", "rb")),
    "HYUNDAI": pickle.load(open("scalers/hyundai_scaler.pkl", "rb")),
    "ITC": pickle.load(open("scalers/itc_scaler.pkl", "rb")),
}

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        stock = request.form['stock']
        open_price = float(request.form['open'])
        high = float(request.form['high'])
        low = float(request.form['low'])
        volume = float(request.form['volume'])

        # Backend validation
        if any(x <= 0 for x in [open_price, high, low]) or volume < 0:
            return render_template('index.html', prediction="Error: Open, High, and Low must be greater than 0. Volume cannot be negative.")

        model = models[stock]
        scaler = scalers[stock]

        input_data = np.array([[open_price, high, low, volume]])
        scaled_input = scaler.transform(input_data)
        predicted_close = model.predict(scaled_input)[0]

        recent = {
            "stock": stock,
            "open": open_price,
            "high": high,
            "low": low,
            "volume": volume,
            "predicted": round(predicted_close, 2)
        }

        return render_template('index.html', prediction=recent)

    except ValueError:
        return render_template('index.html', prediction="Error: Please enter valid numeric values only.")



if __name__ == "__main__":
    app.run(debug=True)
