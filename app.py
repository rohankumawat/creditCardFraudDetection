from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('models/LogisticRegression_model.pkl')
# Load the trained scaler
scaler = joblib.load('models/scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get features from the form
    input_data = request.form['features']  # features except Time and Amount
    time = float(request.form['time'])
    amount = float(request.form['amount'])

    # Convert the features into a list of floats
    features = [float(x) for x in input_data.split(',')]

    # Scale Time and Amount separately
    scaled_time = scaler.transform([[time]])[0][0]  # Scaling Time
    scaled_amount = scaler.transform([[amount]])[0][0]  # Scaling Amount

    # Append the scaled Time and Amount to the features
    features.append(scaled_time)
    features.append(scaled_amount)

    # Make prediction using the trained model
    prediction = model.predict([features])

    # Return result
    result = 'Fraud' if prediction[0] == 1 else 'Not Fraud'

    return render_template('index.html', prediction_text=f'Prediction: {result}')


if __name__ == "__main__":
    app.run(debug=True)
