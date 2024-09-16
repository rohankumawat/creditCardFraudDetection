from flask import Flask, request, jsonify, render_template
import joblib
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

# Load environment variables from .env
load_dotenv()

# Set your OpenAI API key from the environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

# Load the trained model
logistic_regression_model = joblib.load('models/LogisticRegression_model.pkl')
random_forest_model = joblib.load('models/RandomForest_model.pkl')
xgboost_model = joblib.load('models/XGBoost_model.pkl')

# Load the trained scaler
scaler = joblib.load('models/scaler.pkl')

# Initialize the OpenAI language model
llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")

# Define the function for generating explanations using LangChain
def generate_explanation(features, prediction, model_name):
    """
    Generate a natural language explanation for the model's prediction using LangChain.
    """
    # Define the system message and prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Answer all questions to the best of your ability.",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    # Create the template for features and prediction
    template = (
        "The transaction has the following features: {features}. "
        "The model predicted {prediction}. Explain why the {model_name} made this prediction."
    )

    # Format the template with actual features and prediction
    message_content = template.format(features=features, prediction=prediction, model_name=model_name)

    # Invoke the model to generate the explanation
    response = llm.invoke([HumanMessage(content=message_content)])

    # Return the content of the response
    return response.content

# define the risk category based on the fraud probability
def risk_category(probability):
    if probability < 0.1:
        return 'Low'
    elif probability < 0.5:
        return 'Medium'
    else:
        return 'High'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get features from the form
    input_data = request.form['features']  # features except Time and Amount
    time = float(request.form['time'])
    amount = float(request.form['amount'])
    model_choice = request.form['model']

    # Convert the features into a list of floats
    features = [float(x) for x in input_data.split(',')]

    # Scale Time and Amount separately
    scaled_time = scaler.transform([[time]])[0][0]  # Scaling Time
    scaled_amount = scaler.transform([[amount]])[0][0]  # Scaling Amount

    # Append the scaled Time and Amount to the features
    features.append(scaled_time)
    features.append(scaled_amount)

    if model_choice == 'Logistic Regression':
        model = logistic_regression_model
        model_name = 'Logistic Regression'
    elif model_choice == 'Random Forest':
        model = random_forest_model
        model_name = 'Random Forest'
    else:
        model = xgboost_model
        model_name = 'XGBoost'

    # Make prediction using the trained model
    prediction = model.predict([features])[0]
    fraud_probability = model.predict_proba([features])[0][1]
    prediction_text = 'Fraud' if prediction == 1 else 'Not Fraud'

    # Return result
    # result = 'Fraud' if prediction[0] == 1 else 'Not Fraud'

    # Generate explanation using OpenAI
    explanation = generate_explanation(features, prediction_text, model_name)

    return render_template('index.html', 
                           prediction_text=f'Prediction: {prediction_text} (Probablity: {fraud_probability:.2f})', 
                           explanation_text=f'Explanation: {explanation}')


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # Use PORT from environment or default to 8080
    app.run(host="0.0.0.0", port=port, debug=True)  # Run the app